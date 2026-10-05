from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


SOLVENT_ALIASES = {
    "3-Me-2-O": "3-Me-2-Oxazolidinone",
    "3-MeSul": "3-MeSulfolane",
    "Ethylb": "Ethylbenzene",
    "Ethyld": "Ethyldiglyme",
    "Ethylm": "Ethylmonoglyme",
    "Freon": "Freon 11",
    "MC": "Methylene chloride",
    "g-Buty": "g-Butyrolactone",
}


def load_mo_descriptors(path, aliases=None):
    descriptors = {}
    text = Path(path).read_text(encoding="utf-8")
    dictionary_entries = re.findall(
        r"['\"]([^'\"]+)['\"]\s*:\s*np\.array\(\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]\)",
        text,
    )
    if dictionary_entries:
        for name, first, second in dictionary_entries:
            if aliases is not None:
                name = aliases.get(name, name)
            descriptors[name] = np.asarray([first, second], dtype=float)
        return descriptors
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        name = " ".join(fields[2:])
        if aliases is not None:
            name = aliases.get(name, name)
        descriptors[name] = np.asarray(fields[:2], dtype=float)
    return descriptors


def load_descriptor_libraries(data_dir):
    data_dir = Path(data_dir)
    salts = load_mo_descriptors(data_dir / "salt_MO.txt")
    solvents = load_mo_descriptors(
        data_dir / "solvent_MO.txt", aliases=SOLVENT_ALIASES
    )
    return salts, solvents


class ElectrolyteDataset:
    def __init__(self, df, solvent_features_dict, salt_features_dict, normalize_salt_solvent=True):
        self.df = df.copy()
        self.solvent_features_dict = solvent_features_dict
        self.salt_features_dict = salt_features_dict
        self.normalize_salt_solvent = normalize_salt_solvent
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.tc_scaler = MinMaxScaler()
        self.salt_scaler = MinMaxScaler()
        self.solvent_scaler = MinMaxScaler()
        self._fit_encoders()

    def _fit_encoders(self):
        self.ohe.fit(self.df[["c units", "solvent ratio type"]])
        self.tc_scaler.fit(self.df[["T", "c"]])
        salt_mat = np.vstack(list(self.salt_features_dict.values()))
        solvent_mat = np.vstack(list(self.solvent_features_dict.values()))
        if self.normalize_salt_solvent:
            self.salt_scaler.fit(salt_mat)
            self.solvent_scaler.fit(solvent_mat)
        self.salt_dim = salt_mat.shape[1]
        self.solv_dim = solvent_mat.shape[1]

    def _get_salt_vector(self, salt):
        vector = self.salt_features_dict.get(salt, np.zeros(self.salt_dim))
        if self.normalize_salt_solvent:
            vector = self.salt_scaler.transform([vector])[0]
        return vector

    def _compute_weighted_solvent_vector(self, row):
        vector = np.zeros(self.solv_dim)
        for index in range(1, 5):
            solvent = row.get(f"solvent_{index}")
            ratio = row.get(f"ratio_{index}")
            if pd.notna(solvent) and pd.notna(ratio):
                vector += ratio * self.solvent_features_dict.get(
                    solvent, np.zeros(self.solv_dim)
                )
        if self.normalize_salt_solvent:
            vector = self.solvent_scaler.transform([vector])[0]
        return vector

    def get_features_and_targets(self):
        temperature_concentration = self.tc_scaler.transform(self.df[["T", "c"]])
        categorical = self.ohe.transform(
            self.df[["c units", "solvent ratio type"]]
        )
        salt_vectors = np.vstack(
            [self._get_salt_vector(salt) for salt in self.df["salt"]]
        )
        solvent_vectors = np.vstack(
            [
                self._compute_weighted_solvent_vector(row)
                for _, row in self.df.iterrows()
            ]
        )
        formulations = np.concatenate(
            [temperature_concentration, categorical, salt_vectors, solvent_vectors],
            axis=1,
        )
        conductivity_conditions = self.df["k"].to_numpy().reshape(-1, 1)
        return formulations, conductivity_conditions

    def decode(self, formulations):
        temperature_concentration = self.tc_scaler.inverse_transform(
            formulations[:, :2]
        )
        categorical_dimension = self.ohe.transform(
            [["mol/kg", "w"]]
        ).shape[1]
        categorical = formulations[:, 2 : 2 + categorical_dimension]
        categorical_labels = self.ohe.inverse_transform(categorical)
        salt_start = 2 + categorical_dimension
        salt_end = salt_start + self.salt_dim
        solvent_end = salt_end + self.solv_dim
        salt_vectors = formulations[:, salt_start:salt_end]
        solvent_vectors = formulations[:, salt_end:solvent_end]
        if self.normalize_salt_solvent:
            salt_vectors = self.salt_scaler.inverse_transform(salt_vectors)
            solvent_vectors = self.solvent_scaler.inverse_transform(solvent_vectors)
        return (
            temperature_concentration,
            categorical_labels,
            salt_vectors,
            solvent_vectors,
        )
