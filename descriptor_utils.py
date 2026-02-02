import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class ElectrolyteDataset:
    def __init__(self, df, solvent_features_dict, salt_features_dict, normalize_salt_solvent=True):
        self.df = df.copy()
        self.solvent_features_dict = solvent_features_dict
        self.salt_features_dict = salt_features_dict
        self.normalize_salt_solvent = normalize_salt_solvent

        # Encoders / Scalers
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.tc_scaler = MinMaxScaler()
        self.salt_scaler = MinMaxScaler()
        self.solvent_scaler = MinMaxScaler()

        self._fit_encoders()

    def _fit_encoders(self):
        self.ohe.fit(self.df[['c units', 'solvent ratio type']])
        self.tc_scaler.fit(self.df[['T', 'c']])

        salt_mat = np.vstack([v for v in self.salt_features_dict.values()])
        solv_mat = np.vstack([v for v in self.solvent_features_dict.values()])
        if self.normalize_salt_solvent:
            self.salt_scaler.fit(salt_mat)
            self.solvent_scaler.fit(solv_mat)

        self.salt_dim = salt_mat.shape[1]
        self.solv_dim = solv_mat.shape[1]

    def _get_salt_vector(self, salt):
        vec = self.salt_features_dict.get(salt, np.zeros(self.salt_dim))
        if self.normalize_salt_solvent:
            vec = self.salt_scaler.transform([vec])[0]
        return vec

    def _compute_weighted_solvent_vector(self, row):
        vec = np.zeros(self.solv_dim)
        for i in range(1, 5):
            s = row.get(f'solvent_{i}')
            r = row.get(f'ratio_{i}')
            if pd.notna(s) and pd.notna(r):
                solvent_vec = self.solvent_features_dict.get(s, np.zeros(self.solv_dim))
                vec += r * solvent_vec
        if self.normalize_salt_solvent:
            vec = self.solvent_scaler.transform([vec])[0]
        return vec

    def get_features_and_targets(self):
        # Target k as condition
        ks = self.df['k'].values.reshape(-1, 1)

        # Target y: T, c, categorical, salt_vec, solvent_vec
        Tc = self.tc_scaler.transform(self.df[['T', 'c']])
        cat = self.ohe.transform(self.df[['c units', 'solvent ratio type']])
        salt_vecs = np.vstack([self._get_salt_vector(s) for s in self.df['salt']])
        solv_vecs = np.vstack([self._compute_weighted_solvent_vector(row) for _, row in self.df.iterrows()])
        y = np.concatenate([Tc, cat, salt_vecs, solv_vecs], axis=1)

        return ks, y

    def decode(self, y):
        # Recover T, c
        Tc = self.tc_scaler.inverse_transform(y[:, :2])

        # Recover categorical
        cat_dim = self.ohe.transform([['mol/kg', 'w']]).shape[1]
        cat_encoded = y[:, 2:2 + cat_dim]
        cat_labels = self.ohe.inverse_transform(cat_encoded)

        # Recover salt & solvent vectors
        salt_start = 2 + cat_dim
        salt_end = salt_start + self.salt_dim
        solv_end = salt_end + self.solv_dim

        salt_vecs = y[:, salt_start:salt_end]
        solv_vecs = y[:, salt_end:solv_end]

        if self.normalize_salt_solvent:
            salt_vecs = self.salt_scaler.inverse_transform(salt_vecs)
            solv_vecs = self.solvent_scaler.inverse_transform(solv_vecs)

        return Tc, cat_labels, salt_vecs, solv_vecs

