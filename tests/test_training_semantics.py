import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from descriptor_utils import ElectrolyteDataset, load_descriptor_libraries
except ImportError:
    ElectrolyteDataset = None
    load_descriptor_libraries = None


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(ElectrolyteDataset is None, "Scientific Python dependencies are not installed")
class TrainingSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / "data" / "training_data.csv", na_values=["null"])
        cls.salts, cls.solvents = load_descriptor_libraries(ROOT / "data")

    def test_y_first_column_is_conductivity_condition(self):
        dataset = ElectrolyteDataset(self.data, self.solvents, self.salts)
        formulations, Y = dataset.get_features_and_targets()
        self.assertEqual(Y.shape, (len(self.data), 1))
        np.testing.assert_allclose(Y[:, 0], self.data["k"].to_numpy())
        self.assertEqual(formulations.shape[0], len(self.data))

    def test_descriptor_libraries_cover_training_identifiers(self):
        self.assertEqual(set(self.data["salt"]), set(self.salts))
        observed = set()
        for index in range(1, 5):
            observed.update(self.data[f"solvent_{index}"].dropna())
        self.assertEqual(observed, set(self.solvents))


if __name__ == "__main__":
    unittest.main()
