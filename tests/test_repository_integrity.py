import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryIntegrityTests(unittest.TestCase):
    def test_training_data_has_expected_schema_and_rows(self):
        path = ROOT / "data" / "training_data.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        self.assertEqual(len(rows), 13_302)
        self.assertTrue({
            "k", "T", "c", "salt", "c units", "solvent ratio type",
            "solvent_1", "ratio_1",
        }.issubset(reader.fieldnames or []))

    def test_descriptor_libraries_have_reported_sizes(self):
        for filename, row_count in {"salt_MO.txt": 13, "solvent_MO.txt": 38}.items():
            with self.subTest(filename=filename):
                rows = [line for line in (ROOT / "data" / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
                self.assertEqual(len(rows), row_count)

    def test_released_checkpoint_inventory(self):
        generators = {"MLPD.pth", "R-MLPD.pth", "CVAE.pt", "R-CVAE.pt"}
        self.assertEqual({path.name for path in (ROOT / "trained_pt").iterdir()}, generators)
        self.assertEqual(len(list((ROOT / "trained_pth").glob("fold_*_model.pth"))), 5)


if __name__ == "__main__":
    unittest.main()
