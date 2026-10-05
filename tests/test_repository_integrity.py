import csv
import re
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
                source = (ROOT / "data" / filename).read_text(encoding="utf-8")
                self.assertEqual(len(re.findall(r"np\.array\(", source)), row_count)

    def test_released_checkpoint_inventory(self):
        generators = {"MLPD.pth", "R-MLPD.pth", "CVAE.pt", "R-CVAE.pt"}
        self.assertEqual({path.name for path in (ROOT / "trained_pt").iterdir()}, generators)
        self.assertEqual(len(list((ROOT / "trained_pth").glob("fold_*_model.pth"))), 5)

    def test_condition_semantics_are_explicit_in_training_sources(self):
        descriptor_source = (ROOT / "descriptor_utils.py").read_text(encoding="utf-8")
        self.assertIn('conductivity_conditions = self.df["k"]', descriptor_source)
        self.assertIn("return formulations, conductivity_conditions", descriptor_source)

        scripts = (
            "model/MLPD/train.py",
            "model/R-MLPD/train.py",
            "model/CVAE/train.py",
            "model/R-CVAE/train.py",
        )
        for relative_path in scripts:
            with self.subTest(script=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn("formulations_raw, Y_raw", source)
                self.assertIn('"condition": "Y[:,0] = ionic conductivity k"', source)

    def test_standard_and_routed_cvae_import_distinct_classes(self):
        standard = (ROOT / "model/CVAE/train.py").read_text(encoding="utf-8")
        routed = (ROOT / "model/R-CVAE/train.py").read_text(encoding="utf-8")
        self.assertIn("from model import CVAE", standard)
        self.assertNotIn("CVAEWithRouting", standard)
        self.assertIn("from model_r import CVAEWithRouting", routed)


if __name__ == "__main__":
    unittest.main()
