import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

if np is not None:
    from search_salt_solvent import search_best_2solvent, search_best_salt


@unittest.skipIf(np is None, "NumPy is not installed")
class MolecularReadoutTests(unittest.TestCase):
    def test_salt_search_recovers_exact_descriptor(self):
        descriptors = {
            "salt_a": np.array([-5.0, 1.0]),
            "salt_b": np.array([-3.0, 2.0]),
        }
        match, distance = search_best_salt(np.array([-3.0, 2.0]), descriptors)
        self.assertEqual(match, "salt_b")
        self.assertAlmostEqual(distance, 0.0)

    def test_binary_solvent_search_recovers_midpoint(self):
        descriptors = {
            "solvent_a": np.array([0.0, 0.0]),
            "solvent_b": np.array([2.0, 2.0]),
        }
        match, distance = search_best_2solvent(np.array([1.0, 1.0]), descriptors)
        self.assertEqual(match[0], "solvent_a")
        self.assertAlmostEqual(match[1], 0.5)
        self.assertEqual(match[2], "solvent_b")
        self.assertAlmostEqual(match[3], 0.5)
        self.assertAlmostEqual(distance, 0.0)


if __name__ == "__main__":
    unittest.main()
