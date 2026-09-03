import importlib.util
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ModelInterfaceTests(unittest.TestCase):
    def test_mlp_diffusion_preserves_formulation_shape(self):
        module = load_module("mlpd_model", "model/MLPD/model.py")
        model = module.MLPDiffusionModel(input_dim=11, cond_dim=1)
        output = model(torch.zeros(4, 11), torch.zeros(4, 1), torch.zeros(4, 1))
        self.assertEqual(tuple(output.shape), (4, 11))

    def test_routed_mlp_diffusion_preserves_formulation_shape(self):
        module = load_module("r_mlpd_model", "model/R-MLPD/model.py")
        model = module.MLPDiffusionModelWithRouting(input_dim=11, cond_dim=1)
        output = model(torch.zeros(4, 11), torch.zeros(4, 1), torch.zeros(4, 1))
        self.assertEqual(tuple(output.shape), (4, 11))

    def test_cvae_forward_contract(self):
        module = load_module("cvae_model", "model/CVAE/model.py")
        model = module.CVAE(cond_dim=1, out_dim=11)
        reconstruction, mean, log_variance = model(torch.zeros(4, 11), torch.zeros(4, 1))
        self.assertEqual(tuple(reconstruction.shape), (4, 11))
        self.assertEqual(tuple(mean.shape), (4, 16))
        self.assertEqual(tuple(log_variance.shape), (4, 16))

    def test_routed_cvae_forward_contract(self):
        module = load_module("r_cvae_model", "model/R-CVAE/model_r.py")
        model = module.CVAEWithRouting(cond_dim=1, out_dim=11)
        reconstruction, mean, log_variance = model(torch.zeros(4, 11), torch.zeros(4, 1))
        self.assertEqual(tuple(reconstruction.shape), (4, 11))
        self.assertEqual(tuple(mean.shape), (4, 16))
        self.assertEqual(tuple(log_variance.shape), (4, 16))


if __name__ == "__main__":
    unittest.main()
