"""
Unit tests for EVFaultDualHeadModel architecture and pretrained weight loading behavior.
Uses mocks when network or cache is unavailable.
"""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))


class TestModelArchitectureAndPretrainedLoading(unittest.TestCase):

    def test_model_unpretrained_initialization(self):
        """Model with pretrained=False initializes without attempting download."""
        try:
            import torch
            import torch.nn as nn
            from src.model import EVFaultDualHeadModel
        except ImportError:
            self.skipTest("torch not installed in current test environment")

        model = EVFaultDualHeadModel(pretrained=False)
        self.assertIsInstance(model.condition_head, nn.Linear)
        self.assertEqual(model.condition_head.out_features, 2)
        self.assertEqual(model.component_head.out_features, 4)
        self.assertEqual(model.shared_fc[0].in_features, 576)

    def test_model_pretrained_official_weights_success(self):
        """Model with pretrained=True uses torchvision MobileNet_V3_Small_Weights.DEFAULT."""
        try:
            import torch
            import torchvision.models as models
            from src.model import EVFaultDualHeadModel
        except ImportError:
            self.skipTest("torch not installed in current test environment")

        mock_mobilenet = MagicMock()
        mock_mobilenet.features = MagicMock()
        mock_mobilenet.avgpool = MagicMock()

        with patch("torchvision.models.mobilenet_v3_small", return_value=mock_mobilenet) as mock_m:
            model = EVFaultDualHeadModel(pretrained=True)
            mock_m.assert_called_once_with(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    def test_model_pretrained_failure_raises_runtime_error(self):
        """Model with pretrained=True raises RuntimeError if torchvision weights cannot be obtained."""
        try:
            import torch
            from src.model import EVFaultDualHeadModel
        except ImportError:
            self.skipTest("torch not installed in current test environment")

        with patch("torchvision.models.mobilenet_v3_small", side_effect=Exception("Network/cache error")):
            with patch("pathlib.Path.exists", return_value=False):
                with self.assertRaises(RuntimeError) as ctx:
                    EVFaultDualHeadModel(pretrained=True)
                self.assertIn("Failed to obtain official torchvision MobileNetV3-Small pretrained weights", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
