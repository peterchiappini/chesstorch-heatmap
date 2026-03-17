"""
ChessCNN: a small convolutional network that predicts move destinations
from 17-plane board encodings, with built-in Grad-CAM hook support.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChessCNN(nn.Module):
    """
    4-block CNN: (17, 8, 8) board tensor → 64-dim logits (one per square).

    Architecture
    ────────────
    Block 1:  17 →  64 channels   (3×3 conv, BatchNorm, ReLU)
    Block 2:  64 → 128 channels
    Block 3: 128 → 256 channels
    Block 4: 256 → 256 channels   ← Grad-CAM target layer

    All convolutions use padding=1 so the spatial size stays 8×8
    throughout, which keeps a clean 1-to-1 mapping between feature-map
    cells and board squares — exactly what we need for heatmaps.

    After the final block the features are flattened (256 × 8 × 8 = 16384)
    and projected to 64 logits via a single linear layer.
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Convolutional backbone ──
        self.block1 = self._make_block(17, 64)
        self.block2 = self._make_block(64, 128)
        self.block3 = self._make_block(128, 256)
        self.block4 = self._make_block(256, 256)  # Grad-CAM target

        # ── Two-headed classifier ──
        self.fc_from = nn.Linear(256 * 8 * 8, 64)
        self.fc_to = nn.Linear(256 * 8 * 8, 64)

        # ── Grad-CAM storage (per layer) ──
        self._gradcam_activations: dict[str, torch.Tensor] = {}
        self._gradcam_gradients: dict[str, torch.Tensor] = {}
        self._gradcam_hooks_installed = False

        self._layer_map = {
            "layer1": self.block1,
            "layer2": self.block2,
            "layer3": self.block3,
            "layer4": self.block4,
        }

    # ── Hook factories ──

    def _make_activation_hook(self, name: str):
        def hook(_module: nn.Module, _input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self._gradcam_activations[name] = output.detach()
        return hook

    def _make_gradient_hook(self, name: str):
        def hook(_module: nn.Module, _grad_input: tuple[torch.Tensor | None, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
            self._gradcam_gradients[name] = grad_output[0].detach()
        return hook

    # ── Public Grad-CAM helpers ──

    def install_gradcam_hooks(self) -> None:
        """Register forward/backward hooks for Grad-CAM. Only needs to be called once,
        typically after loading weights and calling .eval(). Kept separate from __init__
        so hooks don't fire during training."""
        if self._gradcam_hooks_installed:
            return
        for name, block in self._layer_map.items():
            block.register_forward_hook(self._make_activation_hook(name))
            block.register_full_backward_hook(self._make_gradient_hook(name))
        self._gradcam_hooks_installed = True

    def get_gradcam_activations(self, layer: str = "final") -> torch.Tensor:
        """Return the cached activations for the given layer."""
        if layer not in self._gradcam_activations:
            raise RuntimeError(f"No activations cached for '{layer}' — run a forward pass first.")
        return self._gradcam_activations[layer]

    def get_gradcam_gradients(self, layer: str = "final") -> torch.Tensor:
        """Return the cached gradients for the given layer."""
        if layer not in self._gradcam_gradients:
            raise RuntimeError(f"No gradients cached for '{layer}' — run a backward pass first.")
        return self._gradcam_gradients[layer]

    # ── Internals ──

    @staticmethod
    def _make_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Conv 3×3 → BatchNorm → ReLU.  Padding=1 keeps the 8×8 spatial size."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    # ── Forward pass ──

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 17, 8, 8) board tensor.
        Returns:
            (logits_from, logits_to) — each (batch, 64) raw logits.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)          # activations saved by hook
        x = x.flatten(start_dim=1)  # (batch, 256*8*8)
        return self.fc_from(x), self.fc_to(x)


def get_device() -> torch.device:
    """Pick the best available accelerator: MPS (Apple Silicon) → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
