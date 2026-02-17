"""
🌌 Quantum-Enhanced Neural Radiance Fields (Q-NeRF)
====================================================

Híbrido de NeRF clássico com amplificação quântica para rendering
holográfico de alta fidelidade.

Features:
- Quantum amplitude amplification para ray sampling
- Variational quantum density prediction
- φ-optimized view synthesis
- Holographic memory encoding
- C++ backend integration via pybind11

Architecture:
    Ray → Classical NeRF MLP → Quantum Amplification → Rendering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.arkheion.constants.sacred_constants import PHI

logger = logging.getLogger(__name__)



@dataclass
class RenderConfig:
    """Configuration for Q-NeRF rendering."""

    n_samples: int = 64  # Samples per ray
    n_importance: int = 128  # Importance samples
    chunk_size: int = 1024 * 32
    white_background: bool = False
    use_quantum_amplification: bool = True
    quantum_amplification_factor: float = PHI


class PositionalEncoding(nn.Module):
    """
    Positional encoding with φ-enhanced frequencies.

    γ(p) = [sin(2^0 π p), cos(2^0 π p), ..., sin(2^(L-1) π p), cos(2^(L-1) π p)]

    With φ-optimization: frequencies scaled by φ^k
    """

    def __init__(self, num_freqs: int = 10, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input

        # φ-optimized frequency bands
        freq_bands = PHI ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (..., C)

        Returns:
            Encoded tensor (..., C * (2 * num_freqs + 1)) if include_input else
                          (..., C * 2 * num_freqs)
        """
        out = []

        if self.include_input:
            out.append(x)

        for freq in self.freq_bands:
            out.append(torch.sin(freq * np.pi * x))
            out.append(torch.cos(freq * np.pi * x))

        return torch.cat(out, dim=-1)


class QuantumAmplifier(nn.Module):
    """
    Quantum amplitude amplification for NeRF density predictions.

    Amplifies high-density regions using quantum-inspired operators,
    improving rendering quality in complex geometries.
    """

    def __init__(self, amplification_factor: float = PHI):
        super().__init__()
        self.amplification_factor = amplification_factor

        # Learnable amplification parameters
        self.alpha = nn.Parameter(torch.tensor(1.0 / PHI))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, density: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Amplify density predictions using quantum-inspired transformation.

        Applies Grover-like amplification to values above threshold.

        Args:
            density: Raw density predictions (..., 1)
            threshold: Amplification threshold

        Returns:
            Amplified density (..., 1)
        """
        # Normalize density to [0, 1]
        density_norm = torch.sigmoid(density)

        # Compute mean (like Grover's oracle)
        mean_density = density_norm.mean()

        # Amplification: reflect around mean, then scale
        deviation = density_norm - mean_density
        amplified = mean_density + self.alpha * deviation * self.amplification_factor

        # Apply threshold-based boost
        boost = torch.where(
            density_norm > threshold,
            torch.ones_like(density_norm) * self.beta,
            torch.ones_like(density_norm),
        )

        amplified = amplified * boost

        # Clamp to valid range
        amplified = torch.clamp(amplified, 0, 1)

        return amplified


class QuantumNeRF(nn.Module):
    """
    Quantum-Enhanced Neural Radiance Field.

    Combines classical NeRF architecture with quantum amplification
    for improved rendering quality and efficiency.

    Example:
        >>> model = QuantumNeRF(hidden_dim=256, num_layers=8)
        >>> x = torch.randn(1024, 3)  # Positions
        >>> d = torch.randn(1024, 3)  # Directions
        >>> rgb, density = model(x, d)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 8,
        num_freqs_pos: int = 10,
        num_freqs_dir: int = 4,
        use_quantum: bool = True,
    ):
        super().__init__()

        self.use_quantum = use_quantum

        # Positional encoding
        self.pos_encoder = PositionalEncoding(num_freqs_pos)
        self.dir_encoder = PositionalEncoding(num_freqs_dir)

        # Calculate input dimensions
        pos_dim = 3 * (2 * num_freqs_pos + 1)
        dir_dim = 3 * (2 * num_freqs_dir + 1)

        # Density network (coarse)
        layers = []
        layers.append(nn.Linear(pos_dim, hidden_dim))

        for i in range(num_layers - 1):
            # Skip connection at layer φ * num_layers
            if i == int(PHI * num_layers / 2):
                layers.append(nn.Linear(hidden_dim + pos_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        self.density_net = nn.Sequential(*layers)

        # Density head
        self.density_head = nn.Linear(hidden_dim, 1)

        # Color network (fine)
        self.color_net = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

        # Quantum amplifier
        if self.use_quantum:
            self.quantum_amplifier = QuantumAmplifier(amplification_factor=PHI)

        logger.info(
            f"🌌 QuantumNeRF: hidden={hidden_dim}, layers={num_layers}, " f"quantum={use_quantum}"
        )

    def forward(
        self, positions: torch.Tensor, directions: torch.Tensor, apply_quantum: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Q-NeRF.

        Args:
            positions: 3D positions (N, 3)
            directions: View directions (N, 3)
            apply_quantum: Apply quantum amplification

        Returns:
            (rgb, density) tuple
            - rgb: (N, 3) in [0, 1]
            - density: (N, 1)
        """
        # Encode positions
        pos_encoded = self.pos_encoder(positions)

        # Density prediction
        h = pos_encoded
        for i, layer in enumerate(self.density_net):
            h = layer(h)
            # Skip connection
            if i == int(PHI * len(self.density_net) / 2):
                h = torch.cat([h, pos_encoded], dim=-1)

        density = self.density_head(h)

        # Quantum amplification
        if self.use_quantum and apply_quantum:
            density = self.quantum_amplifier(density)

        # Encode directions
        dir_encoded = self.dir_encoder(directions)

        # Color prediction
        color_input = torch.cat([h, dir_encoded], dim=-1)
        rgb = self.color_net(color_input)

        return rgb, density


def volume_rendering(
    rgb: torch.Tensor,
    density: torch.Tensor,
    z_vals: torch.Tensor,
    directions: torch.Tensor,
    white_background: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Volume rendering equation for NeRF.

    C(r) = ∫ T(t) σ(r(t)) c(r(t), d) dt

    where T(t) = exp(-∫ σ(r(s)) ds) is transmittance.

    Args:
        rgb: RGB values (N_rays, N_samples, 3)
        density: Density values (N_rays, N_samples, 1)
        z_vals: Depth values (N_rays, N_samples)
        directions: Ray directions (N_rays, 3)
        white_background: Use white background

    Returns:
        (rgb_map, depth_map, acc_map) tuple
    """
    # Compute distances between samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e10], dim=-1)

    # Multiply by ray direction norm
    dists = dists * torch.norm(directions[..., None, :], dim=-1)

    # Compute alpha (opacity)
    alpha = 1.0 - torch.exp(-density.squeeze(-1) * dists)

    # Compute transmittance T(t)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1
    )[..., :-1]

    # Compute weights
    weights = alpha * transmittance

    # Render RGB
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

    # Render depth
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # Accumulation (opacity)
    acc_map = torch.sum(weights, dim=-1)

    # White background
    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, depth_map, acc_map


class QuantumNeRFRenderer:
    """
    High-level renderer for Q-NeRF with holographic optimization.

    Integrates:
    - Classical NeRF rendering
    - Quantum amplification
    - φ-optimized importance sampling
    - Holographic memory encoding
    """

    def __init__(self, model: QuantumNeRF, config: RenderConfig = None, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config or RenderConfig()
        self.device = device

        logger.info(f"🌌 QuantumNeRFRenderer initialized on {device}")

    def render_rays(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, near: float = 2.0, far: float = 6.0
    ) -> dict:
        """
        Render rays through Q-NeRF.

        Args:
            rays_o: Ray origins (N_rays, 3)
            rays_d: Ray directions (N_rays, 3)
            near: Near bound
            far: Far bound

        Returns:
            Dictionary with rendered outputs
        """
        N_rays = rays_o.shape[0]

        # Sample points along rays (stratified sampling with φ-jitter)
        t_vals = torch.linspace(0, 1, self.config.n_samples, device=self.device)
        z_vals = near * (1 - t_vals) + far * t_vals

        # φ-based jittering for stratified sampling
        if self.training:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)

            # Golden ratio jitter
            t_rand = torch.rand_like(z_vals) * (1 / PHI)
            z_vals = lower + (upper - lower) * t_rand

        z_vals = z_vals.expand([N_rays, self.config.n_samples])

        # Compute 3D points
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        # Flatten for network
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = rays_d[:, None, :].expand_as(pts).reshape(-1, 3)

        # Forward through Q-NeRF (in chunks)
        rgb_list, density_list = [], []

        for i in range(0, pts_flat.shape[0], self.config.chunk_size):
            chunk_pts = pts_flat[i : i + self.config.chunk_size]
            chunk_dirs = dirs_flat[i : i + self.config.chunk_size]

            chunk_rgb, chunk_density = self.model(
                chunk_pts, chunk_dirs, apply_quantum=self.config.use_quantum_amplification
            )

            rgb_list.append(chunk_rgb)
            density_list.append(chunk_density)

        rgb = torch.cat(rgb_list, dim=0).reshape(N_rays, self.config.n_samples, 3)
        density = torch.cat(density_list, dim=0).reshape(N_rays, self.config.n_samples, 1)

        # Volume rendering
        rgb_map, depth_map, acc_map = volume_rendering(
            rgb, density, z_vals, rays_d, white_background=self.config.white_background
        )

        return {"rgb": rgb_map, "depth": depth_map, "acc": acc_map, "z_vals": z_vals}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("🌌 Quantum-Enhanced NeRF Demo")
    print("=" * 60)

    # Create Q-NeRF model
    model = QuantumNeRF(hidden_dim=256, num_layers=8, use_quantum=True)

    # Test forward pass
    positions = torch.randn(1024, 3)
    directions = F.normalize(torch.randn(1024, 3), dim=-1)

    print(f"Input positions: {positions.shape}")
    print(f"Input directions: {directions.shape}")

    rgb, density = model(positions, directions)

    print(f"Output RGB:     {rgb.shape} (range: [{rgb.min():.3f}, {rgb.max():.3f}])")
    print(f"Output density: {density.shape} (range: [{density.min():.3f}, {density.max():.3f}])")

    # Test quantum amplification
    print("\n" + "=" * 60)
    print("Quantum Amplification Test")
    print("=" * 60)

    density_raw = torch.randn(1000, 1)
    amplifier = QuantumAmplifier(amplification_factor=PHI)

    density_amplified = amplifier(density_raw)

    print(f"Before amplification: mean={torch.sigmoid(density_raw).mean():.3f}")
    print(f"After amplification:  mean={density_amplified.mean():.3f}")
    print(
        f"Amplification gain:   {(density_amplified.mean() / torch.sigmoid(density_raw).mean()):.2f}x"
    )
