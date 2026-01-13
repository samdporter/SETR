#!/usr/bin/env python3
"""Visualise the Hessian-related quantities for the Relative Difference Prior (RDP).

This script builds a simple 3D synthetic volume, evaluates the RDP value,
its gradient, Hessian–diagonal, and a Hessian–vector product, and produces a
figure showing representative slices.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from recon_core.priors.rdp import RelativeDifferencePrior


class _BoxGeometry:
    """Minimal geometry wrapper exposing voxel_sizes() for the RDP prior."""

    def __init__(self, voxel_sizes):
        self._voxel_sizes = voxel_sizes

    def voxel_sizes(self):
        return self._voxel_sizes


def create_volume(shape=(64, 64, 32)):
    """Construct a smooth 3D test volume with mixed geometric primitives."""
    nx, ny, nz = shape
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-1.0, 1.0, ny)
    zs = np.linspace(-1.0, 1.0, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    # Base: smooth ramp
    volume = 40.0 * (1.0 + 0.3 * X + 0.2 * Y)

    # Add spherical hotspot
    r = np.sqrt(X**2 + Y**2 + Z**2)
    volume += 120.0 * np.exp(-((r - 0.35) ** 2) / 0.02)

    # Add rectangular prism
    prism = (np.abs(X) < 0.45) & (np.abs(Y) < 0.25) & (np.abs(Z) < 0.55)
    volume[prism] += 60.0

    # Hollow cylinder
    cyl = np.sqrt(X**2 + Y**2)
    ring = (cyl > 0.3) & (cyl < 0.45) & (np.abs(Z) < 0.6)
    volume[ring] += 80.0

    # Normalise slightly
    volume -= volume.min()
    volume /= volume.max() + 1e-8
    volume *= 200.0
    return volume.astype(np.float32)


def central_slices(arr):
    """Return axial, coronal, and sagittal mid-slices of a 3D array."""
    nz, ny, nx = arr.shape
    return (
        arr[nz // 2, :, :],
        arr[:, ny // 2, :],
        arr[:, :, nx // 2],
    )


def prepare_prior(voxel_sizes, gamma, epsilon, stencil, both_directions):
    geom = _BoxGeometry(voxel_sizes)
    return RelativeDifferencePrior(
        domain_geometry=geom,
        gamma=gamma,
        epsilon=epsilon,
        stencil=stencil,
        both_directions=both_directions,
    )


def visualise(volume, grad, diag, hv, output_dir):
    """Create a 3×3 grid summarising original volume and Hessian quantities."""
    output_dir.mkdir(parents=True, exist_ok=True)

    vol_slices = central_slices(volume)
    grad_slices = tuple(np.abs(g) for g in central_slices(grad))
    diag_slices = central_slices(diag)
    hv_slices = central_slices(hv)

    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    titles = ["Axial", "Coronal", "Sagittal"]

    def _show(row, slices, cmap, row_title, log=False):
        for col, sl in enumerate(slices):
            ax = axes[row, col]
            data = sl
            if log:
                data = np.sign(data) * np.log1p(np.abs(data))
            im = ax.imshow(data, cmap=cmap, origin="lower")
            ax.set_title(f"{row_title} - {titles[col]}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)

    _show(0, vol_slices, "magma", "Volume")
    _show(1, grad_slices, "viridis", "|Gradient|")
    _show(2, diag_slices, "plasma", "Hessian diag")
    _show(3, hv_slices, "coolwarm", "H·v (log)", log=True)

    fig.suptitle("Relative Difference Prior Diagnostics", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out_file = output_dir / "rdp_hessian_overview.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualisation to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualise the RDP Hessian components")
    parser.add_argument(
        "--shape", type=int, nargs=3, default=[64, 64, 32], help="Volume shape (nx ny nz)"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        default=[2.0, 2.0, 3.0],
        help="Voxel spacing (dx dy dz)",
    )
    parser.add_argument("--gamma", type=float, default=0.1, help="RDP gamma parameter")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="RDP epsilon stabiliser")
    parser.add_argument(
        "--stencil", type=str, default="6", choices=["6", "18", "26"], help="Neighbour stencil"
    )
    parser.add_argument(
        "--both-directions", action="store_true", help="Use both directions in gradient stencils"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/rdp_hessian"), help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: auto)",
    )

    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    device = torch.device(args.device)
    output_dir = Path(args.output)

    print("\nGenerating synthetic volume...")
    volume_np = create_volume(tuple(args.shape))
    volume_t = torch.as_tensor(volume_np, device=device)

    print("Setting up RDP prior...")
    prior = prepare_prior(
        tuple(args.voxel_size), args.gamma, args.epsilon, args.stencil, args.both_directions
    )

    print("Evaluating objective, gradient, and Hessian components...")
    val = prior._value_tensor(volume_t)
    grad = prior._grad_tensor(volume_t).detach().cpu().numpy()
    diag = prior._hess_diag_tensor(volume_t).detach().cpu().numpy()

    probe = torch.randn_like(volume_t)
    hv = prior._hess_vec_tensor(volume_t, probe).detach().cpu().numpy()

    print(f"Objective value: {val.item():.6f}")
    print(f"Gradient L2 norm: {np.linalg.norm(grad):.6f}")
    print(
        "Hessian diag stats: min={:.6f}, max={:.6f}, mean={:.6f}".format(
            float(diag.min()), float(diag.max()), float(diag.mean())
        )
    )

    visualise(
        volume_np,
        grad,
        diag,
        hv,
        output_dir,
    )

    print("Done.")


if __name__ == "__main__":
    main()
