#!/usr/bin/env python3
"""
Test script for VTV preconditioners on synthetic 3D geometric data.

Tests all 4 canonical preconditioner methods:
1. svd_principal_alpha - SVD principal + isotropic α (baseline)
2. mm_jensen - MM surrogate with SVD-free computation
3. frobenius_surrogate_pd - Frobenius norm + surrogate (positive-definite)
4. vector_tv_per_modality - Per-modality vector TV (exact radial, may not be PD)

Uses 3D synthetic geometric data with 2 modalities.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from sirf.STIR import ImageData

from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.priors.vtv import WeightedVectorialTotalVariation


def _quantiles(x, qs=(0.0, 1.0, 50.0, 90.0, 99.0, 100.0)):
    x = np.asarray(x).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"q{int(q)}": np.nan for q in qs}
    return {f"q{int(q)}": float(np.percentile(x, q)) for q in qs}


def debug_slow_components(vtv: WeightedVectorialTotalVariation, data, output_dir: Path):
    """Print diagnostic stats for the `svd_principal_alpha` diagonal components.

    Reports per-modality statistics for:
    - alpha_total = Σ_k φ'(σ_k)/σ_k
    - sum_sens2 = Σ_dir S^2
    - alpha_image = alpha_total * (b^2 * Σ S^2)
    - principal_image = Σ_k φ''(σ_k) * (J^T[w·(u_k v_k^T)])^2
    - slow_total = alpha_image + principal_image (recomputed)
    """
    try:
        import torch
    except Exception:
        print("[debug] torch not available; skipping slow diagnostics")
        return

    print("\n[debug] Computing slow preconditioner diagnostics...")

    x_arr = vtv.bdc2a.direct(data)  # torch tensor (nx,ny,nz,M)
    J = vtv.jacobian.direct(x_arr)
    w = vtv.weights
    A = w.unsqueeze(-1) * J

    # Spectral pieces
    try:
        coeffs, rank_one = vtv.vtv.hessian_components(A)
        sigma_w_half = vtv.vtv.hessian_surrogate(A)
    except Exception as e:
        print(f"[debug] hessian_components/hessian_surrogate not available: {e}")
        return

    alpha_total = 2.0 * torch.sum(sigma_w_half, dim=-1)  # (nx,ny,nz)
    S = vtv.jacobian.sensitivity(x_arr)
    S = torch.as_tensor(S, device=A.device, dtype=A.dtype)
    sum_sens2 = torch.sum(S * S, dim=-1)  # (nx,ny,nz,M)
    alpha_image = alpha_total.unsqueeze(-1) * (w * w) * sum_sens2  # (nx,ny,nz,M)

    # Principal term
    principal = torch.zeros_like(x_arr)
    r = rank_one.shape[-3]
    w_b = w.unsqueeze(-1)
    for k in range(r):
        Ck = rank_one[..., k, :, :]
        z = vtv.jacobian.adjoint(w_b * Ck)
        principal += coeffs[..., k].unsqueeze(-1) * (z * z)

    slow_total = alpha_image + principal

    # Print stats per modality
    for m in range(slow_total.shape[-1]):
        a_tot = alpha_total.detach().cpu().numpy()
        sens = sum_sens2[..., m].detach().cpu().numpy()
        a_img = alpha_image[..., m].detach().cpu().numpy()
        prn = principal[..., m].detach().cpu().numpy()
        tot = slow_total[..., m].detach().cpu().numpy()

        print(f"[debug] Slow diagnostics - modality {m+1}")
        print("  alpha_total (Σ φ'(σ)/σ):", _quantiles(a_tot))
        print("  Σ S^2:", _quantiles(sens))
        print("  alpha_image = alpha_total * b^2 * Σ S^2:", _quantiles(a_img))
        print("  principal_image (Σ φ'' z_k^2):", _quantiles(prn))
        print("  slow_total (alpha + principal):", _quantiles(tot))


def create_synthetic_3d_data(shape=(64, 64, 32), voxel_size=(2.0, 2.0, 3.0)):
    """
    Create synthetic 3D multi-modality test data with geometric features.

    Args:
        shape: (nx, ny, nz) spatial dimensions
        voxel_size: (dx, dy, dz) voxel spacing in mm

    Returns:
        dict with 'geometry', 'data', 'weights' keys
    """
    nx, ny, nz = shape

    # Create coordinate grids
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Modality 1: Nested spheres
    r1 = np.sqrt(X**2 + Y**2 + Z**2)
    modality1 = np.zeros_like(r1)
    modality1[r1 < 0.3] = 100.0  # Inner sphere
    modality1[(r1 >= 0.4) & (r1 < 0.6)] = 150.0  # Middle shell
    modality1[(r1 >= 0.7) & (r1 < 0.9)] = 80.0  # Outer shell

    # Modality 2: Box + cylinder
    modality2 = np.zeros_like(r1)

    # Box in center
    box_mask = (np.abs(X) < 0.4) & (np.abs(Y) < 0.4) & (np.abs(Z) < 0.3)
    modality2[box_mask] = 120.0

    # Cylinder along z-axis
    r_xy = np.sqrt(X**2 + Y**2)
    cyl_mask = (r_xy < 0.25) & (np.abs(Z) < 0.7)
    modality2[cyl_mask] = 90.0

    # Add smooth gradients
    modality1 += 20.0 * (1.0 + X)
    modality2 += 15.0 * (1.0 + Y)

    # Create SIRF ImageData templates
    template = ImageData()
    template.initialise(dim=(nz, ny, nx), vsize=voxel_size[::-1])  # SIRF uses (z,y,x)

    img1 = template.clone()
    img1.fill(modality1.T)  # Transpose for SIRF convention

    img2 = template.clone()
    img2.fill(modality2.T)

    # Create geometry (BlockDataContainer for multi-modality)
    geometry = EnhancedBlockDataContainer(img1, img2)

    # Create weights (spatially varying to test weight handling)
    weight1 = template.clone()
    weight2 = template.clone()

    # Radially varying weights
    weight_field = 0.5 + 0.5 * (1.0 - r1)  # Higher at center
    weight1.fill(weight_field.T)
    weight2.fill(weight_field.T)

    weights = EnhancedBlockDataContainer(weight1, weight2)

    # Create test data
    data = EnhancedBlockDataContainer(img1, img2)

    return {
        "geometry": geometry,
        "data": data,
        "weights": weights,
        "template": template,
        "shape": shape,
        "voxel_size": voxel_size,
    }


def visualize_inputs(test_data, output_dir: Path, cmap="magma"):
    """Save a quick-look figure summarising synthetic input modalities."""

    containers = test_data["data"].containers
    n_modalities = len(containers)
    views = ("Axial", "Coronal", "Sagittal")

    fig, axes = plt.subplots(n_modalities, len(views), figsize=(4 * len(views), 4 * n_modalities))
    axes = np.atleast_2d(axes)

    for mod_idx, img in enumerate(containers):
        arr = img.as_array()  # (z, y, x)
        nz, ny, nx = arr.shape
        slices = (
            arr[nz // 2, :, :],
            arr[:, ny // 2, :],
            arr[:, :, nx // 2],
        )

        for view_idx, sl in enumerate(slices):
            ax = axes[mod_idx, view_idx]
            im = ax.imshow(sl, cmap=cmap, origin="lower")
            ax.set_title(f"Modality {mod_idx + 1} - {views[view_idx]}", fontsize=10)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Synthetic Input Modalities", fontsize=14, y=0.98)
    plt.tight_layout()

    output_path = output_dir / "synthetic_inputs.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Input visualization saved to: {output_path}")


def test_preconditioner_methods(
    test_data,
    delta=1e-2,
    smoothing="charbonnier",
    debug=False,
    debug_dir=None,
):
    """
    Test all 4 preconditioner methods and compare.

    Args:
        test_data: Dict from create_synthetic_3d_data()
        delta: VTV smoothing parameter
        smoothing: Smoothing function ('charbonnier', 'fair', 'perona_malik')
        debug: If True, print diagnostics for svd_principal_alpha
        debug_dir: Directory for any debug output (currently prints only)

    Returns:
        DataFrame with results
    """
    # Use canonical names for clarity in output/plots
    methods = [
        "svd_principal_alpha",
        "mm_jensen",
        "frobenius_surrogate_pd",
        "vector_tv_per_modality",
    ]
    results = []

    print("=" * 70)
    print("Testing VTV Preconditioners on 3D Synthetic Data")
    print(f"  Shape: {test_data['shape']}")
    print(f"  Voxel size: {test_data['voxel_size']}")
    print(f"  Smoothing: {smoothing}, delta: {delta}")
    print("=" * 70)

    # Store preconditioners and outputs for comparison
    vtv_instances = {}
    hessian_outputs = {}
    inv_hessian_outputs = {}

    for method in methods:
        print(f"\n--- Testing method: {method} ---")

        # Create FRESH copies of geometry/weights for each VTV instance
        # This prevents any potential state interference between instances
        geometry = EnhancedBlockDataContainer(
            test_data["geometry"].containers[0].clone(), test_data["geometry"].containers[1].clone()
        )
        weights = EnhancedBlockDataContainer(
            test_data["weights"].containers[0].clone(), test_data["weights"].containers[1].clone()
        )

        # Create VTV with this method
        vtv = WeightedVectorialTotalVariation(
            geometry=geometry,
            weights=weights,
            delta=delta,
            smoothing=smoothing,
            norm="nuclear",
            hessian=method,
            stencil="18",
            bnd_cond="Periodic",
            both_directions=True
        )

        vtv_instances[method] = vtv

        if debug and method == "svd_principal_alpha":
            debug_slow_components(
                vtv,
                test_data["data"],
                output_dir=debug_dir if debug_dir is not None else Path("."),
            )

        # Test 1: Objective value (should be identical for all methods)
        print("  Computing objective...")
        t0 = time.time()
        obj_value = vtv(test_data["data"])
        obj_time = time.time() - t0
        print(f"    Objective: {obj_value:.6f} (time: {obj_time:.4f}s)")

        # Test 2: Gradient (should be identical for all methods)
        print("  Computing gradient...")
        t0 = time.time()
        gradient = vtv.gradient(test_data["data"])
        grad_time = time.time() - t0
        grad_norm = np.linalg.norm([np.linalg.norm(g.as_array()) for g in gradient.containers])
        print(f"    Gradient norm: {grad_norm:.6f} (time: {grad_time:.4f}s)")

        # Test 3: Hessian diagonal
        print("  Computing hessian_diag...")
        t0 = time.time()
        hess_diag = vtv.hessian_diag(test_data["data"])
        hess_time = time.time() - t0
        hessian_outputs[method] = hess_diag

        # Compute statistics
        hess_arrays = [h.as_array() for h in hess_diag.containers]
        hess_mean = np.mean([np.mean(h) for h in hess_arrays])
        hess_std = np.mean([np.std(h) for h in hess_arrays])
        hess_min = np.min([np.min(h) for h in hess_arrays])
        hess_max = np.max([np.max(h) for h in hess_arrays])

        print(f"    Hessian diag: mean={hess_mean:.6f}, std={hess_std:.6f}")
        print(f"                  min={hess_min:.6f}, max={hess_max:.6f}")
        print(f"                  time: {hess_time:.4f}s")

        # Test 4: Inverse Hessian diagonal
        print("  Computing inv_hessian_diag...")
        t0 = time.time()
        inv_hess_diag = vtv.inv_hessian_diag(test_data["data"])
        inv_hess_time = time.time() - t0
        inv_hessian_outputs[method] = inv_hess_diag

        # Compute statistics
        inv_hess_arrays = [h.as_array() for h in inv_hess_diag.containers]
        inv_hess_mean = np.mean([np.mean(h) for h in inv_hess_arrays])
        inv_hess_std = np.mean([np.std(h) for h in inv_hess_arrays])
        inv_hess_min = np.min([np.min(h) for h in inv_hess_arrays])
        inv_hess_max = np.max([np.max(h) for h in inv_hess_arrays])

        print(f"    Inv Hessian: mean={inv_hess_mean:.6f}, std={inv_hess_std:.6f}")
        print(f"                 min={inv_hess_min:.6f}, max={inv_hess_max:.6f}")
        print(f"                 time: {inv_hess_time:.4f}s")

        # Check positive definiteness
        is_positive = hess_min > 0
        print(f"    Positive definite: {is_positive}")

        results.append(
            {
                "method": method,
                "objective": obj_value,
                "gradient_norm": grad_norm,
                "hess_mean": hess_mean,
                "hess_std": hess_std,
                "hess_min": hess_min,
                "hess_max": hess_max,
                "inv_hess_mean": inv_hess_mean,
                "inv_hess_std": inv_hess_std,
                "inv_hess_min": inv_hess_min,
                "inv_hess_max": inv_hess_max,
                "positive_definite": is_positive,
                "obj_time": obj_time,
                "grad_time": grad_time,
                "hess_time": hess_time,
                "inv_hess_time": inv_hess_time,
            }
        )

    # Compare methods pairwise
    print(f"\n{'=' * 70}")
    print("Pairwise Comparisons (relative to 'svd_principal_alpha' baseline)")
    print("=" * 70)

    baseline = "svd_principal_alpha"
    for method in methods:
        if method == baseline:
            continue

        print(f"\n{method} vs {baseline}:")

        # Compare Hessian outputs
        for i, mod_name in enumerate(["PET", "SPECT"]):
            h_base = hessian_outputs[baseline].containers[i].as_array()
            h_test = hessian_outputs[method].containers[i].as_array()

            abs_diff = np.abs(h_test - h_base)
            rel_diff = np.abs((h_test - h_base) / (h_base + 1e-10))

            print(f"  {mod_name} Hessian:")
            print(f"    Max abs diff: {np.max(abs_diff):.6e}")
            print(f"    Mean rel diff: {np.mean(rel_diff):.6e}")
            print(f"    Max rel diff: {np.max(rel_diff):.6e}")

    return pd.DataFrame(results), hessian_outputs, inv_hessian_outputs


def visualize_preconditioners(test_data, hessian_outputs, output_dir):
    """Create visualization comparing preconditioner outputs."""

    methods = list(hessian_outputs.keys())
    n_methods = len(methods)

    # Create figure with subplots
    fig, axes = plt.subplots(2, n_methods, figsize=(4 * n_methods, 8))

    for i, method in enumerate(methods):
        # Modality 1 (PET)
        data1 = hessian_outputs[method].containers[0].as_array()
        mid_slice = data1.shape[0] // 2

        im1 = axes[0, i].imshow(data1[mid_slice, :, :], cmap="viridis")
        axes[0, i].set_title(f"{method}\n(Modality 1)", fontsize=10)
        axes[0, i].axis("off")
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046)

        # Modality 2 (SPECT)
        data2 = hessian_outputs[method].containers[1].as_array()

        im2 = axes[1, i].imshow(data2[mid_slice, :, :], cmap="viridis")
        axes[1, i].set_title(f"{method}\n(Modality 2)", fontsize=10)
        axes[1, i].axis("off")
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046)

    plt.suptitle("Hessian Diagonal Preconditioners (Mid-Axial Slice)", fontsize=14, y=0.98)
    plt.tight_layout()

    output_file = output_dir / "preconditioner_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_file}")
    plt.close()


def visualize_preconditioners_logscale(test_data, hessian_outputs, output_dir):
    """Create a log-scale visualization to expose small values.

    Uses a shared LogNorm per modality (row) based on robust percentiles
    across all methods, and clips zeros to vmin for display.
    """

    methods = list(hessian_outputs.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(2, n_methods, figsize=(4 * n_methods, 8))

    for mod_idx, mod_name in enumerate(["Modality 1", "Modality 2"]):
        # Collect mid-slice arrays for this modality across methods
        slices = []
        for method in methods:
            data = hessian_outputs[method].containers[mod_idx].as_array()
            mid_slice = data.shape[0] // 2
            slices.append(data[mid_slice, :, :])

        # Build robust vmin/vmax from positive values across all methods
        all_vals = np.concatenate([s.ravel() for s in slices])
        pos = all_vals[all_vals > 0]
        if pos.size == 0:
            vmin_log, vmax_log = 1.0, 1.0
        else:
            vmin_log = max(1e-12, np.percentile(pos, 1.0))
            vmax_log = np.percentile(pos, 99.5)
            if not np.isfinite(vmax_log) or vmax_log <= vmin_log:
                vmax_log = vmin_log * 10.0

        norm = LogNorm(vmin=vmin_log, vmax=vmax_log)

        for i, method in enumerate(methods):
            sl = slices[i]
            # Avoid log(0) by clipping to vmin
            sl_disp = np.maximum(sl, vmin_log)
            im = axes[mod_idx, i].imshow(sl_disp, cmap="viridis", norm=norm)
            axes[mod_idx, i].set_title(f"{method}\n({mod_name})", fontsize=10)
            axes[mod_idx, i].axis("off")
            plt.colorbar(im, ax=axes[mod_idx, i], fraction=0.046)

    plt.suptitle("Hessian Diagonal Preconditioners (Log Scale)", fontsize=14, y=0.98)
    plt.tight_layout()

    output_file = output_dir / "preconditioner_comparison_log.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Visualization (log scale) saved to: {output_file}")
    plt.close()


def create_timing_plot(results_df, output_dir):
    """Create bar plot comparing computation times."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    methods = results_df["method"].values

    # Plot 1: Objective time
    axes[0, 0].bar(methods, results_df["obj_time"].values)
    axes[0, 0].set_title("Objective Computation Time")
    axes[0, 0].set_ylabel("Time (s)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Plot 2: Gradient time
    axes[0, 1].bar(methods, results_df["grad_time"].values)
    axes[0, 1].set_title("Gradient Computation Time")
    axes[0, 1].set_ylabel("Time (s)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Plot 3: Hessian diag time
    axes[1, 0].bar(methods, results_df["hess_time"].values)
    axes[1, 0].set_title("Hessian Diagonal Time")
    axes[1, 0].set_ylabel("Time (s)")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Plot 4: Inverse Hessian time
    axes[1, 1].bar(methods, results_df["inv_hess_time"].values)
    axes[1, 1].set_title("Inverse Hessian Diagonal Time")
    axes[1, 1].set_ylabel("Time (s)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.suptitle("Computation Time Comparison", fontsize=14)
    plt.tight_layout()

    output_file = output_dir / "timing_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Timing plot saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test VTV preconditioners on synthetic 3D data")
    parser.add_argument(
        "--shape", type=int, nargs=3, default=[64, 64, 32], help="3D shape (nx ny nz)"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        default=[2.0, 2.0, 3.0],
        help="Voxel size in mm (dx dy dz)",
    )
    parser.add_argument("--delta", type=float, default=1e-2, help="VTV smoothing parameter")
    parser.add_argument(
        "--smoothing",
        type=str,
        default="charbonnier",
        choices=["charbonnier", "fair", "perona_malik"],
        help="Smoothing function",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/preconditioner_test_synthetic",
        help="Output directory",
    )
    parser.add_argument(
        "--debug-slow",
        action="store_true",
        help="Print diagnostics for the svd_principal_alpha components",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("VTV Preconditioner Test on Synthetic 3D Data")
    print("=" * 70)

    # Create synthetic data
    print("\nCreating synthetic 3D geometric data...")
    test_data = create_synthetic_3d_data(shape=tuple(args.shape), voxel_size=tuple(args.voxel_size))
    print(f"  Created {len(test_data['data'].containers)} modalities")
    print(f"  Shape: {test_data['shape']}")
    print(f"  Voxel size: {test_data['voxel_size']} mm")

    # Test preconditioners
    results_df, hessian_outputs, inv_hessian_outputs = test_preconditioner_methods(
        test_data,
        delta=args.delta,
        smoothing=args.smoothing,
        debug=args.debug_slow,
        debug_dir=output_dir,
    )

    # Visualise the input modalities for reference
    visualize_inputs(test_data, output_dir)

    # Save results
    results_file = output_dir / "results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n\nResults saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nObjective values (should be identical):")
    print(results_df[["method", "objective"]].to_string(index=False))

    print("\nGradient norms (should be identical):")
    print(results_df[["method", "gradient_norm"]].to_string(index=False))

    print("\nHessian diagonal statistics:")
    print(
        results_df[["method", "hess_mean", "hess_min", "hess_max", "positive_definite"]].to_string(
            index=False
        )
    )

    print("\nComputation times:")
    print(results_df[["method", "hess_time", "inv_hess_time"]].to_string(index=False))

    # Speedup analysis
    baseline_hess_time = results_df[results_df["method"] == "svd_principal_alpha"]["hess_time"].values[0]
    print("\nSpeedup vs 'svd_principal_alpha' baseline (Hessian diagonal):")
    for _, row in results_df.iterrows():
        speedup = baseline_hess_time / row["hess_time"]
        print(f"  {row['method']:20s}: {speedup:.2f}x")

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_preconditioners(test_data, hessian_outputs, output_dir)
    visualize_preconditioners_logscale(test_data, hessian_outputs, output_dir)
    create_timing_plot(results_df, output_dir)

    print("\n" + "=" * 70)
    print("Testing complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
