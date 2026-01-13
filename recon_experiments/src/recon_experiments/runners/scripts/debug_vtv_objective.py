#!/usr/bin/env python3
"""Debug script to check if VTV objectives match across different hessian types."""

import numpy as np
from sirf.STIR import ImageData

from recon_core.cil_extensions.framework.framework import EnhancedBlockDataContainer
from recon_core.priors.vtv import WeightedVectorialTotalVariation

# Create simple test data
shape = (16, 16, 8)
nx, ny, nz = shape

# Simple gradient pattern
data1 = np.arange(nx * ny * nz, dtype=np.float32).reshape(nz, ny, nx) + 100.0
data2 = np.arange(nx * ny * nz, dtype=np.float32).reshape(nz, ny, nx) * 2.0 + 50.0

# Create SIRF images
template = ImageData()
template.initialise(dim=(nz, ny, nx), vsize=(3.0, 2.0, 2.0))

img1 = template.clone()
img1.fill(data1)

img2 = template.clone()
img2.fill(data2)

# Create containers
geometry = EnhancedBlockDataContainer(img1, img2)
weights = EnhancedBlockDataContainer(img1.get_uniform_copy(1.0), img2.get_uniform_copy(1.0))
test_data = EnhancedBlockDataContainer(img1, img2)

print("=" * 70)
print("VTV Objective Consistency Test")
print("=" * 70)
print(f"Data shape: {shape}")
print(f"Modality 1 range: [{data1.min():.2f}, {data1.max():.2f}]")
print(f"Modality 2 range: [{data2.min():.2f}, {data2.max():.2f}]")
print()

methods = [
    "svd_principal_alpha",
    "mm_jensen",
    "frobenius_surrogate_pd",
    "vector_tv_per_modality",
]
results = {}

for method in methods:
    print(f"Testing method: {method}")

    vtv = WeightedVectorialTotalVariation(
        geometry=geometry,
        weights=weights,
        delta=0.01,
        smoothing="charbonnier",
        hessian=method,
        stencil="6",
        bnd_cond="Periodic",
    )

    # Compute objective
    obj = vtv(test_data)
    print(f"  Objective: {obj}")

    # Compute gradient
    grad = vtv.gradient(test_data)
    grad_norm = np.sqrt(sum([np.sum(g.as_array() ** 2) for g in grad.containers]))
    print(f"  Gradient norm: {grad_norm}")

    results[method] = {"obj": obj, "grad_norm": grad_norm}
    print()

print("=" * 70)
print("COMPARISON")
print("=" * 70)

baseline = results["svd_principal_alpha"]
print(
    f"Baseline (svd_principal_alpha): obj={baseline['obj']:.6f}, grad_norm={baseline['grad_norm']:.6f}"
)
print()

for method in ["mm_jensen", "frobenius_surrogate_pd", "vector_tv_per_modality"]:
    obj_diff = abs(results[method]["obj"] - baseline["obj"])
    grad_diff = abs(results[method]["grad_norm"] - baseline["grad_norm"])

    print(f"{method}:")
    print(f"  Objective diff: {obj_diff:.6e} (match: {obj_diff < 1e-3})")
    print(f"  Gradient diff: {grad_diff:.6e} (match: {grad_diff < 1e-3})")
