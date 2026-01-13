#!/usr/bin/env python3
"""Debug script to check what's happening with weights."""

import numpy as np
import torch
from sirf.STIR import ImageData

from recon_core.cil_extensions.framework.framework import EnhancedBlockDataContainer
from recon_core.utils import BlockDataContainerToArray

# Create simple test data
shape = (8, 8, 4)
nz, ny, nx = shape

# Create SIRF images
template = ImageData()
template.initialise(dim=(nz, ny, nx), vsize=(3.0, 2.0, 2.0))

img1 = template.clone()
img1.fill(np.ones((nz, ny, nx)) * 100.0)

img2 = template.clone()
img2.fill(np.ones((nz, ny, nx)) * 200.0)

# Create containers
geometry = EnhancedBlockDataContainer(img1, img2)
weights = EnhancedBlockDataContainer(img1.get_uniform_copy(1.0), img2.get_uniform_copy(1.0))

print("Original geometry containers:")
for i, c in enumerate(geometry.containers):
    arr = c.as_array()
    print(f"  Container {i}: mean={arr.mean():.2f}, shape={arr.shape}")

print("\nOriginal weights containers:")
for i, c in enumerate(weights.containers):
    arr = c.as_array()
    print(f"  Container {i}: mean={arr.mean():.2f}, shape={arr.shape}")

# Create two converters (simulating two VTV instances)
bdc2a_1 = BlockDataContainerToArray(geometry)
bdc2a_2 = BlockDataContainerToArray(geometry)

print("\n" + "=" * 70)
print("Converting weights with bdc2a_1...")
weights_1 = bdc2a_1.direct(weights)
print(f"weights_1 shape: {weights_1.shape}")
print(f"weights_1 mean: {weights_1.mean().item():.6f}")
print(f"weights_1 unique values: {torch.unique(weights_1).cpu().numpy()}")

print("\n" + "=" * 70)
print("Converting weights with bdc2a_2...")
weights_2 = bdc2a_2.direct(weights)
print(f"weights_2 shape: {weights_2.shape}")
print(f"weights_2 mean: {weights_2.mean().item():.6f}")
print(f"weights_2 unique values: {torch.unique(weights_2).cpu().numpy()}")

print("\n" + "=" * 70)
print("Are weights identical?", torch.allclose(weights_1, weights_2))

print("\n" + "=" * 70)
print("Geometry after conversions:")
for i, c in enumerate(geometry.containers):
    arr = c.as_array()
    print(f"  Container {i}: mean={arr.mean():.2f}")
