import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ==========================================
# 1. SETUP & GEOMETRY
# ==========================================
np.random.seed(42)

# Parameters
LENGTH = 140
CENTER = 70
VOXEL_SIZE = 2.0  # mm

NOISE_LEVEL = 0.1
SMOOTHING_SIGMA = 1.5

# DEMO 1: Two Tumors - Demonstrating need for BOUNDED SPOKES
# Primary tumor (close, fuzzy) vs Secondary tumor (far, sharp edge)
TUMOR1_CENTER = CENTER
TUMOR1_WIDTH = 7.0    # Fuzzy edge
TUMOR1_INTENSITY = 1.0

TUMOR2_CENTER = CENTER + 50  # Far away
TUMOR2_WIDTH = 2.0    # Sharp edge (stronger gradient)
TUMOR2_INTENSITY = 0.9

BACKGROUND_INTENSITY = 0.1

# DEMO 2: Tumor near Liver - Demonstrating need for INTENSITY THRESHOLD
LIVER_CENTER = CENTER  # Liver center
LIVER_WIDTH = 90.0  # Much wider liver
LIVER_SHARPNESS = 0.1  # Sharp edge
LIVER_INTENSITY = 0.35  # Below 42% threshold

TUMOR_NEAR_LIVER_CENTER = CENTER + 25  # Tumor close to one side of liver
TUMOR_NEAR_LIVER_WIDTH = 8.0  # Fuzzy edge
TUMOR_NEAR_LIVER_INTENSITY = 0.8

# Intensity threshold: 42% of (max - background)
INTENSITY_THRESHOLD_FRACTION = 0.42

# set save path as parent dir of file
SAVE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 2. GENERATE PROFILES
# ==========================================
def generate_two_tumor_profile():
    """Demo 1: Two tumors to show need for bounded spokes"""
    x = np.arange(LENGTH)
    
    # Background
    signal = np.ones(LENGTH) * BACKGROUND_INTENSITY
    
    # Primary tumor (close, fuzzy edge - lower gradient)
    tumor1_signal = (TUMOR1_INTENSITY - BACKGROUND_INTENSITY) * np.exp(-((x - TUMOR1_CENTER)**2) / (2 * TUMOR1_WIDTH**2))
    
    # Secondary tumor (far, sharp edge - higher gradient)
    tumor2_signal = (TUMOR2_INTENSITY - BACKGROUND_INTENSITY) * np.exp(-((x - TUMOR2_CENTER)**2) / (2 * TUMOR2_WIDTH**2))
    
    # Combine
    signal = signal + tumor1_signal + tumor2_signal
    
    # Add Noise
    noise = np.random.normal(0, NOISE_LEVEL, size=LENGTH)
    noisy_signal = np.clip(signal + noise, 0, None)
    
    return x, signal, noisy_signal

def generate_tumor_near_liver_profile():
    """Demo 2: Tumor near liver boundary to show need for intensity threshold"""
    x = np.arange(LENGTH)
    
    # Background
    signal = np.ones(LENGTH) * BACKGROUND_INTENSITY
    
    # Liver with sharp edge (sigmoid)
    liver_dist = np.abs(x - LIVER_CENTER)
    liver_signal = (LIVER_INTENSITY - BACKGROUND_INTENSITY) / (1 + np.exp((liver_dist - LIVER_WIDTH/2) / LIVER_SHARPNESS))
    
    # Tumor near liver boundary (fuzzy edge)
    tumor_signal = (TUMOR_NEAR_LIVER_INTENSITY - BACKGROUND_INTENSITY) * np.exp(-((x - TUMOR_NEAR_LIVER_CENTER)**2) / (TUMOR_NEAR_LIVER_WIDTH**2))
    
    # Combine (tumor on top of background, liver separate)
    signal = np.maximum(signal + liver_signal, signal + tumor_signal)
    
    # Add Noise
    noise = np.random.normal(0, NOISE_LEVEL, size=LENGTH)
    noisy_signal = np.clip(signal + noise, 0, None)
    
    return x, signal, noisy_signal


# ==========================================
# 3. ALGORITHMS
# ==========================================
def get_gradient(signal):
    """Get gradient (keeping sign information)"""
    return np.gradient(signal)

def find_global_max_gradient(signal, gradient, seed_idx, max_radius=None, intensity_threshold=None):
    """Finds the steepest NEGATIVE slope (descending from tumor center), optionally bounded and with intensity threshold."""
    edges = []
    for direction in [-1, 1]:
        # Build ray indices
        if direction == -1:
            ray = np.arange(seed_idx, -1, -1)
        else:
            ray = np.arange(seed_idx, len(signal))
            
        # Apply Bounding Box (radial spoke limit)
        if max_radius:
            ray = ray[:max_radius]
        
        # Apply intensity threshold mask
        if intensity_threshold is not None:
            valid_mask = signal[ray] >= intensity_threshold
            ray = ray[valid_mask]
            
        # Find max NEGATIVE gradient on this ray (steepest descent from viewpoint of spoke)
        if len(ray) > 0:
            ray_grads = gradient[ray]
            
            # From the viewpoint of the spoke going outward from seed:
            # - Left spoke (direction=-1): moving left, want signal to decrease (gradient > 0 means decreasing as we go left)
            # - Right spoke (direction=+1): moving right, want signal to decrease (gradient < 0 means decreasing as we go right)
            
            if direction == -1:
                # Going left: gradient > 0 means signal decreases as we move left (descending spoke)
                # We want the most positive gradient (steepest descent)
                descending_grads = ray_grads
            else:
                # Going right: gradient < 0 means signal decreases as we move right (descending spoke)
                # We want the most negative gradient, so flip sign to find maximum
                descending_grads = -ray_grads
            
            # Only consider descending slopes (positive values after transformation)
            if np.any(descending_grads > 0):
                local_max_idx = np.argmax(descending_grads)
                edges.append(ray[local_max_idx])
            
    return sorted(edges) if len(edges) == 2 else None


# ==========================================
# 4. DEMO 1: NEED FOR BOUNDED SPOKES
# ==========================================
print("=" * 60)
print("DEMO 1: Demonstrating need for BOUNDED RADIAL SPOKES")
print("=" * 60)

x1, true_signal1, noisy_signal1 = generate_two_tumor_profile()

# Smooth the data (Standard PET preprocessing)
smooth_signal1 = gaussian_filter(noisy_signal1, sigma=SMOOTHING_SIGMA)
smooth_gradient1 = get_gradient(smooth_signal1)

# Run Algorithms
# 1. Unbounded: "Go find the sharpest edge anywhere" - WRONG (finds distant tumor)
unbounded_edges1 = find_global_max_gradient(smooth_signal1, smooth_gradient1, TUMOR1_CENTER, max_radius=None)

# 2. Bounded: "Find sharpest edge within 20 voxels" - CORRECT (finds actual tumor)
bounded_edges1 = find_global_max_gradient(smooth_signal1, smooth_gradient1, TUMOR1_CENTER, max_radius=20)

print(f"Unbounded edges: {unbounded_edges1}")
print(f"Bounded edges (radius=20): {bounded_edges1}")

# ==========================================
# 5. DEMO 2: NEED FOR INTENSITY THRESHOLD
# ==========================================
print("\n" + "=" * 60)
print("DEMO 2: Demonstrating need for INTENSITY THRESHOLD")
print("=" * 60)

x2, true_signal2, noisy_signal2 = generate_tumor_near_liver_profile()

# Smooth the data
smooth_signal2 = gaussian_filter(noisy_signal2, sigma=SMOOTHING_SIGMA)
smooth_gradient2 = get_gradient(smooth_signal2)

# Calculate 42% threshold between liver and tumor
signal_max = np.max(smooth_signal2)
liver_intensity = LIVER_INTENSITY
intensity_threshold = liver_intensity + INTENSITY_THRESHOLD_FRACTION * (signal_max - liver_intensity)

print(f"Signal max: {signal_max:.3f}")
print(f"Liver intensity: {liver_intensity:.3f}")
print(f"42% threshold: {intensity_threshold:.3f}")

# Run Algorithms
# 1. Without threshold: Spokes hit liver edge - WRONG
no_threshold_edges2 = find_global_max_gradient(smooth_signal2, smooth_gradient2, TUMOR_NEAR_LIVER_CENTER, max_radius=30)

# 2. With threshold: Liver excluded, finds tumor edge - CORRECT
with_threshold_edges2 = find_global_max_gradient(smooth_signal2, smooth_gradient2, TUMOR_NEAR_LIVER_CENTER, 
                                                   max_radius=30, intensity_threshold=intensity_threshold)

print(f"Without threshold: {no_threshold_edges2}")
print(f"With 42% threshold: {with_threshold_edges2}")


# ==========================================
# 6. PLOTTING DEMO 1: BOUNDED SPOKES
# ==========================================
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5))

# Results plot only
ax = ax1
ax.plot(x1, smooth_signal1, 'k-', alpha=0.4, linewidth=2, label="PET Signal")

# Plot Unbounded Fail
if unbounded_edges1 and len(unbounded_edges1) == 2:
    u_l, u_r = unbounded_edges1
    ax.scatter([u_l, u_r], [smooth_signal1[u_l], smooth_signal1[u_r]], 
               c='red', marker='x', s=300, linewidth=4, zorder=10, label="Unbounded (FAILS - hits distant tumor)")
    ax.axvspan(u_l, u_r, alpha=0.1, color='red')

# Plot Bounded Success
if bounded_edges1 and len(bounded_edges1) == 2:
    b_l, b_r = bounded_edges1
    ax.scatter([b_l, b_r], [smooth_signal1[b_l], smooth_signal1[b_r]], 
               c='green', marker='*', s=400, zorder=11, label="Bounded (CORRECT - finds primary tumor)")
    ax.axvspan(b_l, b_r, alpha=0.15, color='green')

ax.axvline(TUMOR1_CENTER, color='blue', linestyle='--', alpha=0.5, linewidth=1, label="Primary Tumor Seed")
ax.axvline(TUMOR2_CENTER, color='orange', linestyle='--', alpha=0.5, linewidth=1, label="Secondary Tumor")
#ax.set_title("Demo 1: Bounded Spokes Prevent Wrong Tumor Selection", fontsize=13, fontweight='bold')
ax.set_xlabel("Position (voxels)", fontsize=11)
ax.set_ylabel("Intensity", fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "demo1_bounded_spokes.png"), dpi=150)
print(f"\nSaved: {os.path.join(SAVE_PATH, 'demo1_bounded_spokes.png')}")

# ==========================================
# 7. PLOTTING DEMO 2: INTENSITY THRESHOLD
# ==========================================
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))

# Results plot only
ax = ax2
ax.plot(x2, smooth_signal2, 'k-', alpha=0.4, linewidth=2, label="PET Signal")
ax.axhline(intensity_threshold, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=f"42% Threshold")

# Plot Without Threshold - FAILS
if no_threshold_edges2 and len(no_threshold_edges2) == 2:
    n_l, n_r = no_threshold_edges2
    ax.scatter([n_l, n_r], [smooth_signal2[n_l], smooth_signal2[n_r]], 
               c='red', marker='x', s=300, linewidth=4, zorder=10, 
               label="Without Threshold (FAILS - hits liver)")
    ax.axvspan(n_l, n_r, alpha=0.1, color='red')

# Plot With Threshold - SUCCESS
if with_threshold_edges2 and len(with_threshold_edges2) == 2:
    w_l, w_r = with_threshold_edges2
    ax.scatter([w_l, w_r], [smooth_signal2[w_l], smooth_signal2[w_r]], 
               c='green', marker='*', s=400, zorder=11, 
               label="With Threshold (CORRECT - finds tumor)")
    ax.axvspan(w_l, w_r, alpha=0.15, color='green')

ax.axvline(TUMOR_NEAR_LIVER_CENTER, color='blue', linestyle='--', alpha=0.5, linewidth=1, label="Tumor Seed")
#ax.set_title("Demo 2: Intensity Threshold Excludes Low-Intensity Liver", fontsize=13, fontweight='bold')
ax.set_xlabel("Position (voxels)", fontsize=11)
ax.set_ylabel("Intensity", fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "demo2_intensity_threshold.png"), dpi=150)
print(f"Saved: {os.path.join(SAVE_PATH, 'demo2_intensity_threshold.png')}")


# ==========================================
# 8. 2D VISUALIZATION (Simplified Ray Casting - kept from original)
# ==========================================
# Create simple 2D image
Y, X = np.ogrid[:100, :100]
dist_tumor = np.sqrt((X - 50)**2 + (Y - 50)**2)
dist_organ = np.sqrt((X - 85)**2 + (Y - 50)**2) # Organ to the right

img_2d = 1.0 * np.exp(-dist_tumor**2 / (2 * 10**2))
img_2d += 0.8 * np.exp(-dist_organ**2 / (2 * 15**2))
img_2d += np.random.normal(0, 0.05, size=img_2d.shape) # Noise
img_smooth = gaussian_filter(img_2d, sigma=2.0)

# Compute 2D Gradient
gy, gx = np.gradient(img_smooth)
g_mag_2d = np.sqrt(gy**2 + gx**2)

plt.figure(figsize=(10, 5))

# Plot Image with Rays
plt.subplot(1, 2, 1)
plt.imshow(img_smooth, cmap='hot', origin='lower')
#plt.title("2D PET simulation")
plt.colorbar(label="Intensity", shrink = 0.7)

# Simulate a few rays
seed = (50, 50)
angles = np.linspace(0, 2*np.pi, 12, endpoint=False) # 12 rays

for angle in angles:
    # Ray direction
    dx, dy = np.cos(angle), np.sin(angle)
    
    # Cast ray (finding max grad)
    max_grad = 0
    best_r = 0
    # Search up to 25 voxels (Bounded)
    for r in range(1, 25):
        xi = int(seed[0] + r * dx)
        yi = int(seed[1] + r * dy)
        if 0 <= xi < 100 and 0 <= yi < 100:
            g = g_mag_2d[yi, xi]
            if g > max_grad:
                max_grad = g
                best_r = r
    
    # Plot Ray
    plt.plot([seed[0], seed[0] + best_r*dx], [seed[1], seed[1] + best_r*dy], 'c-', alpha=0.5)
    # Plot Hit
    plt.plot(seed[0] + best_r*dx, seed[1] + best_r*dy, 'c.', markersize=8)

plt.scatter(seed[0], seed[1], c='red', marker='+', s=100, label="Seed")
plt.legend(loc='lower right')

# Plot Gradient Map
plt.subplot(1, 2, 2)
plt.imshow(g_mag_2d, cmap='gray', origin='lower')
#plt.title("Gradient Magnitude (Edge Map)")
plt.colorbar(label="| Gradient |", shrink = 0.7)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "lesion_grower_2d_visualization.png"), dpi=150)
print(f"Saved: {os.path.join(SAVE_PATH, 'lesion_grower_2d_visualization.png')}")

print("\n" + "=" * 60)
print("DEMONSTRATIONS COMPLETE")
print("=" * 60)
plt.show()
