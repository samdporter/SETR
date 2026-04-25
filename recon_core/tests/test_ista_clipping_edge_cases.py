import numpy as np


def _simulate_global_clip_step(x: np.ndarray, grad: np.ndarray, step_size: float) -> np.ndarray:
    """Mimic the current ISTA clipping rule that uses global max over modalities."""
    m_global = float(np.max(x))
    clip = m_global / step_size
    grad_clipped = np.clip(grad, -clip, clip)
    x_new = x - step_size * grad_clipped
    return np.maximum(x_new, 0.0)


def _simulate_per_modality_clip_step(x: np.ndarray, grad: np.ndarray, step_size: float) -> np.ndarray:
    """Reference variant: clip each modality with its own max value."""
    x_new = np.empty_like(x)
    for modality in range(x.shape[-1]):
        m_mod = float(np.max(x[..., modality]))
        clip = m_mod / step_size
        grad_m = np.clip(grad[..., modality], -clip, clip)
        x_new[..., modality] = x[..., modality] - step_size * grad_m
    return np.maximum(x_new, 0.0)


def _salt_pepper(shape, salt_prob=0.05, pepper_prob=0.05, salt_value=100.0, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.zeros(shape, dtype=np.float64)
    salt = rng.random(shape) < salt_prob
    pepper = rng.random(shape) < pepper_prob
    arr[salt] = salt_value
    arr[pepper] = 0.0
    return arr


def test_global_clip_allows_large_updates_in_zero_modality_with_salt_pepper_other_modality():
    """
    Edge case requested by user:
    - modality 0: salt-and-pepper outliers (large max)
    - modality 1: near-zero image

    Global clipping uses M=max over *both* modalities, so modality-1 updates can
    become very large relative to modality-1 scale.
    """
    shape = (24, 20, 2)  # (voxels..., modalities)
    x = np.zeros(shape, dtype=np.float64)
    x[..., 0] = _salt_pepper(shape[:-1], salt_prob=0.04, pepper_prob=0.04, salt_value=100.0, seed=1)
    x[..., 1] = 1e-3  # effectively zero modality

    grad = np.zeros_like(x)
    grad[..., 1] = -1.0  # drives modality-1 upwards in this update convention
    step_size = 1.0

    x_new = _simulate_global_clip_step(x, grad, step_size)

    modality1_update = np.abs(x_new[..., 1] - x[..., 1])
    rel_change = modality1_update / np.maximum(x[..., 1], 1e-12)

    # In this setup, update is ~1.0 while baseline scale is 1e-3 -> ~1e3 relative jump.
    assert float(np.max(rel_change)) > 100.0, (
        f"Expected large cross-modal relative jump in modality 1, got max ratio={np.max(rel_change):.1f}"
    )


def test_per_modality_clipping_controls_same_salt_pepper_edge_case():
    """Per-modality clipping should strongly reduce modality-1 jump in the same setup."""
    shape = (24, 20, 2)
    x = np.zeros(shape, dtype=np.float64)
    x[..., 0] = _salt_pepper(shape[:-1], salt_prob=0.04, pepper_prob=0.04, salt_value=100.0, seed=2)
    x[..., 1] = 1e-3

    grad = np.zeros_like(x)
    grad[..., 1] = -1.0
    step_size = 1.0

    x_global = _simulate_global_clip_step(x, grad, step_size)
    x_mod = _simulate_per_modality_clip_step(x, grad, step_size)

    jump_global = float(np.max(np.abs(x_global[..., 1] - x[..., 1])))
    jump_mod = float(np.max(np.abs(x_mod[..., 1] - x[..., 1])))

    # Per-modality clipping caps modality-1 update at its own M (~1e-3).
    assert jump_mod <= float(np.max(x[..., 1])) + 1e-12
    assert jump_global > 100.0 * jump_mod, (
        f"Expected global clip to permit much larger modality-1 jump "
        f"(global={jump_global:.3e}, per-modality={jump_mod:.3e})"
    )
