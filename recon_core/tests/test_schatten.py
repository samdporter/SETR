#!/usr/bin/env python3
"""
Test the three Schatten-norm backends using Jacobians of real images
from skimage.data (grayscale or RGB).

Backends:
- schatten_norm_gpu_slow.GPUVectorialTotalVariation  (gold standard)
- schatten_norm_gpu_stable.GPUVectorialTotalVariation
- schatten_norm_gpu.GPUVectorialTotalVariation       (SVD-free; d_dirs = 2 or 3 only)
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import data, img_as_float

from recon_core.core.gradients import Jacobian  # your Jacobian class
from recon_core.priors.vtv.schatten_norm_gpu import GPUVectorialTotalVariation as VTV_SVDFREE
from recon_core.priors.vtv.schatten_norm_gpu_slow import GPUVectorialTotalVariation as VTV_SLOW
from recon_core.priors.vtv.schatten_norm_gpu_stable import GPUVectorialTotalVariation as VTV_STABLE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- utils ----------------


def prepare_images(kind="kidney"):
    """
    Load an image from skimage.data and return a tensor (L,M,N,n_params).
    If grayscale: shape (H,W,1); if RGB: (H,W,3).
    Optionally create >3 params by adding blurred/noisy variants.
    """
    if kind == "kidney":
        img = img_as_float(data.kidney())  # shape (16,512,512,3)
    elif kind == "cells3d":
        img = img_as_float(data.cells3d())  # shape (60,2,512,512)
        # need to transpose
        img = img.transpose(0, 2, 3, 1)  # shape (60,512,512,2)

    return (img - img.min()) / (img.max() - img.min())


def rel_err(a, b, eps=1e-12):
    return np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), eps)


def map_values(vtv, U):
    with torch.no_grad():
        U_t = torch.as_tensor(U, device=device, dtype=torch.float32)
        v = vtv.direct(U_t)
    return v.cpu().numpy()


def map_grad_norm(vtv, U):
    with torch.no_grad():
        U_t = torch.as_tensor(U, device=device, dtype=torch.float32)
        G = vtv.gradient(U_t)
    G = G.cpu().numpy()
    return np.sqrt((G**2).sum(axis=(-2, -1)))


def map_prox_norm(vtv, U, tau):
    with torch.no_grad():
        U_t = torch.as_tensor(U, device=device, dtype=torch.float32)
        P = vtv.proximal(U_t, tau)
    P = P.cpu().numpy()
    return np.sqrt((P**2).sum(axis=(-2, -1)))


def map_sigma_norm(vtv, U):
    with torch.no_grad():
        U_t = torch.as_tensor(U, device=device, dtype=torch.float32)
        S = vtv.hessian_surrogate(U_t)
    S = S.cpu().numpy()
    return np.sqrt((S**2).sum(axis=(-1)))


def plot_row(imgs, titles, suptitle, cmap="viridis"):
    K = len(imgs)
    fig, axs = plt.subplots(1, K, figsize=(5 * K, 4), constrained_layout=True)
    if K == 1:
        axs = [axs]
    imgs_abs = [np.abs(im) for im in imgs]
    vmax = min(im.max() for im in imgs_abs)
    for ax, im, ttl in zip(axs, imgs_abs, titles):
        h = ax.imshow(im, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(ttl)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = plt.colorbar(h, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.set_ylabel("value", rotation=90)
    fig.suptitle(suptitle)
    return fig


def _sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def avg_time(func, n_runs=10, n_warmup=2):
    """
    Time a zero-arg callable. GPU-safe (uses cuda synchronize).
    Returns average wall time over n_runs.
    """
    # warm-up
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = func()
    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = func()
    _sync()
    t1 = time.perf_counter()
    return (t1 - t0) / n_runs


# ---------------- main ----------------


def run_tests():
    # Backends
    delta = 1e-3
    norm = "nuclear"
    smoothing = "fair"
    tail = None
    tau = 0.2
    n_runs = 1
    # Show central slice

    # load real images
    for kind in ["cells3d", "kidney"]:
        print(f"\n=== Loading images for: {kind} ===")
        X = prepare_images(kind)
        print(X.shape)
        H, W, N, n_params = X.shape
        print(f"Images shape: {X.shape} → n_params={n_params}")

        # Jacobian operator
        Jop = Jacobian(
            voxel_sizes=(1.0, 1.0, 1.0),
            bnd_cond="Neumann",
            anatomical=None,
            stencil="6",
            both_directions=False,
            normalize=True,
            numpy_out=True,
        )

        # Apply Jacobian: (H,W,N,n_params,d_dirs)
        U = Jop.direct(X)
        d_dirs = U.shape[-1]
        print(f"Jacobian output shape: {U.shape} → d_dirs={d_dirs}")

        # Tensors on device for timing
        U_t = torch.as_tensor(U, device=device, dtype=torch.float32)

        # Backends (numpy_out=False to keep tensors on device)
        vtv_slow = VTV_SLOW(
            eps=delta, norm=norm, smoothing_function=smoothing, tail=tail, numpy_out=False
        )
        vtv_stable = VTV_STABLE(
            eps=delta, norm=norm, smoothing_function=smoothing, tail=tail, numpy_out=False
        )
        vtv_svdfr = VTV_SVDFREE(
            eps=delta, norm=norm, smoothing_function=smoothing, tail=tail, numpy_out=False
        )

        # ----- values for correctness -----
        val_slow = map_values(vtv_slow, U)
        val_stable = map_values(vtv_stable, U)
        if d_dirs in (2, 3):
            val_svdfr = map_values(vtv_svdfr, U)

        print("value rel.err stable vs slow:", rel_err(val_stable, val_slow))
        if d_dirs in (2, 3):
            print("value rel.err svd-free vs slow:", rel_err(val_svdfr, val_slow))

        # ----- gradients for correctness -----
        g_slow = map_grad_norm(vtv_slow, U)
        g_stable = map_grad_norm(vtv_stable, U)
        if d_dirs in (2, 3):
            g_svdfr = map_grad_norm(vtv_svdfr, U)
        print("grad  rel.err stable vs slow:", rel_err(g_stable, g_slow))
        if d_dirs in (2, 3):
            print("grad  rel.err svd-free vs slow:", rel_err(g_svdfr, g_slow))

        # ----- prox for correctness -----
        p_slow = map_prox_norm(vtv_slow, U, tau)
        p_stable = map_prox_norm(vtv_stable, U, tau)
        if d_dirs in (2, 3):
            p_svdfr = map_prox_norm(vtv_svdfr, U, tau)
        print("prox  rel.err stable vs slow:", rel_err(p_stable, p_slow))
        if d_dirs in (2, 3):
            print("prox  rel.err svd-free vs slow:", rel_err(p_svdfr, p_slow))

        # ----- hessian surrogate for correctness -----
        h_slow = map_sigma_norm(vtv_slow, U)
        h_stable = map_sigma_norm(vtv_stable, U)
        if d_dirs in (2, 3):
            h_svdfr = map_sigma_norm(vtv_svdfr, U)
        print("hessian rel.err stable vs slow:", rel_err(h_stable, h_slow))
        if d_dirs in (2, 3):
            print("hessian rel.err svd-free vs slow:", rel_err(h_svdfr, h_slow))

        # ----- timings -----
        print(f"\n--- Average runtimes over {n_runs} runs (seconds) ---")

        t_dir_slow = avg_time(lambda: vtv_slow.direct(U_t), n_runs=n_runs)
        t_grad_slow = avg_time(lambda: vtv_slow.gradient(U_t), n_runs=n_runs)
        t_prox_slow = avg_time(lambda: vtv_slow.proximal(U_t, tau), n_runs=n_runs)
        t_hess_slow = avg_time(lambda: vtv_slow.hessian_surrogate(U_t), n_runs=n_runs)
        print(
            f"SLOW     | direct: {t_dir_slow:.6f} | gradient: {t_grad_slow:.6f} | proximal: {t_prox_slow:.6f} | hessian: {t_hess_slow:.6f}"
        )

        # print stability report
        print(vtv_stable.stability_report(U_t))

        t_dir_stb = avg_time(lambda: vtv_stable.direct(U_t), n_runs=n_runs)
        t_grad_stb = avg_time(lambda: vtv_stable.gradient(U_t), n_runs=n_runs)
        t_prox_stb = avg_time(lambda: vtv_stable.proximal(U_t, tau), n_runs=n_runs)
        t_hess_stb = avg_time(lambda: vtv_stable.hessian_surrogate(U_t), n_runs=n_runs)
        print(
            f"STABLE   | direct: {t_dir_stb:.6f} | gradient: {t_grad_stb:.6f} | proximal: {t_prox_stb:.6f} | hessian: {t_hess_stb:.6f}"
        )

        if d_dirs in (2, 3):
            t_dir_svd = avg_time(lambda: vtv_svdfr.direct(U_t), n_runs=n_runs)
            t_grad_svd = avg_time(lambda: vtv_svdfr.gradient(U_t), n_runs=n_runs)
            t_prox_svd = avg_time(lambda: vtv_svdfr.proximal(U_t, tau), n_runs=n_runs)
            t_hess_svd = avg_time(lambda: vtv_svdfr.hessian_surrogate(U_t), n_runs=n_runs)
            print(
                f"SVD-FREE | direct: {t_dir_svd:.6f} | gradient: {t_grad_svd:.6f} | proximal: {t_prox_svd:.6f} | hessian: {t_hess_svd:.6f}"
            )
        else:
            print("SVD-FREE | skipped (requires d_dirs ∈ {2,3})")

        # choose central slice
        z = H // 2

        # ----- values -----
        imgs_val = [val_slow[z], val_stable[z]]
        titles_val = ["slow:value", "stable:value"]
        if d_dirs in (2, 3):
            imgs_val.append(val_svdfr[z])
            titles_val.append("svd-free:value")
        plot_row(imgs_val, titles_val, "Value maps")

        # ----- gradient norms -----
        imgs_grad = [g_slow[z], g_stable[z]]
        titles_grad = ["slow:‖grad‖", "stable:‖grad‖"]
        if d_dirs in (2, 3):
            imgs_grad.append(g_svdfr[z])
            titles_grad.append("svd-free:‖grad‖")
        plot_row(imgs_grad, titles_grad, "Gradient norms")

        # ----- prox norms -----
        imgs_prox = [p_slow[z], p_stable[z]]
        titles_prox = [f"slow:‖prox‖ (τ={tau})", f"stable:‖prox‖ (τ={tau})"]
        if d_dirs in (2, 3):
            imgs_prox.append(p_svdfr[z])
            titles_prox.append(f"svd-free:‖prox‖ (τ={tau})")
        plot_row(imgs_prox, titles_prox, "Proximal outputs")

        # ----- hessian norms -----
        imgs_hess = [h_slow[z], h_stable[z]]
        titles_hess = ["slow:‖hess‖", "stable:‖hess‖"]
        if d_dirs in (2, 3):
            imgs_hess.append(h_svdfr[z])
            titles_hess.append("svd-free:‖hess‖")
        plot_row(imgs_hess, titles_hess, "Hessian norms")

        plt.show()


if __name__ == "__main__":
    run_tests()
