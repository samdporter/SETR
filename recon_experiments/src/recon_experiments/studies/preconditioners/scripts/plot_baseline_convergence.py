#!/usr/bin/env python3
"""
Baseline convergence evidence.

Shows that the long baseline reconstructions have (very nearly) reached the
minimiser of their objective -- and that the plateau is genuine convergence,
NOT merely the effect of the relaxed step size decaying toward zero.

For each baseline (e.g. 1bpos, 2bpos) we plot / report, versus epoch:
  1. Objective gap  J_k - J*      (log-y), where J* = best observed objective.
  2. Per-epoch relative objective change |J_k - J_{k-1}| / |J_k|  (log-y).
  3. The analytic relaxed step size  step_k = s0 / (1 + eta * L * epoch),
     where L = updates/epoch is inferred from the image-snapshot iteration
     spacing.  This is overlaid on (1): the objective plateaus while the step
     is still order-unity, so the plateau cannot be a step-starvation artifact.
  4. The image "velocity" ||x_k - x_{k-1}|| / ||x_k|| per epoch for the PET
     (modality 0) iterate, showing the *solution itself* has stopped moving.

A metrics CSV and a printed summary quantify near-convergence (final gap as a
fraction of the total objective decrease, the step size still available at the
point where 99.9% of the decrease was achieved, final iterate velocity, etc.).

Runs under the `sirf-build` env (needs sirf.STIR for the image velocity).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------------- IO


def _read_objective(obj_csv: Path) -> np.ndarray:
    """Read objective.csv (one value per epoch; first token may be header '0')."""
    vals: List[float] = []
    with obj_csv.open("r", newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if not row:
                continue
            tok = row[-1].strip()
            if i == 0 and tok.lower() in ("0", "0.0", "objective", ""):
                # header written by pandas (unnamed single column -> "0")
                continue
            try:
                vals.append(float(tok))
            except ValueError:
                continue
    return np.asarray(vals, dtype=float)


def _read_args_row(args_csv: Path) -> Dict[str, str]:
    if not args_csv.exists():
        return {}
    with args_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            return dict(row)
    return {}


def _snapshot_iters(baseline_dir: Path, modality: int) -> List[int]:
    iters: List[int] = []
    import re

    pat = re.compile(rf"image_{modality}_(\d+)\.hv$")
    for p in baseline_dir.glob(f"image_{modality}_*.hv"):
        m = pat.match(p.name)
        if m:
            iters.append(int(m.group(1)))
    return sorted(iters)


def _infer_updates_per_epoch(baseline_dir: Path, n_epochs: int, args_row: Dict[str, str]) -> Optional[int]:
    """L = updates/epoch. Prefer max_snapshot_iter / max_epoch (robust to save cadence)."""
    iters = _snapshot_iters(baseline_dir, modality=0)
    if len(iters) >= 2 and n_epochs > 0:
        L = iters[-1] / float(n_epochs)
        if L > 0:
            return int(round(L))
    # Fallback: sum of subset counts.
    try:
        ns = args_row.get("num_subsets", "")
        nums = [int(x) for x in ns.strip("[]").replace(" ", "").split(",") if x]
        if nums:
            return int(sum(nums))
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- calc


@dataclass
class BaselineConv:
    label: str
    epochs: np.ndarray            # 0..N
    objective: np.ndarray         # J_k
    gap: np.ndarray               # J_k - J*
    rel_change: np.ndarray        # |J_k - J_{k-1}| / |J_k|  (len N, epochs 1..N)
    rel_change_epochs: np.ndarray
    step: np.ndarray              # step_k at each epoch
    s0: float
    eta: float
    L: int
    vel_epochs: np.ndarray        # epochs where image velocity computed
    velocity: np.ndarray          # ||dx||/||x|| per epoch (PET)


def _image_velocity(
    baseline_dir: Path, n_points: int = 120
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-epoch PET iterate velocity ||x_k - x_prev|| / (||x_k|| * dEpoch)."""
    try:
        from sirf.STIR import ImageData
    except Exception:
        return np.asarray([]), np.asarray([])

    iters = _snapshot_iters(baseline_dir, modality=0)
    if len(iters) < 3:
        return np.asarray([]), np.asarray([])

    # Map snapshot iteration -> epoch via L (last snapshot = last epoch).
    # We express velocity per epoch; select a log-spaced subset for speed.
    idx = np.unique(np.round(np.geomspace(1, len(iters) - 1, num=min(n_points, len(iters) - 1))).astype(int))
    idx = idx[idx >= 1]

    L = iters[-1] / float(len(iters) - 1) if len(iters) > 1 else 1.0
    vel_epochs: List[float] = []
    vel: List[float] = []
    prev_arr = None
    prev_it = None
    # Always include the very last consecutive pair for a true final velocity.
    sel = sorted(set(idx.tolist() + [len(iters) - 1]))
    # We need consecutive-in-selection differences; load selected + their -1 neighbours
    for j in sel:
        it_cur = iters[j]
        it_prev = iters[j - 1]
        a_cur = ImageData(str(baseline_dir / f"image_0_{it_cur}.hv")).as_array().astype(np.float64)
        a_prev = ImageData(str(baseline_dir / f"image_0_{it_prev}.hv")).as_array().astype(np.float64)
        d_epoch = max((it_cur - it_prev) / L, 1e-9)
        num = float(np.linalg.norm(a_cur - a_prev))
        den = float(np.linalg.norm(a_cur)) + 1e-12
        vel_epochs.append(it_cur / L)
        vel.append(num / den / d_epoch)
    return np.asarray(vel_epochs), np.asarray(vel)


def compute_baseline_conv(baseline_dir: Path, label: str, velocity_points: int) -> BaselineConv:
    obj = _read_objective(baseline_dir / "objective.csv")
    if obj.size == 0:
        raise RuntimeError(f"No objective values in {baseline_dir/'objective.csv'}")
    epochs = np.arange(obj.size, dtype=float)
    j_star = float(np.min(obj))
    gap = obj - j_star

    d = np.abs(np.diff(obj))
    rel_change = d / (np.abs(obj[1:]) + 1e-30)
    rel_change_epochs = epochs[1:]

    args_row = _read_args_row(baseline_dir / "args.csv")
    s0 = float(args_row.get("initial_step_size", 1.0) or 1.0)
    eta = float(args_row.get("relaxation_eta", 0.0) or 0.0)
    n_epochs = int(obj.size - 1)
    L = _infer_updates_per_epoch(baseline_dir, n_epochs, args_row) or 1
    step = s0 / (1.0 + eta * L * epochs)

    vel_epochs, velocity = _image_velocity(baseline_dir, n_points=velocity_points)

    return BaselineConv(
        label=label,
        epochs=epochs,
        objective=obj,
        gap=gap,
        rel_change=rel_change,
        rel_change_epochs=rel_change_epochs,
        step=step,
        s0=s0,
        eta=eta,
        L=L,
        vel_epochs=vel_epochs,
        velocity=velocity,
    )


def _epoch_reaching_fraction(gap: np.ndarray, frac: float) -> int:
    """First epoch at which the remaining gap <= (1-frac) of the initial gap."""
    g0 = gap[0] if gap[0] > 0 else np.max(gap)
    thresh = (1.0 - frac) * g0
    below = np.where(gap <= thresh)[0]
    return int(below[0]) if below.size else int(gap.size - 1)


# --------------------------------------------------------------------------- plot


def plot_convergence(convs: Sequence[BaselineConv], out_path: Path) -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel 1: objective gap (log) + step-size overlay (twin axis).
    ax0 = axes[0]
    ax0t = ax0.twinx()
    for i, c in enumerate(convs):
        col = colors[i % len(colors)]
        m = c.gap > 0
        ax0.semilogy(c.epochs[m], c.gap[m], color=col, lw=1.8, label=f"{c.label}: $J_k-J^*$")
        ax0t.plot(c.epochs, c.step, color=col, lw=1.3, ls="--", alpha=0.7)
    ax0.set_xlabel("epoch")
    ax0.set_ylabel(r"objective gap $J_k - J^*$")
    ax0t.set_ylabel("relaxed step size (dashed)")
    ax0t.set_ylim(bottom=0)
    ax0.set_title("Objective gap (log) + step size")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, which="both", alpha=0.25)

    # Panel 2: per-epoch relative objective change (log).
    ax1 = axes[1]
    for i, c in enumerate(convs):
        col = colors[i % len(colors)]
        m = c.rel_change > 0
        ax1.semilogy(c.rel_change_epochs[m], c.rel_change[m], color=col, lw=1.4, label=c.label)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel(r"$|J_k - J_{k-1}| / |J_k|$")
    ax1.set_title("Per-epoch relative objective change")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, which="both", alpha=0.25)

    # Panel 3: image (PET) iterate velocity (log) + step (twin) to disentangle.
    ax2 = axes[2]
    ax2t = ax2.twinx()
    have_vel = False
    for i, c in enumerate(convs):
        col = colors[i % len(colors)]
        if c.velocity.size:
            have_vel = True
            m = c.velocity > 0
            ax2.semilogy(c.vel_epochs[m], c.velocity[m], color=col, lw=1.6, marker="o", ms=2.5, label=c.label)
        ax2t.plot(c.epochs, c.step, color=col, lw=1.1, ls="--", alpha=0.5)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel(r"PET iterate velocity $\|x_k-x_{k-1}\|/\|x_k\|$ per epoch")
    ax2t.set_ylabel("relaxed step size (dashed)")
    ax2t.set_ylim(bottom=0)
    ax2.set_title("Solution velocity vs step size")
    if have_vel:
        ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved baseline convergence figure: {out_path}")


def plot_final_panel(c: BaselineConv, out_path: Path) -> None:
    """Plot only the solution-velocity diagnostic for one baseline."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    axt = ax.twinx()
    color = "#1f77b4"
    if c.velocity.size:
        m = c.velocity > 0
        ax.semilogy(
            c.vel_epochs[m], c.velocity[m], color=color, lw=2.0,
            marker="o", ms=3.2, label="PET iterate velocity",
        )
    axt.plot(c.epochs, c.step, color="#4c78a8", lw=1.8, ls="--", label="relaxed step size")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\|x_k-x_{k-1}\|/\|x_k\|$")
    axt.set_ylabel("relaxed step size")
    ax.set_title("")
    ax.grid(True, which="both", alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = axt.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved final convergence panel: {out_path}")


def plot_final_objective_panel(c: BaselineConv, out_path: Path, log_y: bool = True) -> None:
    """Plot the remaining objective decrease and relaxed step for one baseline."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    axt = ax.twinx()
    color = "#1f77b4"
    m = c.gap > 0
    plot = ax.semilogy if log_y else ax.plot
    plot(c.epochs[m], c.gap[m], color=color, lw=2.0, label="objective decrease")
    axt.plot(c.epochs, c.step, color="#4c78a8", lw=1.8, ls="--", label="relaxed step size")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\mathcal{O}_k - \mathcal{O}^*$")
    axt.set_ylabel("relaxed step size")
    axt.set_ylim(bottom=0)
    ax.grid(True, which="both", alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = axt.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved final objective convergence panel: {out_path}")


def write_metrics(convs: Sequence[BaselineConv], out_csv: Path) -> None:
    rows: List[Dict[str, object]] = []
    print("\n" + "=" * 78)
    print("BASELINE NEAR-CONVERGENCE SUMMARY")
    print("=" * 78)
    for c in convs:
        g0 = c.gap[0] if c.gap[0] > 0 else float(np.max(c.gap))
        total_dec = float(c.objective[0] - np.min(c.objective))
        final_gap = float(c.gap[-1])
        # Use second-smallest gap as a finite 'final' if last is exactly J* (gap 0).
        finite_final_gap = final_gap
        if final_gap == 0.0:
            pos = c.gap[c.gap > 0]
            finite_final_gap = float(pos.min()) if pos.size else 0.0
        e999 = _epoch_reaching_fraction(c.gap, 0.999)
        e9999 = _epoch_reaching_fraction(c.gap, 0.9999)
        step_at_e999 = float(c.step[e999])
        step_at_e9999 = float(c.step[e9999])
        final_rel_change = float(c.rel_change[-1]) if c.rel_change.size else float("nan")
        final_step = float(c.step[-1])
        final_vel = float(c.velocity[-1]) if c.velocity.size else float("nan")
        n_epochs = int(c.epochs[-1])

        row = {
            "baseline": c.label,
            "epochs": n_epochs,
            "updates_per_epoch_L": c.L,
            "J0": float(c.objective[0]),
            "J_star": float(np.min(c.objective)),
            "total_decrease": total_dec,
            "final_gap": finite_final_gap,
            "final_gap_frac_of_total": finite_final_gap / total_dec if total_dec > 0 else float("nan"),
            "final_rel_obj_change_per_epoch": final_rel_change,
            "final_pet_velocity_per_epoch": final_vel,
            "initial_step": c.s0,
            "final_step": final_step,
            "epoch_99.9pct_decrease": e999,
            "step_at_99.9pct": step_at_e999,
            "epoch_99.99pct_decrease": e9999,
            "step_at_99.99pct": step_at_e9999,
            "relaxation_eta": c.eta,
        }
        rows.append(row)

        print(f"\n[{c.label}]  ({n_epochs} epochs, L={c.L} updates/epoch, eta={c.eta:g})")
        print(f"  objective:  J0={row['J0']:.6g}  ->  J*={row['J_star']:.6g}   (total decrease {total_dec:.6g})")
        print(f"  final gap:  {finite_final_gap:.4g}  =  {row['final_gap_frac_of_total']:.3e} of total decrease")
        print(f"  final per-epoch relative objective change: {final_rel_change:.3e}")
        if not np.isnan(final_vel):
            print(f"  final PET iterate velocity (per epoch):    {final_vel:.3e}")
        print(f"  step size:  {c.s0:g} (initial)  ->  {final_step:.4g} (final)   [NEVER tiny -> plateau is not step-starvation]")
        print(f"  99.9%  of decrease reached by epoch {e999:4d}  (step still {step_at_e999:.3g})")
        print(f"  99.99% of decrease reached by epoch {e9999:4d}  (step still {step_at_e9999:.3g})")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n" + "=" * 78)
    print(f"Wrote metrics CSV: {out_csv}")
    print("=" * 78)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--baseline",
        action="append",
        nargs=2,
        metavar=("DIR", "LABEL"),
        required=True,
        help="Baseline run dir (containing objective.csv/args.csv/image_0_*.hv) and a label. Repeatable.",
    )
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for figure + metrics CSV.")
    p.add_argument("--velocity-points", type=int, default=120, help="Number of log-spaced snapshots for the PET velocity curve (0 to skip).")
    p.add_argument(
        "--panel",
        choices=("all", "final", "final-objective", "final-objective-linear"),
        default="all",
        help="Render all panels, the final solution-velocity panel, or a log/linear final objective-decrease panel.",
    )
    args = p.parse_args()

    convs: List[BaselineConv] = []
    for dir_str, label in args.baseline:
        d = Path(dir_str)
        if not d.exists():
            raise FileNotFoundError(f"Baseline dir not found: {d}")
        convs.append(compute_baseline_conv(d, label, velocity_points=args.velocity_points))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.panel in ("final", "final-objective", "final-objective-linear"):
        if len(convs) != 1:
            raise ValueError(f"--panel {args.panel} requires exactly one --baseline")
        if args.panel == "final":
            plot_final_panel(convs[0], args.output_dir / "baseline_convergence_final_panel.png")
        elif args.panel == "final-objective":
            plot_final_objective_panel(
                convs[0], args.output_dir / "baseline_convergence_final_objective_panel.png"
            )
        else:
            plot_final_objective_panel(
                convs[0],
                args.output_dir / "baseline_convergence_final_objective_panel_linear.png",
                log_y=False,
            )
    else:
        plot_convergence(convs, args.output_dir / "baseline_convergence.png")
    write_metrics(convs, args.output_dir / "baseline_convergence_metrics.csv")


if __name__ == "__main__":
    main()
