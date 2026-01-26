"""
Command line interface for the analysis helpers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import AnalysisConfig, analyse_parameter_grid, write_results_csv
from .masking import MaskConfig
from .plotting import plot_l_curve_svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse cluster reconstruction sweeps.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory that contains alpha/beta sweep folders.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=["final_image_0.hv", "final_image_1.hv"],
        help="Image header file names to include in the analysis.",
    )
    parser.add_argument(
        "--mask-method",
        choices=["fraction", "percentile", "absolute", "nonzero"],
        default="fraction",
        help="Mask generation strategy (default: fraction of maximum).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.5,
        help="Fraction of the global maximum used by the 'fraction' mask method.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile threshold for the 'percentile' mask method.",
    )
    parser.add_argument(
        "--absolute-threshold",
        type=float,
        default=None,
        help="Absolute threshold for the 'absolute' mask method.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_results"),
        help="Directory used for CSV outputs and plots.",
    )
    parser.add_argument(
        "--x-metric",
        default="mean",
        help="Metric used for the L-curve x-axis (default: mean).",
    )
    parser.add_argument(
        "--y-metric",
        default="coefficient_of_variation",
        help="Metric used for the L-curve y-axis (default: coefficient_of_variation).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable log10 scaling for the L-curve axes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_config = MaskConfig(
        method=args.mask_method,
        fraction=args.fraction,
        percentile=args.percentile,
        absolute_threshold=args.absolute_threshold,
    )
    config = AnalysisConfig(root=args.root.resolve(), image_names=tuple(args.images), mask=mask_config)
    image_results, aggregate_results = analyse_parameter_grid(config)
    csv_image, csv_parameter = write_results_csv(image_results, aggregate_results, args.output)
    plot_path = plot_l_curve_svg(
        aggregate_results,
        args.output / "l_curve.svg",
        x_metric=args.x_metric,
        y_metric=args.y_metric,
        log_scale=not args.no_log,
    )
    print(f"Wrote per-image metrics to {csv_image}")
    print(f"Wrote parameter-level metrics to {csv_parameter}")
    print(f"Wrote L-curve plot to {plot_path}")


if __name__ == "__main__":
    main()
