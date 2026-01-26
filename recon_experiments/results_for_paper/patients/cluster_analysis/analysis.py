"""
High level orchestration for analysing parameter sweeps.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .interfile import load_image, get_image_array
from .masking import MaskConfig, create_mask
from .stats import compute_basic_stats, combine_partial_stats


@dataclass
class AnalysisConfig:
    """Configuration for analysing a parameter sweep."""

    root: Path
    image_names: Tuple[str, ...] = ("final_image_0.hv", "final_image_1.hv")
    mask: MaskConfig = field(default_factory=MaskConfig)
    regex: re.Pattern[str] = re.compile(r"alpha_([\d.]+)_beta_([\d.]+)")


@dataclass
class ImageResult:
    alpha: float
    beta: float
    image_name: str
    path: Path
    stats: Dict[str, float]
    mask_method: str


@dataclass
class AggregateResult:
    alpha: float
    beta: float
    stats: Dict[str, float]


def _iter_image_headers(root: Path, image_names: Sequence[str]) -> Iterable[Path]:
    for header in root.rglob("*.hv"):
        if header.name in image_names:
            yield header


def _extract_parameters(path: Path, pattern: re.Pattern[str]) -> Tuple[float, float] | None:
    match = pattern.search(str(path))
    if not match:
        return None
    alpha, beta = match.groups()
    try:
        return float(alpha), float(beta)
    except ValueError:
        return None


def analyse_parameter_grid(config: AnalysisConfig) -> Tuple[List[ImageResult], List[AggregateResult]]:
    """
    Load images beneath ``config.root`` and compute statistics for each
    parameter pair.
    """

    image_results: List[ImageResult] = []
    aggregates: Dict[Tuple[float, float], Dict[str, float]] = {}

    for header_path in sorted(_iter_image_headers(config.root, config.image_names)):
        params = _extract_parameters(header_path.parent, config.regex)
        if params is None:
            continue
        alpha, beta = params
        image = load_image(header_path)
        image_array = get_image_array(image)
        values = image_array.ravel().tolist()
        mask = create_mask(values, config.mask)
        stats = compute_basic_stats(values, mask)
        image_results.append(
            ImageResult(
                alpha=alpha,
                beta=beta,
                image_name=header_path.name,
                path=header_path,
                stats=stats,
                mask_method=config.mask.method,
            )
        )
        key = (alpha, beta)
        if key not in aggregates:
            aggregates[key] = dict(stats)
        else:
            aggregates[key] = combine_partial_stats(aggregates[key], stats)

    aggregate_results = [
        AggregateResult(alpha=k[0], beta=k[1], stats=v) for k, v in sorted(aggregates.items())
    ]
    return image_results, aggregate_results


def _stats_to_row(stats: Dict[str, float], exclude: Sequence[str] = ()) -> Dict[str, float]:
    return {k: v for k, v in stats.items() if k not in exclude}


def write_results_csv(
    image_results: Sequence[ImageResult],
    aggregate_results: Sequence[AggregateResult],
    output_directory: Path,
) -> Tuple[Path, Path]:
    """
    Persist analysis results as CSV tables.

    Returns the paths of the per-image and aggregated CSV files.
    """

    output_directory.mkdir(parents=True, exist_ok=True)
    image_csv = output_directory / "voxel_metrics.csv"
    parameter_csv = output_directory / "parameter_metrics.csv"

    image_headers = ["alpha", "beta", "image_name", "mask_method"]
    image_stats_keys = sorted(
        key for key in image_results[0].stats.keys() if key not in {"m2"}  # stable order
    ) if image_results else []

    with image_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(image_headers + image_stats_keys)
        for result in image_results:
            row = [
                result.alpha,
                result.beta,
                result.image_name,
                result.mask_method,
            ]
            for key in image_stats_keys:
                row.append(result.stats[key])
            writer.writerow(row)

    aggregate_headers = ["alpha", "beta"]
    aggregate_stats_keys = sorted(
        key for key in aggregate_results[0].stats.keys() if key not in {"m2"}
    ) if aggregate_results else []

    with parameter_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(aggregate_headers + aggregate_stats_keys)
        for result in aggregate_results:
            row = [result.alpha, result.beta]
            for key in aggregate_stats_keys:
                row.append(result.stats[key])
            writer.writerow(row)

    return image_csv, parameter_csv


__all__ = [
    "AnalysisConfig",
    "ImageResult",
    "AggregateResult",
    "analyse_parameter_grid",
    "write_results_csv",
]
