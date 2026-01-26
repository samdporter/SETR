"""
SVG based plotting utilities for visualising metric trade-offs.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

from .analysis import AggregateResult


def _axis_limits(values: Sequence[float], padding_ratio: float = 0.05) -> tuple[float, float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return 0.0, 1.0
    minimum = min(finite)
    maximum = max(finite)
    if minimum == maximum:
        offset = abs(minimum) * padding_ratio or 1.0
        return minimum - offset, maximum + offset
    padding = (maximum - minimum) * padding_ratio
    return minimum - padding, maximum + padding


def plot_l_curve_svg(
    results: Sequence[AggregateResult],
    output_path: Path,
    x_metric: str = "mean",
    y_metric: str = "coefficient_of_variation",
    log_scale: bool = True,
) -> Path:
    """
    Render an L-curve-like visualisation and persist it as an SVG file.

    The default axes plot ``log10(mean)`` against ``log10(coefficient_of_variation)``.
    """

    if not results:
        raise ValueError("No aggregate results supplied – nothing to plot.")

    usable: list[tuple[AggregateResult, float, float]] = []
    epsilon = 1e-12
    for result in results:
        x_raw = float(result.stats.get(x_metric, math.nan))
        y_raw = float(result.stats.get(y_metric, math.nan))
        if not math.isfinite(x_raw) or not math.isfinite(y_raw):
            continue
        if log_scale:
            if x_raw <= 0 or y_raw <= 0:
                continue
            x_val = math.log10(max(x_raw, epsilon))
            y_val = math.log10(max(y_raw, epsilon))
        else:
            x_val = x_raw
            y_val = y_raw
        usable.append((result, x_val, y_val))

    if not usable:
        raise ValueError("No finite results available for plotting.")

    xs = [entry[1] for entry in usable]
    ys = [entry[2] for entry in usable]

    x_min, x_max = _axis_limits(xs)
    y_min, y_max = _axis_limits(ys)
    width, height = 900, 600
    margin_left, margin_bottom, margin_top, margin_right = 90, 70, 40, 40
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def project_x(value: float) -> float:
        if x_max == x_min:
            return margin_left + plot_width / 2
        return margin_left + (value - x_min) / (x_max - x_min) * plot_width

    def project_y(value: float) -> float:
        if y_max == y_min:
            return margin_top + plot_height / 2
        return margin_top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    svg = ET.Element("svg", attrib={"xmlns": "http://www.w3.org/2000/svg", "width": str(width), "height": str(height)})

    # Axes
    axis_group = ET.SubElement(svg, "g", attrib={"stroke": "#333", "stroke-width": "1.5", "fill": "none"})
    # X-axis
    ET.SubElement(axis_group, "line", attrib={
        "x1": str(margin_left),
        "y1": str(height - margin_bottom),
        "x2": str(width - margin_right),
        "y2": str(height - margin_bottom),
    })
    # Y-axis
    ET.SubElement(axis_group, "line", attrib={
        "x1": str(margin_left),
        "y1": str(margin_top),
        "x2": str(margin_left),
        "y2": str(height - margin_bottom),
    })

    tick_group = ET.SubElement(svg, "g", attrib={"stroke": "#555", "stroke-width": "1"})
    label_group = ET.SubElement(svg, "g", attrib={"fill": "#222", "font-size": "14"})

    tick_count = 6
    for idx in range(tick_count):
        x_value = x_min + (x_max - x_min) * idx / (tick_count - 1)
        x_pos = project_x(x_value)
        ET.SubElement(tick_group, "line", attrib={
            "x1": str(x_pos),
            "y1": str(height - margin_bottom),
            "x2": str(x_pos),
            "y2": str(height - margin_bottom + 8),
        })
        label = f"{x_value:.2f}"
        ET.SubElement(label_group, "text", attrib={
            "x": str(x_pos),
            "y": str(height - margin_bottom + 26),
            "text-anchor": "middle",
        }).text = label

        y_value = y_min + (y_max - y_min) * idx / (tick_count - 1)
        y_pos = project_y(y_value)
        ET.SubElement(tick_group, "line", attrib={
            "x1": str(margin_left - 8),
            "y1": str(y_pos),
            "x2": str(margin_left),
            "y2": str(y_pos),
        })
        label = f"{y_value:.2f}"
        ET.SubElement(label_group, "text", attrib={
            "x": str(margin_left - 12),
            "y": str(y_pos + 4),
            "text-anchor": "end",
        }).text = label

    # Axis labels
    x_axis_label = f"log10({x_metric})" if log_scale else x_metric
    y_axis_label = f"log10({y_metric})" if log_scale else y_metric
    ET.SubElement(label_group, "text", attrib={
        "x": str(margin_left + plot_width / 2),
        "y": str(height - 20),
        "text-anchor": "middle",
        "font-weight": "bold",
    }).text = x_axis_label

    y_label = ET.SubElement(label_group, "text", attrib={
        "x": str(30),
        "y": str(margin_top + plot_height / 2),
        "text-anchor": "middle",
        "font-weight": "bold",
        "transform": f"rotate(-90 {30} {margin_top + plot_height / 2})",
    })
    y_label.text = y_axis_label

    point_group = ET.SubElement(svg, "g", attrib={"fill": "#1f77b4", "stroke": "none"})
    annotation_group = ET.SubElement(svg, "g", attrib={"fill": "#1f77b4", "font-size": "12"})

    for result, x_val, y_val in usable:
        cx = project_x(x_val)
        cy = project_y(y_val)
        ET.SubElement(point_group, "circle", attrib={
            "cx": str(cx),
            "cy": str(cy),
            "r": "4",
        })
        label = f"α={result.alpha:g}, β={result.beta:g}"
        ET.SubElement(annotation_group, "text", attrib={
            "x": str(cx + 6),
            "y": str(cy - 6),
        }).text = label

    tree = ET.ElementTree(svg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


__all__ = ["plot_l_curve_svg"]
