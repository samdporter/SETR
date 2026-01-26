"""
Visualization utilities for creating GIFs and plotting reconstructions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import cmasher
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches


def create_gradient_legend_handle(colormap, label, n_segments=50):
    """
    Create a custom legend handle showing a color gradient.

    Args:
        colormap: Matplotlib colormap to use for gradient
        label: Label text for the legend entry
        n_segments: Number of color segments in gradient (higher = smoother)

    Returns:
        tuple: (handle, label) for use with ax.legend(handles=...)
    """
    # Create a list of colored rectangles that will be stacked horizontally
    colors = [colormap(i / (n_segments - 1)) for i in range(n_segments)]

    # Create a patch collection representing the gradient
    # We'll use a custom handler to render this
    class GradientPatch(mpatches.Patch):
        def __init__(self, colors, **kwargs):
            self.colors = colors
            super().__init__(**kwargs)

    return GradientPatch(colors), label


class GradientLegendHandler:
    """Custom legend handler for gradient patches."""

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        """Create the legend artist (the colored box in the legend)."""
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height

        # Create thin rectangles for each color segment
        n_colors = len(orig_handle.colors)
        segment_width = width / n_colors

        patches = []
        for i, color in enumerate(orig_handle.colors):
            rect = Rectangle(
                (x0 + i * segment_width, y0),
                segment_width,
                height,
                facecolor=color,
                edgecolor='none',
                transform=handlebox.get_transform()
            )
            handlebox.add_artist(rect)
            patches.append(rect)

        # Return the first patch (matplotlib requirement)
        return patches[0]


def make_multi_curve_constant_beta_gif(
    lesion,
    mask_type,
    beta_values,
    output_path,
    alpha_beta_data_pet,
    alpha_beta_data_spect,
    hkem_data,
    recon_type='PET',
    reference_image='vendor',
    hkem_iters=None,
    dpi=130,
    frame_ms=400,
    show_counter=True,
    colormap=None,
    hkem_colormap=None,
    figsize=(10, 7),
):
    """
    Create GIF showing multiple constant-beta curves (one per beta value).
    Each curve shows varying alpha values, progressively revealed point by point.

    Args:
        lesion: Lesion name (e.g., 'lesion_4')
        mask_type: Mask type (e.g., 'original')
        beta_values: List of beta values to show as separate curves
        output_path: Path to save GIF
        alpha_beta_data_pet: DataFrame with PET alpha/beta data
        alpha_beta_data_spect: DataFrame with SPECT alpha/beta data
        hkem_data: DataFrame with HKEM data
        recon_type: 'PET' or 'SPECT'
        reference_image: Reference image name
        hkem_iters: List of HKEM iteration numbers to show (e.g., [9, 18, 27])
        dpi: Resolution
        frame_ms: Frame duration in milliseconds
        show_counter: Show beta threshold counter
        colormap: Colormap for beta curves (default: cmasher.tropical)
        hkem_colormap: Colormap for HKEM points (default: viridis)
        figsize: Figure size as (width, height) tuple (default: (10, 7))
    """
    if hkem_iters is None:
        hkem_iters = []

    # Select data
    ab_data = alpha_beta_data_pet if recon_type == 'PET' else alpha_beta_data_spect

    # Filter for this lesion and mask
    ab_panel = ab_data[
        (ab_data['reference_image'] == reference_image) &
        (ab_data['lesion'] == lesion) &
        (ab_data['mask_type'] == mask_type)
    ].copy()

    if len(ab_panel) == 0:
        print(f"WARNING: No data found for {lesion}, {mask_type}")
        return

    # Get HKEM data
    hkem_panel = hkem_data[
        (hkem_data['reference_image'] == reference_image) &
        (hkem_data['lesion'] == lesion) &
        (hkem_data['mask_type'] == mask_type) &
        (hkem_data['reconstruction'] == 'HKEM')
    ].copy()

    print(f"DEBUG: HKEM data for {lesion}/{mask_type}/{reference_image}:")
    print(f"  Total HKEM rows (after filtering): {len(hkem_panel)}")
    if len(hkem_panel) > 0:
        print(f"  Available iterations: {sorted(hkem_panel['hkem_iter'].unique())}")
        print(f"  HKEM point values:")
        for idx, row in hkem_panel.iterrows():
            print(f"    iter {row['hkem_iter']}: reconstruction='{row['reconstruction']}', TBR={row['tbr_mean']:.2f}, CoV={row['background_cov_pct']:.2f}")
    print(f"  Requested iterations: {hkem_iters}")

    # Setup colors - perceptually uniform colormaps
    beta_colormap = colormap if colormap is not None else cmasher.tropical
    hkem_cmap = hkem_colormap if hkem_colormap is not None else plt.get_cmap('viridis')

    # Compute axis limits from all data
    all_tbr = list(ab_panel['tbr_mean'].values) + list(hkem_panel['tbr_mean'].values)
    all_cov = list(ab_panel['background_cov_pct'].values) + list(hkem_panel['background_cov_pct'].values)

    tbr_range = max(all_tbr) - min(all_tbr)
    cov_range = max(all_cov) - min(all_cov)
    xlim = (min(all_tbr) - 0.05 * tbr_range, max(all_tbr) + 0.05 * tbr_range)
    ylim = (min(all_cov) - 0.05 * cov_range, max(all_cov) + 0.05 * cov_range)

    print(f"DEBUG: Axis limits - x: {xlim}, y: {ylim}")

    # Get all unique beta values for animation sequence (NOT alpha!)
    all_betas_sorted = sorted(ab_panel['beta'].unique())

    # Create frames - frame N shows first N beta values for ALL constant-beta curves
    frames = []

    for num_betas_to_show in range(1, len(all_betas_sorted) + 1):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Get the first N beta values to show in this frame
        betas_to_include = all_betas_sorted[:num_betas_to_show]

        # Plot HKEM points (without individual labels - will use gradient legend)
        for i, hkem_iter in enumerate(hkem_iters):
            hkem_subset = hkem_panel[hkem_panel['hkem_iter'] == hkem_iter]
            if len(hkem_subset) > 0:
                ax.scatter(
                    hkem_subset['tbr_mean'],
                    hkem_subset['background_cov_pct'],
                    s=150,
                    alpha=0.85,
                    marker='o',
                    edgecolors='black',
                    linewidths=1.5,
                    color=hkem_cmap(i / max(len(hkem_iters)-1, 1)),
                    zorder=4
                )

        # Plot ALL constant-beta curves (all curves visible from frame 1)
        # Each frame adds the next beta point (alpha point on each curve) to ALL curves
        for beta_idx, beta_val in enumerate(beta_values):
            # Get ALL data for this specific beta value
            beta_curve_data = ab_panel[ab_panel['beta'] == beta_val].copy()
            if len(beta_curve_data) == 0:
                continue

            # Only show points where beta is in the first N beta values
            beta_curve_data = beta_curve_data[beta_curve_data['beta'].isin(betas_to_include)]
            if len(beta_curve_data) == 0:
                continue

            # Sort by alpha for proper line drawing
            beta_curve_data = beta_curve_data.sort_values('alpha')

            ax.plot(
                beta_curve_data['tbr_mean'],
                beta_curve_data['background_cov_pct'],
                linestyle='-',
                linewidth=2.0,
                marker='D',
                markersize=7,
                color=beta_colormap(beta_idx / max(len(beta_values)-1, 1)),
                markeredgecolor='black',
                markeredgewidth=1.0,
                alpha=0.9,
                label=f'β={beta_val}',
                zorder=3
            )

        # Create custom legend with gradient handle for HKEM
        handles, labels = ax.get_legend_handles_labels()

        # Add gradient legend entry for HKEM if we have iterations
        if len(hkem_iters) > 0:
            hkem_gradient_handle, hkem_label = create_gradient_legend_handle(
                hkem_cmap,
                f'HKEM iters [{hkem_iters[0]}:{hkem_iters[-1]}]'
            )
            handles.insert(0, hkem_gradient_handle)
            labels.insert(0, hkem_label)

        # Formatting
        ax.set_xlabel('TBR (Mean)', fontsize=14, fontweight='bold')
        ax.set_ylabel('CoV (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'{lesion} - {mask_type} mask (Constant β, varying α)', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        # Create legend with custom handler for gradient patches
        legend = ax.legend(
            handles=handles,
            labels=labels,
            loc='best',
            fontsize=10,
            framealpha=0.9,
            handler_map={type(handles[0]): GradientLegendHandler()} if len(hkem_iters) > 0 else None
        )

        if show_counter:
            current_max_beta = betas_to_include[-1]
            ax.text(
                0.02, 0.98,
                f'β ≤ {current_max_beta:.3g}',
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=22,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.85, edgecolor='black', linewidth=2)
            )

        plt.tight_layout()

        # Render to image
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())
        frames.append(frame.copy())
        plt.close(fig)

    # Save GIF
    iio.imwrite(output_path, frames, plugin="pillow", duration=frame_ms, loop=0)
    print(f"✓ Saved constant-β GIF to: {output_path} ({len(frames)} frames)")


def make_multi_curve_constant_alpha_gif(
    lesion,
    mask_type,
    alpha_values,
    output_path,
    alpha_beta_data_pet,
    alpha_beta_data_spect,
    hkem_data,
    recon_type='PET',
    reference_image='vendor',
    hkem_iters=None,
    dpi=130,
    frame_ms=400,
    show_counter=True,
    colormap=None,
    hkem_colormap=None,
    figsize=(10, 7),
):
    """
    Create GIF showing multiple constant-alpha curves (one per alpha value).
    Each curve shows varying beta values, progressively revealed point by point.

    Args:
        lesion: Lesion name (e.g., 'lesion_4')
        mask_type: Mask type (e.g., 'original')
        alpha_values: List of alpha values to show as separate curves
        output_path: Path to save GIF
        alpha_beta_data_pet: DataFrame with PET alpha/beta data
        alpha_beta_data_spect: DataFrame with SPECT alpha/beta data
        hkem_data: DataFrame with HKEM data
        recon_type: 'PET' or 'SPECT'
        reference_image: Reference image name
        hkem_iters: List of HKEM iteration numbers to show (e.g., [9, 18, 27])
        dpi: Resolution
        frame_ms: Frame duration in milliseconds
        show_counter: Show beta threshold counter
        colormap: Colormap for alpha curves (default: cmasher.tropical)
        hkem_colormap: Colormap for HKEM points (default: viridis)
        figsize: Figure size as (width, height) tuple (default: (10, 7))
    """
    if hkem_iters is None:
        hkem_iters = []

    # Select data
    ab_data = alpha_beta_data_pet if recon_type == 'PET' else alpha_beta_data_spect

    # Filter for this lesion and mask
    ab_panel = ab_data[
        (ab_data['reference_image'] == reference_image) &
        (ab_data['lesion'] == lesion) &
        (ab_data['mask_type'] == mask_type)
    ].copy()

    if len(ab_panel) == 0:
        print(f"WARNING: No data found for {lesion}, {mask_type}")
        return

    # Get HKEM data
    hkem_panel = hkem_data[
        (hkem_data['reference_image'] == reference_image) &
        (hkem_data['lesion'] == lesion) &
        (hkem_data['mask_type'] == mask_type) &
        (hkem_data['reconstruction'] == 'HKEM')
    ].copy()

    print(f"DEBUG: HKEM data for {lesion}/{mask_type}/{reference_image}:")
    print(f"  Total HKEM rows: {len(hkem_panel)}")
    if len(hkem_panel) > 0:
        print(f"  Available iterations: {sorted(hkem_panel['hkem_iter'].unique())}")
    print(f"  Requested iterations: {hkem_iters}")

    # Setup colors - perceptually uniform colormaps
    alpha_colormap = colormap if colormap is not None else cmasher.tropical
    hkem_cmap = hkem_colormap if hkem_colormap is not None else plt.get_cmap('viridis')

    # Compute axis limits from all data
    all_tbr = list(ab_panel['tbr_mean'].values) + list(hkem_panel['tbr_mean'].values)
    all_cov = list(ab_panel['background_cov_pct'].values) + list(hkem_panel['background_cov_pct'].values)

    # Debug: print HKEM point values
    if len(hkem_panel) > 0:
        print(f"DEBUG: HKEM point values:")
        for idx, row in hkem_panel.iterrows():
            print(f"  iter {row['hkem_iter']}: TBR={row['tbr_mean']:.2f}, CoV={row['background_cov_pct']:.2f}")

    tbr_range = max(all_tbr) - min(all_tbr)
    cov_range = max(all_cov) - min(all_cov)
    xlim = (min(all_tbr) - 0.05 * tbr_range, max(all_tbr) + 0.05 * tbr_range)
    ylim = (min(all_cov) - 0.05 * cov_range, max(all_cov) + 0.05 * cov_range)

    print(f"DEBUG: Axis limits - x: {xlim}, y: {ylim}")

    # Get all unique beta values for animation sequence (increasing beta)
    all_betas_sorted = sorted(ab_panel['beta'].unique())

    # Create frames - frame N shows first N beta values for ALL alpha curves
    frames = []

    for num_betas_to_show in range(1, len(all_betas_sorted) + 1):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Get the first N beta values to show in this frame
        betas_to_include = all_betas_sorted[:num_betas_to_show]
        current_max_beta = betas_to_include[-1]

        # Plot HKEM points (without individual labels - will use gradient legend)
        for i, hkem_iter in enumerate(hkem_iters):
            hkem_subset = hkem_panel[hkem_panel['hkem_iter'] == hkem_iter]
            if len(hkem_subset) > 0:
                ax.scatter(
                    hkem_subset['tbr_mean'],
                    hkem_subset['background_cov_pct'],
                    s=150,
                    alpha=0.85,
                    marker='o',
                    edgecolors='black',
                    linewidths=1.5,
                    color=hkem_cmap(i / max(len(hkem_iters)-1, 1)),
                    zorder=4
                )

        # Plot ALL constant-alpha curves (all curves visible from frame 1)
        # Each frame adds the next beta point to ALL curves simultaneously
        for alpha_idx, alpha_val in enumerate(alpha_values):
            # Get ALL data for this specific alpha value
            alpha_curve_data = ab_panel[ab_panel['alpha'] == alpha_val].copy()
            if len(alpha_curve_data) == 0:
                continue

            # Only show the first N beta values (not threshold, but count)
            alpha_curve_data = alpha_curve_data[alpha_curve_data['beta'].isin(betas_to_include)]
            if len(alpha_curve_data) == 0:
                continue

            # Sort by beta for proper line drawing
            alpha_curve_data = alpha_curve_data.sort_values('beta')

            ax.plot(
                alpha_curve_data['tbr_mean'],
                alpha_curve_data['background_cov_pct'],
                linestyle='-',
                linewidth=2.0,
                marker='D',
                markersize=7,
                color=alpha_colormap(alpha_idx / max(len(alpha_values)-1, 1)),
                markeredgecolor='black',
                markeredgewidth=1.0,
                alpha=0.9,
                label = rf'$\beta_{{PET}} = {alpha_val}$',
                zorder=3
            )

        # Create custom legend with gradient handle for HKEM
        handles, labels = ax.get_legend_handles_labels()

        # Add gradient legend entry for HKEM if we have iterations
        if len(hkem_iters) > 0:
            hkem_gradient_handle, hkem_label = create_gradient_legend_handle(
                hkem_cmap,
                f'SHKEM iters [{hkem_iters[0]}:{hkem_iters[-1]}]'
            )
            handles.insert(0, hkem_gradient_handle)
            labels.insert(0, hkem_label)

        # Formatting
        ax.set_xlabel('TBR (Mean)', fontsize=14, fontweight='bold')
        ax.set_ylabel('CoV (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'{lesion} - {mask_type} mask (Constant α, varying β)', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        # Create legend with custom handler for gradient patches
        legend = ax.legend(
            handles=handles,
            labels=labels,
            loc='best',
            fontsize=10,
            framealpha=0.9,
            handler_map={type(handles[0]): GradientLegendHandler()} if len(hkem_iters) > 0 else None
        )

        if show_counter:
            ax.text(
                0.02, 0.98,
                f'β ≤ {current_max_beta:.3g}',
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=22,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='cyan', alpha=0.85, edgecolor='black', linewidth=2)
            )

        plt.tight_layout()

        # Render to image
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())
        frames.append(frame.copy())
        plt.close(fig)

    # Save GIF
    iio.imwrite(output_path, frames, plugin="pillow", duration=frame_ms, loop=0)
    print(f"✓ Saved constant-α GIF to: {output_path} ({len(frames)} frames)")


def save_final_frame_as_png(final_frame, output_path):
    """
    Save the final frame from a GIF as a static PNG file.

    Args:
        final_frame: The final frame array (RGBA) from make_multi_curve_* functions
        output_path: Path to save the PNG file
    """
    if final_frame is None:
        print(f"⚠ No frame to save")
        return

    iio.imwrite(output_path, final_frame)
    print(f"✓ Saved final frame to: {output_path}")


def make_publication_plot_constant_beta(
    lesion,
    mask_type,
    beta_values,
    output_path,
    alpha_beta_data_pet,
    alpha_beta_data_spect,
    hkem_data,
    vendor_data_pet=None,
    recon_type='PET',
    reference_image='vendor',
    hkem_iters=None,
    dpi=300,
    figsize=(10, 7),
    colormap=None,
    hkem_colormap=None,
):
    """
    Create a clean, publication-ready plot showing all constant-beta curves.
    No counter box, no title - suitable for inclusion in papers.

    Args:
        lesion: Lesion name (e.g., 'lesion_4')
        mask_type: Mask type (e.g., 'original')
        beta_values: List of beta values to show as separate curves
        output_path: Path to save PNG
        alpha_beta_data_pet: DataFrame with PET alpha/beta data
        alpha_beta_data_spect: DataFrame with SPECT alpha/beta data
        hkem_data: DataFrame with HKEM data
        recon_type: 'PET' or 'SPECT'
        reference_image: Reference image name
        hkem_iters: List of HKEM iteration numbers to show (e.g., [9, 18, 27])
        dpi: Resolution (default 300 for publication quality)
        figsize: Figure size as (width, height) tuple (default: (10, 7))
        colormap: Colormap for beta curves (default: cmasher.tropical)
        hkem_colormap: Colormap for HKEM points (default: viridis)
    """
    if hkem_iters is None:
        hkem_iters = []

    # Select data
    ab_data = alpha_beta_data_pet if recon_type == 'PET' else alpha_beta_data_spect

    # Filter for this lesion and mask
    ab_panel = ab_data[
        (ab_data['reference_image'] == reference_image) &
        (ab_data['lesion'] == lesion) &
        (ab_data['mask_type'] == mask_type)
    ].copy()

    if len(ab_panel) == 0:
        print(f"WARNING: No data found for {lesion}, {mask_type}")
        return

    # Get HKEM data
    hkem_panel = hkem_data[
        (hkem_data['reference_image'] == reference_image) &
        (hkem_data['lesion'] == lesion) &
        (hkem_data['mask_type'] == mask_type) &
        (hkem_data['reconstruction'] == 'HKEM')
    ].copy()

    # Setup colors - perceptually uniform colormaps
    beta_colormap = colormap if colormap is not None else cmasher.tropical
    hkem_cmap = hkem_colormap if hkem_colormap is not None else plt.get_cmap('viridis')

    # Compute axis limits from all data
    all_tbr = list(ab_panel['tbr_mean'].values) + list(hkem_panel['tbr_mean'].values)
    all_cov = list(ab_panel['background_cov_pct'].values) + list(hkem_panel['background_cov_pct'].values)

    tbr_range = max(all_tbr) - min(all_tbr)
    cov_range = max(all_cov) - min(all_cov)
    xlim = (min(all_tbr) - 0.05 * tbr_range, max(all_tbr) + 0.05 * tbr_range)
    ylim = (min(all_cov) - 0.05 * cov_range, max(all_cov) + 0.05 * cov_range)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot HKEM points (without individual labels)
    for i, hkem_iter in enumerate(hkem_iters):
        hkem_subset = hkem_panel[hkem_panel['hkem_iter'] == hkem_iter]
        if len(hkem_subset) > 0:
            ax.scatter(
                hkem_subset['tbr_mean'],
                hkem_subset['background_cov_pct'],
                s=75,
                alpha=0.5,
                marker='o',
                color=hkem_cmap(i / max(len(hkem_iters)-1, 1)),
                zorder=4
            )

    # Plot vendor point (Q.Clear) if available
    if vendor_data_pet is not None:
        vendor_panel = vendor_data_pet[
            (vendor_data_pet['reference_image'] == reference_image) &
            (vendor_data_pet['lesion'] == lesion) &
            (vendor_data_pet['mask_type'] == mask_type)
        ].copy()

        if len(vendor_panel) > 0:
            ax.scatter(
                vendor_panel['tbr_mean'],
                vendor_panel['background_cov_pct'],
                s=400,
                alpha=1.0,
                marker='*',
                edgecolors='black',
                linewidths=2.5,
                c='#FFD700',  # Gold color (hex)
                label='Vendor (Q.Clear)',
                zorder=6
            )

    # Plot ALL constant-beta curves
    for beta_idx, beta_val in enumerate(beta_values):
        beta_curve_data = ab_panel[ab_panel['beta'] == beta_val].copy()
        if len(beta_curve_data) == 0:
            continue

        # Sort by alpha for proper line drawing
        beta_curve_data = beta_curve_data.sort_values('alpha')

        ax.plot(
            beta_curve_data['tbr_mean'],
            beta_curve_data['background_cov_pct'],
            linestyle='-',
            linewidth=2.0,
            marker='D',
            markersize=7,
            color=beta_colormap(beta_idx / max(len(beta_values)-1, 1)),
            markeredgecolor='black',
            markeredgewidth=1.0,
            alpha=0.9,
            label = rf'$\beta_{{SPECT}} = {beta_val}$',
            zorder=3
        )

    # Create custom legend with gradient handle for HKEM
    handles, labels = ax.get_legend_handles_labels()

    # Add gradient legend entry for HKEM if we have iterations
    if len(hkem_iters) > 0:
        hkem_gradient_handle, hkem_label = create_gradient_legend_handle(
            hkem_cmap,
            f'SHKEM iters [{hkem_iters[0]}:{hkem_iters[-1]}]'
        )
        handles.insert(0, hkem_gradient_handle)
        labels.insert(0, hkem_label)

    # Formatting - clean for publication
    ax.set_xlabel('TBR Mean', fontsize=14, fontweight='bold')
    ax.set_ylabel('CoV ', fontsize=14, fontweight='bold')
    # No title for publication
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Create legend with custom handler for gradient patches
    legend = ax.legend(
        handles=handles,
        labels=labels,
        loc='best',
        fontsize=10,
        framealpha=0.9,
        handler_map={type(handles[0]): GradientLegendHandler()} if len(hkem_iters) > 0 else None
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved publication plot to: {output_path}")
    return fig


def make_publication_plot_constant_alpha(
    lesion,
    mask_type,
    alpha_values,
    output_path,
    alpha_beta_data_pet,
    alpha_beta_data_spect,
    hkem_data,
    vendor_data_pet=None,
    recon_type='PET',
    reference_image='vendor',
    hkem_iters=None,
    dpi=300,
    figsize=(10, 7),
    colormap=None,
    hkem_colormap=None,
):
    """
    Create a clean, publication-ready plot showing all constant-alpha curves.
    No counter box, no title - suitable for inclusion in papers.

    Args:
        lesion: Lesion name (e.g., 'lesion_4')
        mask_type: Mask type (e.g., 'original')
        alpha_values: List of alpha values to show as separate curves
        output_path: Path to save PNG
        alpha_beta_data_pet: DataFrame with PET alpha/beta data
        alpha_beta_data_spect: DataFrame with SPECT alpha/beta data
        hkem_data: DataFrame with HKEM data
        recon_type: 'PET' or 'SPECT'
        reference_image: Reference image name
        hkem_iters: List of HKEM iteration numbers to show (e.g., [9, 18, 27])
        dpi: Resolution (default 300 for publication quality)
        figsize: Figure size as (width, height) tuple (default: (10, 7))
        colormap: Colormap for alpha curves (default: cmasher.tropical)
        hkem_colormap: Colormap for HKEM points (default: viridis)
    """
    if hkem_iters is None:
        hkem_iters = []

    # Select data
    ab_data = alpha_beta_data_pet if recon_type == 'PET' else alpha_beta_data_spect

    # Filter for this lesion and mask
    ab_panel = ab_data[
        (ab_data['reference_image'] == reference_image) &
        (ab_data['lesion'] == lesion) &
        (ab_data['mask_type'] == mask_type)
    ].copy()

    if len(ab_panel) == 0:
        print(f"WARNING: No data found for {lesion}, {mask_type}")
        return

    # Get HKEM data
    hkem_panel = hkem_data[
        (hkem_data['reference_image'] == reference_image) &
        (hkem_data['lesion'] == lesion) &
        (hkem_data['mask_type'] == mask_type) &
        (hkem_data['reconstruction'] == 'HKEM')
    ].copy()

    # Setup colors - perceptually uniform colormaps
    alpha_colormap = colormap if colormap is not None else cmasher.tropical
    hkem_cmap = hkem_colormap if hkem_colormap is not None else plt.get_cmap('viridis')

    # Compute axis limits from all data
    all_tbr = list(ab_panel['tbr_mean'].values) + list(hkem_panel['tbr_mean'].values)
    all_cov = list(ab_panel['background_cov_pct'].values) + list(hkem_panel['background_cov_pct'].values)

    tbr_range = max(all_tbr) - min(all_tbr)
    cov_range = max(all_cov) - min(all_cov)
    xlim = (min(all_tbr) - 0.05 * tbr_range, max(all_tbr) + 0.05 * tbr_range)
    ylim = (min(all_cov) - 0.05 * cov_range, max(all_cov) + 0.05 * cov_range)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot HKEM points (without individual labels)
    for i, hkem_iter in enumerate(hkem_iters):
        hkem_subset = hkem_panel[hkem_panel['hkem_iter'] == hkem_iter]
        if len(hkem_subset) > 0:
            ax.scatter(
                hkem_subset['tbr_mean'],
                hkem_subset['background_cov_pct'],
                s=75,
                alpha=0.85,
                marker='o',
                color=hkem_cmap(i / max(len(hkem_iters)-1, 1)),
                zorder=4
            )

    # Plot vendor point (Q.Clear) if available
    if vendor_data_pet is not None:
        vendor_panel = vendor_data_pet[
            (vendor_data_pet['reference_image'] == reference_image) &
            (vendor_data_pet['lesion'] == lesion) &
            (vendor_data_pet['mask_type'] == mask_type)
        ].copy()

        if len(vendor_panel) > 0:
            ax.scatter(
                vendor_panel['tbr_mean'],
                vendor_panel['background_cov_pct'],
                s=400,
                alpha=1.0,
                marker='*',
                edgecolors='black',
                linewidths=2.5,
                c='#FFD700',  # Gold color (hex)
                label='Vendor (Q.Clear)',
                zorder=6
            )

    # Plot ALL constant-alpha curves
    for alpha_idx, alpha_val in enumerate(alpha_values):
        alpha_curve_data = ab_panel[ab_panel['alpha'] == alpha_val].copy()
        if len(alpha_curve_data) == 0:
            continue

        # Sort by beta for proper line drawing
        alpha_curve_data = alpha_curve_data.sort_values('beta')

        ax.plot(
            alpha_curve_data['tbr_mean'],
            alpha_curve_data['background_cov_pct'],
            linestyle='-',
            linewidth=2.0,
            marker='D',
            markersize=7,
            color=alpha_colormap(alpha_idx / max(len(alpha_values)-1, 1)),
            markeredgecolor='black',
            markeredgewidth=1.0,
            alpha=0.9,
            label=rf"$\beta_{{PET}} = {alpha_val}$",
            zorder=3
        )

    # Create custom legend with gradient handle for HKEM
    handles, labels = ax.get_legend_handles_labels()

    # Add gradient legend entry for HKEM if we have iterations
    if len(hkem_iters) > 0:
        hkem_gradient_handle, hkem_label = create_gradient_legend_handle(
            hkem_cmap,
            f'HKEM iters [{hkem_iters[0]}:{hkem_iters[-1]}]'
        )
        handles.insert(0, hkem_gradient_handle)
        labels.insert(0, hkem_label)

    # Formatting - clean for publication
    ax.set_xlabel('TBR Mean', fontsize=14, fontweight='bold')
    ax.set_ylabel('CoV', fontsize=14, fontweight='bold')
    # No title for publication
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Create legend with custom handler for gradient patches
    legend = ax.legend(
        handles=handles,
        labels=labels,
        loc='best',
        fontsize=10,
        framealpha=0.9,
        handler_map={type(handles[0]): GradientLegendHandler()} if len(hkem_iters) > 0 else None
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved publication plot to: {output_path}")
    return fig


def plot_reconstruction_single_view(
    recon_arr,
    att_arr,
    center,
    output_dir,
    voxel_sizes,
    title="",
    vmax=None,
    mask_arrs=None,
    save_png=False,
    show_colorbar=True,
    lesion_colors=None,
):
    """
    Plot single coronal view of a reconstruction.

    Args:
        recon_arr: numpy array of reconstruction (already cropped if needed)
        att_arr: numpy array of attenuation map (already cropped if needed)
        center: (z, y, x) coordinates of lesion center (relative to the passed arrays)
        output_dir: Directory to save output
        voxel_sizes: Tuple of (vz, vy, vx) voxel sizes for aspect ratio
        title: Plot title
        vmax: Maximum color scale value
        mask_arrs: List of numpy boolean arrays to overlay as contours (already cropped)
        save_png: Whether to save as PNG file
        show_colorbar: Whether to show the colorbar (default: True)
        lesion_colors: List of colors for each mask (default: ['lime', 'red', 'blue', 'magenta', 'cyan'])
    """
    z_center, y_center, x_center = center
    vz, vy, vx = voxel_sizes

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    alpha=0.6
    if att_arr is None:
        att_arr = np.zeros_like(recon_arr)
        alpha=1

    # Coronal slice (y fixed)
    slice_idx = (slice(None), y_center, slice(None))
    ax.imshow(att_arr[slice_idx], cmap='gray', aspect=vz/vx)
    im = ax.imshow(
        recon_arr[slice_idx],
        cmap=cmasher.fall,
        aspect=vz/vx,
        alpha=alpha,
        vmin=0,
        vmax=vmax
    )
    ax.axis('off')
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Optional mask contours
    if mask_arrs:
        if lesion_colors is None:
            lesion_colors = ['lime', 'red', 'blue', 'magenta', 'cyan']
        
        for i, mask_arr in enumerate(mask_arrs):
            ax.contour(
                mask_arr[slice_idx],
                colors=lesion_colors[i % len(lesion_colors)],
                linewidths=2,
                alpha=1
            )

    if title:
        ax.set_title(title, fontsize=14, pad=10)

    if save_png and title:
        filename = f"{title.replace(' ', '_').replace('=', '').replace(',', '')}.png"
        plt.savefig(f"{output_dir}/{filename}", dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir}/{filename}")

    plt.tight_layout()
    plt.show()


def plot_reconstruction_three_views(
    recon_img,
    attenuation_img,
    center,
    output_dir,
    title="",
    vmax=None,
    masks=None,
    save_png=False,
    show_colorbar=True,
    lesion_colors=None,
):
    """
    Plot three orthogonal views (axial, coronal, sagittal) of a reconstruction.

    Args:
        recon_img: SIRF ImageData reconstruction
        attenuation_img: SIRF ImageData attenuation map
        center: (z, y, x) coordinates of lesion center
        title: Plot title
        vmax: Maximum color scale value
        masks: List of SIRF ImageData masks to overlay as contours
        save_png: Whether to save as PNG file
        show_colorbar: Whether to show the colorbar (default: True)
        lesion_colors: List of colors for each mask, same length as masks (default: ['lime', 'red', 'blue', 'magenta', 'cyan'])
    """
    z_center, y_center, x_center = center

    recon_arr = recon_img.as_array()
    att_arr = attenuation_img.as_array()

    # Get voxel sizes for aspect ratios
    vz, vy, vx = recon_img.voxel_sizes()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Axial (z fixed)
    slice_axial = (z_center, slice(None), slice(None))
    axes[0].imshow(att_arr[slice_axial], cmap='gray', aspect=vy/vx)
    im0 = axes[0].imshow(
        recon_arr[slice_axial],
        cmap=cmasher.fall,
        aspect=vy/vx,
        alpha=0.6,
        vmin=0,
        vmax=vmax
    )
    axes[0].set_title('Axial', fontsize=12)
    axes[0].axis('off')
    if show_colorbar:
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Coronal (y fixed)
    slice_coronal = (slice(None), y_center, slice(None))
    axes[1].imshow(att_arr[slice_coronal], cmap='gray', aspect=vz/vx)
    im1 = axes[1].imshow(
        recon_arr[slice_coronal],
        cmap=cmasher.fall,
        aspect=vz/vx,
        alpha=0.6,
        vmin=0,
        vmax=vmax
    )
    axes[1].set_title('Coronal', fontsize=12)
    axes[1].axis('off')
    if show_colorbar:
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Sagittal (x fixed)
    slice_sagittal = (slice(None), slice(None), x_center)
    axes[2].imshow(att_arr[slice_sagittal], cmap='gray', aspect=vz/vy)
    im2 = axes[2].imshow(
        recon_arr[slice_sagittal],
        cmap=cmasher.fall,
        aspect=vz/vy,
        alpha=0.6,
        vmin=0,
        vmax=vmax
    )
    axes[2].set_title('Sagittal', fontsize=12)
    axes[2].axis('off')
    if show_colorbar:
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Optional mask contours on all views
    if masks:
        if lesion_colors is None:
            lesion_colors = ['lime', 'red', 'blue', 'magenta', 'cyan']
        for i, mask in enumerate(masks):
            mask_arr = mask.as_array().astype(bool)
            axes[0].contour(
                mask_arr[slice_axial],
                colors=lesion_colors[i % len(lesion_colors)],
                linewidths=2,
                alpha=1
            )
            axes[1].contour(
                mask_arr[slice_coronal],
                colors=lesion_colors[i % len(lesion_colors)],
                linewidths=2,
                alpha=1
            )
            axes[2].contour(
                mask_arr[slice_sagittal],
                colors=lesion_colors[i % len(lesion_colors)],
                linewidths=2,
                alpha=1
            )

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    if save_png and title:
        filename = f"{title.replace(' ', '_').replace('=', '').replace(',', '')}.png"
        plt.savefig(f"{output_dir}/{filename}", dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir}/{filename}")

    plt.tight_layout()
    plt.show()
