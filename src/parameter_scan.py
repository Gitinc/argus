import os
import sys
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import algo_funcs as algo
import ctc_reformat as ctc
from ctc_metrics import evaluate_sequence
import pandas as pd
import matplotlib as mpl


WORK_DIR = os.getcwd()
PATH_OF_SRC_MODUL = WORK_DIR.replace("notebooks", "src")

YOUR_DATA_PATH = r""
OUTPUT_PATH = r""   # scratch dir, overwritten every iteration
RESULTS_CSV = r""
IMAGE_CHANNEL = 1

# =========================
# SWEEP RANGES  (edit these)
# =========================
CELL_SIZE_VALUES = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
CLIP_PERCENTILE_VALUES = [83, 85, 87, 89, 91, 93, 95, 97, 99]   # "clip threshold" == clip_percentile
#CELL_SIZE_VALUES = [34]
#CLIP_PERCENTILE_VALUES = [60, 95]   # "clip threshold" == clip_percentile

sys.path.append(PATH_OF_SRC_MODUL)

def plot_parameter_surfaces(
    data,
    x_col="cell_size",
    y_col="clip_percentile",
    metrics=("DET", "TRA"),
    x_label="Cell size (px)",
    y_label="Clip percentile",
    z_label="Score",
    cmap="cividis",
    base_fontsize=9,
    shared_colorbar=True,
    elev=28, azim=-122,
    save_path=None,
    dpi=300,
):
    """3D surfaces of swept metrics over a (y_col x x_col) grid. Returns (fig, axes, optima)."""
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "axes.titlesize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
    })
 
    df = pd.read_csv(data) if isinstance(data, str) else data
    metrics = list(metrics)
    grids = {m: df.pivot(index=y_col, columns=x_col, values=m) for m in metrics}
 
    if shared_colorbar:
        vmin = min(np.nanmin(g.values) for g in grids.values())
        vmax = max(np.nanmax(g.values) for g in grids.values())
    else:
        vmin = vmax = None
 
    fig = plt.figure(figsize=(3.6 * len(metrics), 3.4), constrained_layout=True)
    axes, optima, surf = [], {}, None
 
    for i, metric in enumerate(metrics):
        grid = grids[metric]
        X, Y = np.meshgrid(grid.columns.values.astype(float),
                           grid.index.values.astype(float))
        Z = grid.values
        ax = fig.add_subplot(1, len(metrics), i + 1, projection="3d")
 
        lvmin = vmin if vmin is not None else np.nanmin(Z)
        lvmax = vmax if vmax is not None else np.nanmax(Z)
        surf = ax.plot_surface(
            X, Y, Z, cmap=cmap, vmin=lvmin, vmax=lvmax,
            rstride=1, cstride=1, linewidth=0.2, edgecolor="0.3",
            antialiased=True, shade=False,
        )
 
        iy, ix = np.unravel_index(np.nanargmax(Z), Z.shape)
        optima[metric] = (grid.index[iy], grid.columns[ix], np.nanmax(Z))
 
        ax.set_xlabel(x_label, labelpad=2)
        ax.set_ylabel(y_label, labelpad=2)
        ax.set_zlabel(z_label, labelpad=2)
        ax.set_title(metric, pad=0)
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(pad=1)
        # lighten panes for a cleaner look
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_edgecolor("0.85")
            pane.pane.set_alpha(0.4)
        ax.grid(True, linewidth=0.3)
        axes.append(ax)
 
    if shared_colorbar:
        cb = fig.colorbar(surf, ax=axes, fraction=0.025, pad=0.04)
        cb.set_label(z_label)
        cb.outline.set_linewidth(0.6)
        cb.ax.tick_params(labelsize=base_fontsize - 2, width=0.6, length=2)
    
    plt.show()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if save_path.endswith(".png"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")
    return fig, axes, optima

def plot_parameter_metrics(
    data,
    x_col="cell_size",
    y_col="clip_percentile",
    metrics=("DET", "TRA"),       # SEG dropped
    x_label="Cell size",
    y_label="Clip threshold",
    cmap="cividis",
    base_fontsize=9,
    mark_max=False,               # no best-point marker
    shared_colorbar=True,         # single colorbar with a common scale
    cbar_label="Score",
    save_path=None,
    dpi=300,
):
    """Heatmaps of swept metrics over a (y_col x x_col) parameter grid.
 
    Returns (fig, axes, optima). optima: {metric: (y_value, x_value, max_value)}.
    """
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "axes.titlesize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
 
    df = pd.read_csv(data) if isinstance(data, str) else data
    metrics = list(metrics)
    grids = {m: df.pivot(index=y_col, columns=x_col, values=m) for m in metrics}
 
    # common color scale across panels when sharing the colorbar
    if shared_colorbar:
        vmin = min(np.nanmin(g.values) for g in grids.values())
        vmax = max(np.nanmax(g.values) for g in grids.values())
    else:
        vmin = vmax = None
 
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(3.3 * len(metrics), 3.3),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    optima = {}
    im = None
 
    for ax, metric in zip(axes, metrics):
        grid = grids[metric]
        im = ax.imshow(grid.values, origin="lower", aspect="auto",
                       cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
 
        iy, ix = np.unravel_index(np.nanargmax(grid.values), grid.shape)
        optima[metric] = (grid.index[iy], grid.columns[ix], np.nanmax(grid.values))
        if mark_max:
            ax.scatter(ix, iy, marker="*", s=140, c="white",
                       edgecolors="black", linewidths=0.6, zorder=3)
 
        n_cols = len(grid.columns)
        n_rows = len(grid.index)
        xt = range(0, n_cols, 2)
        yt = range(0, n_rows, 2)
        ax.set_xticks(list(xt))
        ax.set_xticklabels(grid.columns[::2])
        ax.set_yticks(list(yt))
        ax.set_yticklabels(grid.index[::2])

        ax.set_xlabel(x_label)
        if metric == metrics[0]:
            ax.set_ylabel(y_label)
        ax.set_title(metric)
        ax.tick_params(direction="out", length=2.5)
 
        if not shared_colorbar:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.outline.set_linewidth(0.6)
            cb.ax.tick_params(labelsize=base_fontsize - 2, width=0.6, length=2)
 
    if shared_colorbar:
        cb = fig.colorbar(im, ax=list(axes), fraction=0.046, pad=0.02)
        cb.set_label(cbar_label)
        cb.outline.set_linewidth(0.6)
        cb.ax.tick_params(labelsize=base_fontsize - 2, width=0.6, length=2)

    plt.show()
 
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, axes, optima

def main():
    IMAGE_PATHS_PATTERN = rf'{YOUR_DATA_PATH}/t*.tif'
    algo.IMAGE_CHANNEL = IMAGE_CHANNEL
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # =========================
    # LOAD IMAGES  (once)
    # =========================
    images_raw = algo.read_images(
        directory=IMAGE_PATHS_PATTERN,
        is_gray_scale=True,
    )
    images_raw = algo.batch_rescale_rgb_to_255(images_raw)
    images_unchanged = images_raw.copy()   # raw rescaled, used for optical-flow tracking

    records = []

    for clip_percentile in CLIP_PERCENTILE_VALUES:

        images_pre = algo.clip_images(
            images=[img.copy() for img in images_raw],   # fresh copy every iteration
            clip_percentile=clip_percentile,
        )
        images_pre = algo.wavelet_denoise_images_all(images=images_pre)

        for cell_size in CELL_SIZE_VALUES:
            print(f"clip_percentile={clip_percentile}  cell_size={cell_size} ...")

            # ----- LOCATE -----
            edges_list, mask_list, labels_list = algo.locate_all_cells_centroids(
                images=images_pre,
                cell_size=cell_size,
                min_distance=cell_size,
            )

            dict_list_original, average_flow, tracked_mask_original, flows = algo.track_points_optical_flow(
                images=np.array(images_unchanged, dtype=np.uint8),
                all_points=edges_list,
                all_labels=labels_list,
                labeled_masks=mask_list,
                max_distance=cell_size/2,
                max_occlusion_frames=5,
            )

            # ----- TRACK -----
            dict_list_refined, tracked_mask_refined, refinement_info = algo.refine_tracks_by_gap_closing(
                dict_list=dict_list_original,
                images=np.array(images_unchanged, dtype=np.uint8),
                labeled_masks=mask_list,
                max_distance=cell_size/2,
                max_gap=5,
                gap_penalty=0.25,
                direction_weight=0.25,
                no_link_cost=2.5,
                flows=flows,
            )

            ctc.save_track_data_to_file(
                dict_list_refined,
                tracked_mask_refined,
                OUTPUT_PATH,
            )

            res = evaluate_sequence(
                OUTPUT_PATH,
                f"{YOUR_DATA_PATH}_GT",
            )

            records.append({
                "clip_percentile": clip_percentile,
                "cell_size": cell_size,
                "DET": res["DET"],
                "TRA": res["TRA"],
                "SEG": res["SEG"],
            })

            print(f"  DET={res['DET']}  TRA={res['TRA']}  SEG={res['SEG']}")

            # write after every combination so a crash mid-sweep keeps finished results
            pd.DataFrame.from_records(records).to_csv(RESULTS_CSV, index=False)

    print(f"\nDone. Saved {len(records)} rows to {RESULTS_CSV}")
    

if __name__ == "__main__":
    main()
    fig, axes, optima = plot_parameter_metrics(r"", base_fontsize=16, save_path="")
    fig, axes, optima = plot_parameter_surfaces(r"")
    for m, (y, x, v) in optima.items():
        print(f"{m}: max {v:.4f} at {y}, {x}")
    