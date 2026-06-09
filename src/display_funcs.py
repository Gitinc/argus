import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
import os

def calculate_mean_square_displacement(paths):
    msd_list = []
    for path in paths:
        displacements = abs(path - path[0])
        squared_displacements = np.sum(displacements**2, axis=1)
        msd = np.cumsum(squared_displacements) / np.arange(1, len(squared_displacements) + 1)
        msd_list.append(msd)
    return msd_list

def _group_mean_sem(msd_list):
    """Edge-pad ragged MSD curves to common length; return x, mean, sem."""
    max_len = max(len(m) for m in msd_list)
    padded = [np.pad(np.asarray(m, float), (0, max_len - len(m)), "edge") for m in msd_list]
    padded = np.vstack(padded)
    mean = np.nanmean(padded, axis=0)
    sem = np.nanstd(padded, axis=0) / np.sqrt(padded.shape[0])
    return np.arange(max_len), mean, sem
 
 
def plot_mean_square_displacement_filtered(
    msd_list,
    msd_list_filtered,
    starting_indices,
    starting_indices_filtered,
    title=None,                  # None for publication; caption carries the title
    figsize=(3.5, 2.8),          # single column (~89 mm); use (7.0, 5.0) for double
    base_fontsize=8,
    x_units="h",                 # "frame", "min", or "h"
    frame_interval=29.0,         # minutes per frame
    pixel_size=0.125,            # \u00b5m per pixel; MSD scales as pixel_size**2
    y_units="\u00b5m$^2$",       # MSD unit after conversion
    filtered_label="High-motility cells",
    save_path=None,
    dpi=300,
):
    """Publication-quality MSD plot: faint per-cell curves + group mean \u00b1 SEM."""
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "axes.titlesize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
 
    # refined two-group colors: neutral baseline, single muted accent
    c_all_ind, c_all_mean = "0.7", "0.15"          # light gray traces, near-black mean
    c_flt_ind, c_flt_mean = "#e3a6a1", "#c0392b"   # light / strong muted red
 
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
 
    area = pixel_size ** 2  # pixel^2 -> physical area units
 
    # frame index -> time on the x-axis
    if x_units == "min":
        t_scale, x_label = frame_interval, "Lag time (min)"
    elif x_units == "h":
        t_scale, x_label = frame_interval / 60.0, "Time (h)"
    else:  # "frame"
        t_scale, x_label = 1.0, "Frame index"
 
    # faint individual curves
    for msd, s in zip(msd_list, starting_indices):
        ax.plot(np.arange(s, s + len(msd)) * t_scale, np.asarray(msd, float) * area,
                color=c_all_ind, lw=0.5, alpha=0.5, zorder=1)
    for msd, s in zip(msd_list_filtered, starting_indices_filtered):
        ax.plot(np.arange(s, s + len(msd)) * t_scale, np.asarray(msd, float) * area,
                color=c_flt_ind, lw=0.5, alpha=0.6, zorder=2)
 
    # group means + SEM bands
    xa, ma, sa = _group_mean_sem(msd_list)
    ma, sa = ma * area, sa * area
    ax.fill_between(xa * t_scale, ma - sa, ma + sa, color=c_all_mean, alpha=0.18, lw=0, zorder=3)
    ax.plot(xa * t_scale, ma, color=c_all_mean, lw=1.6, label="All cells", zorder=5)
 
    xf, mf, sf = _group_mean_sem(msd_list_filtered)
    mf, sf = mf * area, sf * area
    ax.fill_between(xf * t_scale, mf - sf, mf + sf, color=c_flt_mean, alpha=0.20, lw=0, zorder=4)
    ax.plot(xf * t_scale, mf, color=c_flt_mean, lw=1.6, label=filtered_label, zorder=6)
 
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"MSD ({y_units})")
 
    ax.set_xlim(0, (max(len(xa), len(xf)) - 1) * t_scale)
    ax.margins(y=0.04)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.tick_params(direction="out", length=2.5)
    ax.legend(frameon=False, loc="upper left")
 
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        if save_path.lower().endswith(".svg"):
            fig.savefig(save_path[:-4] + ".pdf", bbox_inches="tight")
    return fig, ax

def plot_mean_square_displacement(msd_list, starting_indices, title='Mean Square Displacement of Cells'):
    # Set a professional style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.linewidth': 1.2,
        'grid.linestyle': '--',
        'grid.alpha': 0.7
    })
    
    plt.figure(figsize=(8, 6))
    
    # Plot individual MSD curves
    for msd, starting_index in zip(msd_list, starting_indices):
        x_values = np.arange(starting_index, starting_index + len(msd))
        plt.plot(x_values, msd, alpha=0.5, color='gray', linewidth=1)
    
    # Compute combined MSD
    max_length = max(len(msd) for msd in msd_list)
    padded_msd_list = [np.pad(msd, (0, max_length - len(msd)), 'edge') for msd in msd_list]
    combined_msd = np.nanmean(padded_msd_list, axis=0)
    sem_msd = np.nanstd(padded_msd_list, axis=0) / np.sqrt(len(msd_list))  # standard error
    
    x_combined = np.arange(max_length)
    
    # Plot mean with error shading
    plt.plot(x_combined, combined_msd, color='black', linewidth=2.5, label='Average MSD')
    plt.fill_between(x_combined, combined_msd - sem_msd, combined_msd + sem_msd,
                     color='black', alpha=0.15)
    
    # Labels and title
    plt.xlabel(r'Frame index', fontsize=16)
    plt.ylabel(r'MSD', fontsize=16)
    plt.title(title, fontsize=16, pad=12)
    
    # Increase tick size
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # Set x-axis limits
    plt.xlim(0, max_length)
    
    # Grid and legend with white box
    plt.grid(True)
    plt.legend(frameon=True, facecolor='white', fontsize=12)
    
    plt.tight_layout()

def plot_tracks_3d_with_image_planes(
    images,
    tracks,
    plane_frames=None,
    n_planes=5,
    image_channel=0,
    point_order="xy",
    track_ids=None,
    min_track_length=1,        # plot all paths by default
    max_tracks=None,           # no cap -> every track is drawn
    plane_alpha=0.25,
    plane_downsample=4,
    time_scale=1.0,
    line_width=1.0,
    show_track_points=False,
    point_size=4,
    track_cmap="turbo",
    contrast_percentiles=(1, 99),
    figsize=(7.0, 6.0),        # bigger render; use (3.5, 2.8) for final single-column print
    elev=28,
    azim=-58,
    pixel_size_um=None,
    frame_interval_min=None,
    save_path=None,
    dpi=300,
):
    """3-D space-time plot optimised for speed and CMPD publication quality.

    Image planes are built as a single vectorised PolyCollection per frame
    (no per-pixel Python loop) and rasterised on save, so rendering is much
    faster while trajectories stay as crisp vector lines.
    Strictly grayscale; no RGB branch.
    """
    import matplotlib
    matplotlib.rcParams.update({
        "font.size":         12,
        "axes.linewidth":    0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size":  2,
        "ytick.major.size":  2,
        "svg.fonttype":      "none",
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })

    from mpl_toolkits.mplot3d.art3d import PolyCollection

    images = np.asarray(images)
    if images.ndim not in (3, 4):
        raise ValueError("images must have shape (T, H, W) or (T, H, W, C).")
    if point_order not in ("xy", "yx"):
        raise ValueError("point_order must be 'xy' or 'yx'.")

    T, H, W = images.shape[:3]
    step = max(int(plane_downsample), 1)

    # ------------------------------------------------------------------ #
    # Select plane frames                                                  #
    # ------------------------------------------------------------------ #
    if plane_frames is None:
        plane_frames = np.linspace(0, T - 1, min(int(n_planes), T), dtype=int)
    plane_frames = np.unique(np.clip(np.asarray(plane_frames, dtype=int), 0, T - 1))

    # ------------------------------------------------------------------ #
    # Track helpers                                                        #
    # ------------------------------------------------------------------ #
    def _get_track_frames(track):
        path = np.asarray(track.get("Path", []), dtype=float)
        if path.ndim != 2 or len(path) == 0:
            return path, np.empty(0, dtype=int)
        frames = track.get("Frames", None)
        if frames is not None and len(frames) == len(path):
            frames = np.asarray(frames, dtype=int)
        else:
            start = int(track.get("StartIndex", 0))
            frames = np.arange(start, start + len(path), dtype=int)
        keep = (frames >= 0) & (frames < T)
        return path[keep], frames[keep]

    # ------------------------------------------------------------------ #
    # Filter tracks (defaults keep everything)                             #
    # ------------------------------------------------------------------ #
    allowed_ids = None if track_ids is None else set(track_ids)
    selected = []
    for idx, track in enumerate(tracks):
        tid = int(track.get("ID", idx))
        if allowed_ids is not None and tid not in allowed_ids:
            continue
        path, frames = _get_track_frames(track)
        if len(path) < min_track_length:
            continue
        selected.append((tid, path, frames))

    if max_tracks is not None and len(selected) > max_tracks:
        selected = sorted(selected, key=lambda x: len(x[1]), reverse=True)[:max_tracks]

    # ------------------------------------------------------------------ #
    # Figure                                                               #
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    gray_cmap = plt.get_cmap("gray")

    # ------------------------------------------------------------------ #
    # Image planes — one vectorised PolyCollection per frame              #
    # ------------------------------------------------------------------ #
    for fidx in plane_frames:
        frame = images[fidx]
        gray = frame[..., image_channel].astype(float) if frame.ndim == 3 else frame.astype(float)

        low, high = np.percentile(gray, contrast_percentiles)
        gray_norm = np.clip((gray - low) / max(high - low, 1e-12), 0, 1)

        gray_ds = gray_norm[::step, ::step]
        rows_ds, cols_ds = gray_ds.shape
        if rows_ds < 2 or cols_ds < 2:
            continue

        # Vectorised quad corners (no Python loop over pixels)
        cc, rr = np.meshgrid(np.arange(cols_ds - 1), np.arange(rows_ds - 1))
        cc = cc.ravel()
        rr = rr.ravel()
        x0, x1 = cc * step, (cc + 1) * step
        y0, y1 = rr * step, (rr + 1) * step

        verts = np.empty((x0.size, 4, 2), dtype=float)
        verts[:, 0, 0], verts[:, 0, 1] = x0, y0
        verts[:, 1, 0], verts[:, 1, 1] = x1, y0
        verts[:, 2, 0], verts[:, 2, 1] = x1, y1
        verts[:, 3, 0], verts[:, 3, 1] = x0, y1

        intensity = gray_ds[:-1, :-1].ravel()
        rgba = gray_cmap(intensity)                       # vectorised colour lookup
        rgba[:, 3] = plane_alpha * (0.15 + 0.85 * intensity)

        poly = PolyCollection(
            verts,
            facecolors=rgba,
            edgecolors="none",
            linewidths=0,
            antialiased=False,   # faster fill for the image quads
            rasterized=True,     # collapse thousands of quads to a raster layer on save
            zorder=0,
        )
        ax.add_collection3d(poly, zs=fidx * time_scale, zdir="z")

    # ------------------------------------------------------------------ #
    # Trajectories                                                         #
    # ------------------------------------------------------------------ #
    cmap_tracks = plt.get_cmap(track_cmap, max(len(selected), 1))

    for i, (tid, path, frames) in enumerate(selected):
        x = path[:, 0] if point_order == "xy" else path[:, 1]
        y = path[:, 1] if point_order == "xy" else path[:, 0]
        z = frames.astype(float) * time_scale
        color = cmap_tracks(i)
        ax.plot(x, y, z, linewidth=line_width, color=color, zorder=5)
        if show_track_points:
            ax.scatter(x, y, z, s=point_size, color=[color], depthshade=False, zorder=6)

    # ------------------------------------------------------------------ #
    # Axes / labels                                                        #
    # ------------------------------------------------------------------ #
    if frame_interval_min is not None:
        ax.set_zlabel("Time [min]", fontsize=12, labelpad=6)
        z_ticks = ax.get_zticks()
        ax.set_zticks(z_ticks)  # pin before relabeling so ticks/labels stay aligned
        ax.set_zticklabels([f"{v * frame_interval_min:.0f}" for v in z_ticks], fontsize=12)
    else:
        ax.set_zlabel("Time [frame]", fontsize=12, labelpad=6)
        ax.tick_params(axis="z", labelsize=6)

    # No numeric values on the spatial axes (keep ticks/grid for reference)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    ax.set_xlim(0, W - 1)
    ax.set_ylim(H - 1, 0)
    ax.set_zlim(0, max((T - 1) * time_scale, 1))

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("0.7")
    ax.yaxis.pane.set_edgecolor("0.7")
    ax.zaxis.pane.set_edgecolor("0.7")
    ax.grid(True, linewidth=0.3, color="0.85")

    ax.view_init(elev=elev, azim=azim)

    # Leave room for the (otherwise clipped) time-axis label/ticks
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.96)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)

    return fig, ax

def plot_paths_on_image(
    images,
    all_tracks,
    frame=0,
    channel=1,
    zoom_x=None,
    zoom_y=None,
    line_width=1.5,
    outline=False,
    cmap_image="gray",
    marker_size=6,
):
    mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    active_tracks = [t for t in all_tracks if t["StartIndex"] <= frame]
    n = len(active_tracks)
    if n <= 10:
        palette = plt.cm.tab10(np.arange(10))
    else:
        palette = np.vstack([
            plt.cm.tab20(np.arange(20)),
            plt.cm.tab20b(np.arange(20)),
            plt.cm.tab20c(np.arange(20)),
        ])
    colors = palette[np.arange(max(n, 1)) % len(palette)]
    img = images[frame][..., channel] if images[frame].ndim == 3 else images[frame]
    if zoom_x is None:
        zoom_x = (0, img.shape[1])
    if zoom_y is None:
        zoom_y = (0, img.shape[0])
    fig, ax = plt.subplots(constrained_layout=True)
    ax.imshow(img, cmap=cmap_image)
    effects = [pe.Stroke(linewidth=line_width + 1.4, foreground="white"),
               pe.Normal()] if outline else None
    marker_effects = [pe.Stroke(linewidth=line_width + 1.0, foreground="white"),
                      pe.Normal()] if outline else None
    for i, track in enumerate(active_tracks):
        frames = track["Frames"]
        path = np.asarray(track["Path"], dtype=float)
        cutoff = sum(1 for f in frames if f <= frame)
        path = path[:cutoff]
        if len(path) < 1:
            continue
        ax.plot(path[:, 0], path[:, 1], lw=line_width, color=colors[i],
                alpha=0.9, solid_capstyle="round", path_effects=effects)
        ax.plot(path[-1, 0], path[-1, 1], marker="x", ms=marker_size,
                mew=line_width, color=colors[i], alpha=0.9,
                path_effects=marker_effects)
    ax.set_xlim(zoom_x)
    ax.set_ylim(zoom_y)
    ax.axis("off")
    return fig, ax

def plot_normalized_paths(
    paths,
    figsize=(3.5, 3.5),          # single journal column (~89 mm). Use (7.0, 7.0) for double.
    base_fontsize=12,            # match body-text size at final print scale
    line_width=0.9,
    color_mode="categorical",    # "categorical" -> distinct color per path; "displacement" -> colorbar
    cmap="viridis",
    pixel_size=0.125,            # um per pixel; coordinates scale linearly
    units="\u00b5m",            # axis-label units after conversion
    axis_limit=None,             # symmetric +/- limit in display units; None -> auto from data
    title=None,                  # leave None for publication; the caption carries the title
    show_points=False,
    point_size=6,
    show_legend=False,
):
    """Plot origin-normalized 2D trajectories as a publication-quality displacement plot.
 
    Returns (fig, ax). Save with fig.savefig('tracks.pdf') for vector output.
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
        "legend.fontsize": base_fontsize - 1,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.labelpad": 2.0,
    })
 
    # --- collect valid, origin-normalized tracks, converted to physical units ---
    tracks = []
    for path in paths:
        if path is None:
            continue
        arr = np.asarray(path, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            continue
        tracks.append((arr[:, :2] - arr[0, :2]) * pixel_size)   # px -> um
    if not tracks:
        raise ValueError("No valid 2D paths to plot.")
 
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
 
    net_disp = np.array([np.hypot(*t[-1]) for t in tracks])
 
    if color_mode == "displacement":
        norm = mpl.colors.Normalize(vmin=net_disp.min(), vmax=net_disp.max())
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        for k in np.argsort(net_disp):
            t = tracks[k]
            ax.plot(t[:, 0], t[:, 1], lw=line_width,
                    color=sm.to_rgba(net_disp[k]), solid_capstyle="round")
            if show_points:
                ax.scatter(t[:, 0], t[:, 1], s=point_size, facecolor="none",
                           edgecolor=sm.to_rgba(net_disp[k]), linewidths=0.5)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(f"Net displacement ({units})")
        cbar.outline.set_linewidth(0.6)
    else:  # categorical: one distinct, muted color per path
        n = len(tracks)
        if n <= 10:
            palette = plt.cm.tab10(np.arange(10))
        else:
            palette = np.vstack([
                plt.cm.tab20(np.arange(20)),
                plt.cm.tab20b(np.arange(20)),
                plt.cm.tab20c(np.arange(20)),
            ])
        colors = palette[np.arange(n) % len(palette)]
        for k, t in enumerate(tracks):
            ax.plot(t[:, 0], t[:, 1], lw=line_width, color=colors[k], alpha=0.9,
                    solid_capstyle="round", label=f"Track {k+1}")
            if show_points:
                ax.scatter(t[:, 0], t[:, 1], s=point_size, facecolor="none",
                           edgecolor=colors[k], linewidths=0.5)
        if show_legend:
            ax.legend(frameon=False, loc="best", ncol=1)
 
    # --- symmetric, isotropic, square axes centred on origin ---
    if axis_limit is None:
        axis_limit = max(np.abs(np.concatenate([t.ravel() for t in tracks]))) * 1.05
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal", adjustable="box")
 
    ax.axhline(0, color="0.6", lw=0.5, zorder=0)
    ax.axvline(0, color="0.6", lw=0.5, zorder=0)
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.5)
 
    ax.set_xlabel(f"\u0394x ({units})")
    ax.set_ylabel(f"\u0394y ({units})")
    if title:
        ax.set_title(title)
 
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(direction="out", length=2.5)
 
    return fig, ax