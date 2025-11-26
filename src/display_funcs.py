import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from matplotlib.colors import to_rgba

def calculate_mean_square_displacement(paths):
    msd_list = []
    for path in paths:
        displacements = abs(path - path[0])
        squared_displacements = np.sum(displacements**2, axis=1)
        msd = np.cumsum(squared_displacements) / np.arange(1, len(squared_displacements) + 1)
        msd_list.append(msd)
    return msd_list

def plot_mean_square_displacement_filtered(msd_list, msd_list_filtered, starting_indices, starting_indices_filtered, title='Mean Square Displacement of Cells'):
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

    for msd, starting_index in zip(msd_list_filtered, starting_indices_filtered):
        x_values = np.arange(starting_index, starting_index + len(msd))
        plt.plot(x_values, msd, alpha=0.5, color='red', linewidth=1)
        
    # Compute combined MSD
    max_length = max(len(msd) for msd in msd_list)
    padded_msd_list = [np.pad(msd, (0, max_length - len(msd)), 'edge') for msd in msd_list]
    combined_msd = np.nanmean(padded_msd_list, axis=0)
    sem_msd = np.nanstd(padded_msd_list, axis=0) / np.sqrt(len(msd_list))  # standard error
    
    x_combined = np.arange(max_length)
    
    # Plot mean with error shading
    plt.plot(x_combined, combined_msd, color='black', linewidth=2.5, label='Avg. MSD')
    
    max_length = max(len(msd) for msd in msd_list_filtered)
    padded_msd_list = [np.pad(msd, (0, max_length - len(msd)), 'edge') for msd in msd_list_filtered]
    combined_msd = np.nanmean(padded_msd_list, axis=0)
    sem_msd = np.nanstd(padded_msd_list, axis=0) / np.sqrt(len(msd_list_filtered))  # standard error
    
    x_combined = np.arange(max_length)
    
    # Plot mean with error shading
    plt.plot(x_combined, combined_msd, color='red', linewidth=2.5, label='Avg. MSD > 1000')
    
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

def subtract_mean_channel(images, axis=0):
    images_array = np.array(images)
    mean_img = np.mean(images_array[..., IMAGE_CHANNEL], axis=axis)

    for i in range(len(images)):
        images[i][..., IMAGE_CHANNEL] = np.abs(images[i][..., IMAGE_CHANNEL] - mean_img)

    return images, mean_img

def plot_normalized_paths(
    paths,
    figsize=(9, 9),
    line_width=2.0,
    random_colors=False,
    title="Normalized Paths",
    show_points=False,
    point_size=10
):
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.size": 22,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 1.2,
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
    })

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True, dpi=120)

    # Plot each normalized path
    for i, path in enumerate(paths):
        if path is None:
            continue
        arr = np.asarray(path, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            continue

        arr = arr - arr[0]  # normalize to (0,0)
        x, y = arr[:, 0], arr[:, 1]

        ax.plot(x, y, lw=line_width, label=f"Path {i+1}")
        # Optional points along the path
        if show_points:
            ax.scatter(x, y, s=point_size, facecolor="none", edgecolor=c, linewidths=0.8)

    # Axes styling
    ax.set_xlabel("Δx", fontsize=22)
    ax.set_ylabel("Δy", fontsize=22)
    ax.set_title(title, pad=10, fontsize=22)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(x=0.06, y=0.06)

    # Clean spines & reference axes
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.axhline(0, color="0.2", lw=0.8, alpha=0.35)
    ax.axvline(0, color="0.2", lw=0.8, alpha=0.35)

    # Grid (major + minor)
    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.25)
    ax.set_xlim((-120,130))
    ax.set_ylim((-120,120))

    # Tick appearance
    ax.tick_params(axis="both", which="major", labelsize=20)

    return fig, ax