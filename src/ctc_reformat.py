import numpy as np
import os
import colorsys
from PIL import Image
import tifffile

def generate_unique_colors(n_colors):
    """
    This is a placeholder function. 
    You'll need to implement your own logic to generate n_colors unique colors.
    """
    # Simple example using a color map or algorithm
    colors = []
    for i in range(n_colors):
        # A simple way to generate colors, replace with a better method for more unique colors
        r = (i * 30) % 256
        g = (i * 70) % 256
        b = (i * 110) % 256
        colors.append((r, g, b))
    return colors

def save_track_images(dict_list,images, output_folder = "track_images"):
    start_indices = [entry["StartIndex"] for entry in dict_list]
    all_tracks = [entry["Path"] for entry in dict_list]

    total_frames = len(images)
    shape_frame = images[0].shape[:2]

    img = Image.new('RGB', shape_frame, (0, 0, 0))  # Black background
    pixels = img.load()

    colors = generate_unique_colors(len(all_tracks))

    
    # Iterate through all frames
    for current_image_index in range(total_frames):
        # Create a new black image for the current frame
        img = Image.new('RGB', (shape_frame[1], shape_frame[0]), (0, 0, 0))
        pixels = img.load()

        # Iterate through all tracks
        for i_track in range(len(all_tracks)):
            track_path = all_tracks[i_track]
            start_index = start_indices[i_track]
            color = colors[i_track]

            # Check if the track is active at the current frame
            if start_index <= current_image_index < (start_index + len(track_path)):
                # Get the point for the current frame
                point_index = int(current_image_index - start_index)
                y, x = track_path[point_index]

                # Ensure the coordinates are within the image boundaries
                if 0 <= y < shape_frame[0] and 0 <= x < shape_frame[1]:
                    pixels[x, y] = color

        # Save the generated image
        img.save(os.path.join(output_folder, f"mask{current_image_index:03d}.tif"))

def generate_mask(dict_list, n_frames, output_folder):
    start_indices = [entry["StartIndex"] for entry in dict_list]
    all_tracks = [entry["Path"] for entry in dict_list]
    all_masks = [entry["Mask"] for entry in dict_list]

    n_objects = len(all_masks)
    h, w = all_masks[0][0].shape
    # Create empty 3D array (time, height, width)
    combined_mask = np.zeros((n_frames, h, w), dtype=np.int32)

    for j, obj_masks in enumerate(all_masks):
        for t, mask in enumerate(obj_masks):
            set_mask = (mask > 0)
            combined_mask[t+start_indices[j]][set_mask] = j+1
            
    for j, obj_masks in enumerate(all_masks):
        for t, mask in enumerate(obj_masks):
            combined_mask[t+start_indices[j]][int(all_tracks[j][t][1]),int(all_tracks[j][t][0])] = j+1
            
    return combined_mask

def save_ctc_format(dict_list, tracked_mask, output_dir, digits=3):
    os.makedirs(output_dir, exist_ok=True)

    T = tracked_mask.shape[0]

    # Cast safely. CTC convention is uint16, but bump to uint32 if needed.
    max_label = int(tracked_mask.max()) if tracked_mask.size else 0
    if max_label <= np.iinfo(np.uint16).max:
        out_dtype = np.uint16
    else:
        out_dtype = np.uint32

    # Write per-frame label images
    for t in range(T):
        frame = tracked_mask[t].astype(out_dtype, copy=False)
        fname = os.path.join(output_dir, f"mask{t:0{digits}d}.tif")
        tifffile.imwrite(fname, frame, compression="zlib")

    # Build res_track.txt
    # For each track: L = ID+1, B = StartIndex, E = StartIndex + len(Path) - 1, P = 0
    # Sort by label so the file is tidy.
    rows = []
    for entry in dict_list:
        label = int(entry["ID"]) + 1
        begin = int(entry["StartIndex"])
        end = begin + len(entry["Path"]) - 1
        parent = 0  # no division tracking
        rows.append((label, begin, end, parent))

    rows.sort(key=lambda r: r[0])

    track_path = os.path.join(output_dir, "res_track.txt")
    with open(track_path, "w") as f:
        for label, begin, end, parent in rows:
            f.write(f"{label} {begin} {end} {parent}\n")

    return track_path
        
def _contiguous_runs(frames):
    """Given a sorted list of frame indices, yield (start, end) inclusive runs."""
    if not frames:
        return
    run_start = prev = frames[0]
    for t in frames[1:]:
        if t == prev + 1:
            prev = t
            continue
        yield (run_start, prev)
        run_start = prev = t
    yield (run_start, prev)


def _parent_old_label(entry):
    """Return the parent track's old label (ID + 1), or None if no parent.

    Accepts either 'Parent' (refactored tracker) or 'ParentID' (legacy)."""
    parent_id = entry.get("Parent", entry.get("ParentID"))
    if parent_id is None:
        return None
    return int(parent_id) + 1


def save_track_data_to_file(dict_list, tracked_mask, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    T = tracked_mask.shape[0]

    # Step 1: per tracker ID, find contiguous runs of frames where it is present.
    old_label_to_runs = {}
    for entry in dict_list:
        old_label = int(entry["ID"]) + 1
        begin = int(entry["StartIndex"])
        present = [t for t in range(begin, T) if np.any(tracked_mask[t] == old_label)]
        runs = list(_contiguous_runs(present))
        if runs:
            old_label_to_runs[old_label] = runs

    # Step 2: assign new CTC labels in (begin, old_label) order.
    flat = sorted(
        ((b, old_label, e)
         for old_label, runs in old_label_to_runs.items()
         for (b, e) in runs),
        key=lambda x: (x[0], x[1]),
    )
    new_assignments = {}    # (old_label, b, e) -> new_label
    end_index = {}          # (old_label, end_frame) -> new_label, for parent lookup
    for new_label, (b, old_label, e) in enumerate(flat, start=1):
        new_assignments[(old_label, b, e)] = new_label
        end_index[(old_label, e)] = new_label

    # CTC parent convention: a daughter's parent is the run whose final frame
    # is exactly (daughter_begin - 1). Otherwise 0.
    def resolve_parent_new_label(daughter_old_label, daughter_begin):
        entry = old_id_to_entry.get(daughter_old_label)
        if entry is None:
            return 0
        parent_old = _parent_old_label(entry)
        if parent_old is None:
            return 0
        return end_index.get((parent_old, daughter_begin - 1), 0)

    old_id_to_entry = {int(entry["ID"]) + 1: entry for entry in dict_list}

    # Step 3: rewrite mask with new labels. uint16 is required for CTC
    # (uint8 caps at 255 tracks).
    relabeled = np.zeros((T, *tracked_mask.shape[1:]), dtype=np.uint16)
    for (old_label, b, e), new_label in new_assignments.items():
        run = tracked_mask[b:e + 1]
        relabeled[b:e + 1][run == old_label] = new_label

    # Step 4: write res_track.txt (sorted by new label), with parent lineage.
    rows = sorted(
        (new_label, b, e, resolve_parent_new_label(old_label, b))
        for (old_label, b, e), new_label in new_assignments.items()
    )
    with open(os.path.join(output_folder, "res_track.txt"), "w") as f:
        for L, B, E, P in rows:
            f.write(f"{L} {B} {E} {P}\n")

    # Step 5: save per-frame TIFFs.
    for t in range(T):
        Image.fromarray(relabeled[t]).save(
            os.path.join(output_folder, f"mask{t:03d}.tif")
        )

    print(f"Saved track data to '{output_folder}'")