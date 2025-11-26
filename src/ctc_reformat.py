import numpy as np
import os
import colorsys
from PIL import Image

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
        
def save_track_data_to_file(dict_list, n_frames, output_folder):
    #dict_list = [entry for entry in dict_list if len(entry["Path"])>1]
    
    os.makedirs(output_folder, exist_ok=True)
    start_indices = [entry["StartIndex"] for entry in dict_list]
    all_tracks = [entry["Path"] for entry in dict_list]
    all_ids = [entry["ID"] for entry in dict_list]

    combined_mask = generate_mask(dict_list, n_frames, output_folder)

    end_points = np.array(start_indices) + np.array([len(path) for path in all_tracks])

    # Open the file in write mode
    with open(os.path.join(output_folder,"res_track.txt"), 'w') as f:
        # Iterate through each track in the list
        for i in range(len(all_tracks)):
            # L: Unique label for the track (using a 1-based index)
            # Make sure it is a positive 16-bit value
            L = i+1
            
            # B: Temporal index of the frame in which the track begins
            B = start_indices[i]
            
            # E: Temporal index of the frame in which the track ends
            E = end_points[i]-1
            
            # P: Label of the parent track (always 0 in this case)
            P = 0

            # Write the formatted line to the file
            f.write(f"{L} {B} {E} {P}\n")
                

    for current_image_index, arr in enumerate(combined_mask):
        arr = arr.astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(os.path.join(output_folder, f"mask{current_image_index:03d}.tif"))
    
    print(f"Saved track data to '{output_folder}'")

    return combined_mask
