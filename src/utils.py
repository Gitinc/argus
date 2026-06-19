import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import algo_funcs as algo
import ctc_reformat as ctc
from ctc_metrics import evaluate_sequence

WORK_DIR = os.getcwd()
PATH_OF_SRC_MODUL = WORK_DIR.replace("notebooks", "src")

YOUR_DATA_PATH = r""
MASK_PATH = r""
OUTPUT_PATH = r""

IMAGE_CHANNEL = 1

sys.path.append(PATH_OF_SRC_MODUL)

def main():
    masks = algo.read_images(rf'{MASK_PATH}/man_track*.tif')
    images = algo.read_images(directory = rf'{YOUR_DATA_PATH}/t*.tif', is_gray_scale = True)
    edges_list = []
    labels_list = []
    for mask in masks:
        centers = []          # list of (y, x)
        region_labels = []    # parallel list of labels
        for region in skimage.measure.regionprops(np.array(mask, dtype=np.int16)):
            y, x = region.centroid
            cy = int(round(y))
            cx = int(round(x))
            centers.append((cy, cx))
            region_labels.append(int(region.label))
        edges_list.append(centers)
        labels_list.append(region_labels)

    dict_list, average_flow, tracked_mask = algo.track_points_optical_flow(images = np.array(images,dtype = np.uint8), all_points = edges_list,all_labels=labels_list, labeled_masks = masks, max_distance = 40, max_occlusion_frames=5)
        
    ctc.save_track_data_to_file(
                    dict_list,
                    tracked_mask,
                    OUTPUT_PATH
                )

    res = evaluate_sequence(
        OUTPUT_PATH,
        f"{YOUR_DATA_PATH}_GT"
    )

    print("DET: ", res["DET"])
    print("TRA: ", res["TRA"])
    print("SEG: ", res["SEG"])

if __name__ == "__main__":
    main()