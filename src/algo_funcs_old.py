import glob
import numpy as np
import skimage
import matplotlib.pyplot as plt
import cv2
import itk
import os
import scipy
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from PIL import Image
from bm3d import bm3d, BM3DProfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd

IMAGE_CHANNEL = 0
NO_MATCH = np.array([-1, -1])

def freqgrid2(ysize: int, xsize: int):
    fy = np.fft.fftfreq(ysize)
    fx = np.fft.fftfreq(xsize)
    xGrid, yGrid = np.meshgrid(fx, fy, indexing='xy')
    return yGrid, xGrid

def create_monogenic_filters_log_gabor(ysize: int, xsize: int, wl: float, sigma_onf: float = 0.55):
    # Frequency grid
    yGrid, xGrid = freqgrid2(ysize, xsize)
    w = np.sqrt(yGrid**2 + xGrid**2)
    w[0, 0] = 1.0  # avoid log(0) at DC during filter construction

    # Log-Gabor radial filter
    fo = 1.0 / wl
    w0 = fo
    bpFilt = np.exp(- (np.log(w / w0) ** 2) / (2 * (np.log(sigma_onf) ** 2)))
    bpFilt[0, 0] = 0.0  # no DC response

    if ysize % 2 == 0:
        bpFilt[ysize // 2, :] = 0.0
    if xsize % 2 == 0:
        bpFilt[:, xsize // 2] = 0.0

    ReiszFilt = (1j * yGrid - xGrid) / w

   
    return {
        "bpFilt": bpFilt,
        "ReiszFilt": ReiszFilt,
        "wl": wl,
        "sigmaOnf": sigma_onf,
        "filtType": "lg",
    }


def monogenic_signal(image2d: np.ndarray, filtStruct: dict):
   
    F = np.fft.fft2(image2d)
    Ffilt = F * filtStruct["bpFilt"]

    # Even component
    m1 = np.real(np.fft.ifft2(Ffilt))

    # Odd components via complex Riesz transform
    Fmodd = np.fft.ifft2(Ffilt * filtStruct["ReiszFilt"])
    m2 = np.real(Fmodd)
    m3 = np.imag(Fmodd)
    return m1, m2, m3


def local_energy(m1: np.ndarray, m2: np.ndarray, m3: np.ndarray) -> np.ndarray:
   
    return m1**2 + m2**2 + m3**2


def gaussian_psf_2d(size: int, std: float) -> np.ndarray:
    size = max(3, int(round(size)))
    if size % 2 == 0:
        size += 1
    std = max(1e-6, float(std))
    g1 = scipy.signal.windows.gaussian(size, std=std, sym=True)
    psf = np.outer(g1, g1)
    psf /= psf.sum()
    return psf

def monogenic_filter_single_image(image_data,cw =60,sigma_onf = 0.05, sigma_smooth =8):    
    image_ch = image_data[..., IMAGE_CHANNEL]
    result = image_data.copy()

    Y, X = image_ch.shape

    filtStruct = create_monogenic_filters_log_gabor(Y, X, wl=cw, sigma_onf=sigma_onf)
    m1, m2, m3 = monogenic_signal(image_ch, filtStruct)
    LE = local_energy(m1, m2, m3)

    image_est = scipy.ndimage.gaussian_filter(LE, sigma=sigma_smooth)

    def rescale_to_uint8(x):
        x = np.asarray(x, dtype=np.float64)
        lo, hi = np.percentile(x, (0.5, 99.5))
        x = (x - lo) / max(hi - lo, 1e-12)
        x = np.clip(x, 0.0, 1.0)          # <-- critical: clip BEFORE casting
        return (x * 255.0).astype(np.uint8)
    
    result[..., IMAGE_CHANNEL] = rescale_to_uint8(image_est)
    return result

def monogenic_filter_images(images, n_workers=None, backend="thread"):
    n_workers = n_workers or os.cpu_count()

    if backend == "thread":
        Executor = ThreadPoolExecutor
    elif backend == "process":
        Executor = ProcessPoolExecutor
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    with Executor(max_workers=n_workers) as ex:
        return list(ex.map(monogenic_filter_single_image, images))

def unsharp_mask(image, sigma=1.0, alpha=1.5):
    """Applies unsharp masking by enhancing edges using a blurred image."""
    blurred = scipy.ndimage.gaussian_filter(image, sigma=sigma)
    unsharp = image + alpha * (image - blurred)  # High-boost filtering
    return unsharp

def process_single_image_mask(image, T, cw_monogenic):
    I = image[:,:,1].astype(np.float64)
    filled_image = monogenic_shape_detection_v2(I, T, cw_monogenic)
    return filled_image

def process_single_cell_detection(args):
    idx, image, mask, min_distance = args
    edges = locate_cells_monogenic_filter(image[:,:,1], mask, min_distance=min_distance)
    # Swap coordinates
    edges[:, [0, 1]] = edges[:, [1, 0]]
    return idx, edges

def locate_cells_monogenic_filter(image, thresh_img, min_distance=5):
    binary_mask_dilated = thresh_img > 0
    #binary_mask_dilated = scipy.ndimage.binary_erosion(binary_mask_dilated,iterations=1)
    labeled_mask, num_features = skimage.measure.label(binary_mask_dilated, return_num=True)
    
    # Remove small binary blobs
    min_size = 2  # Define the minimum size of blobs to keep
    labeled_mask = skimage.morphology.remove_small_objects(labeled_mask, min_size=min_size)
    
    all_maxima = []
    
    for region_id in range(1, num_features + 1):
        region_mask = labeled_mask == region_id
        
        props = skimage.measure.regionprops(region_mask.astype(int))
        if not props:
            continue
            
        bbox = props[0].bbox
        y_min, x_min, y_max, x_max = bbox
        
        region_image = image[y_min:y_max, x_min:x_max]
        region_binary = region_mask[y_min:y_max, x_min:x_max]
        
        masked_region = region_image * region_binary
        
        local_maxima = skimage.feature.peak_local_max(
            masked_region,
            min_distance=min_distance,
            num_peaks=2,  # Limit to max_per_mask peaks
            exclude_border=1
        )
        
        # Adjust coordinates to global image space
        if local_maxima.size > 0:
            local_maxima[:, 0] += y_min
            local_maxima[:, 1] += x_min
            all_maxima.append(local_maxima)
    
    # Combine all maxima into a single array
    if all_maxima:
        all_coords = np.vstack(all_maxima)
    else:
        all_coords = np.empty((0, 2), dtype=int)
    
    return all_coords

def monogenic_filter(images, cw_monogenic=np.array([50.0, 60.0, 70.0]), T=0.6, num_processes=None):
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        process_func = partial(process_single_image_mask, T=T, cw_monogenic=cw_monogenic)
        mask_seq = list(executor.map(process_func, images))
    
    return mask_seq

def locate_all_cells_monogenic_filter(images, cw_monogenic=np.array([50.0, 60.0, 70.0]), T=0.6, min_distance=5, num_processes=None):
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Step 1: Generate all masks in parallel
    all_mask = monogenic_filter(images, cw_monogenic, T, num_processes)
    
    # Step 2: Detect cells in parallel
    edges_list = [None] * len(images)
    
    # Create arguments for each cell detection task
    cell_detection_args = [(i, images[i], all_mask[i], min_distance) for i in range(len(images))]
    
    # Process cell detection in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = executor.map(process_single_cell_detection, cell_detection_args)
        
        # Collect results and maintain original order
        for idx, edges in results:
            edges_list[idx] = edges
    
    return edges_list, all_mask

def get_monogenic_filters(sizes, scales, filt_type='Cauchy'):
    aspr = sizes[0] / sizes[1]
    aspr = 1 
    
    ysize = sizes[0]
    xsize = sizes[1]
    
    # Compute the frequency grid
    ymid = ysize // 2
    xmid = xsize // 2
    
    ymax = ymid - 1 if ysize % 2 == 0 else ymid
    xmax = xmid - 1 if xsize % 2 == 0 else xmid
    
    # Create the frequency grid
    Y, X = np.meshgrid(np.arange(-ymid, ymax + 1), np.arange(-xmid, xmax + 1), indexing='ij')
    Y = np.fft.ifftshift(Y) / ysize
    X = np.fft.ifftshift(X) / (xsize * aspr)
    
    if filt_type == 'Poisson':
        ratio = 0.9
    
    # Radial frequency
    freq = np.sqrt(Y**2 + X**2)
    freq[0, 0] = 1  # Prevent division by zero at DC component
    
    # Band-pass filters
    num_scales = len(scales)
    bp_filter = np.zeros((ysize, xsize, num_scales))
    
    for s in range(num_scales):
        scale = scales[s]
        
        if filt_type == 'Poisson':
            s2 = scale / ((ratio - 1)) * np.log(ratio)  # difference of Poissons
            s1 = ratio * s2
            # potential indexing error
            bp_filter[:, :, s] = freq * (np.exp(-freq * s1) - np.exp(-freq * s2))
        elif filt_type == 'GaussianD':
            bp_filter[:, :, s] = freq**2 * np.exp(-freq**2 * scale)  # Gaussian derivative
        else:  # Default to Cauchy
            bp_filter[:, :, s] = freq * np.exp(-freq * scale)  # Cauchy
        
        bp_filter[0, 0, s] = 0
        
        # Remove high-frequency components for even dimensions
        if ysize % 2 == 0:
            bp_filter[ysize // 2, :, s] = 0
        
        if xsize % 2 == 0:
            bp_filter[:, xsize // 2, s] = 0
    
    # Normalize by maximum value of sum
    sum_filt = np.sum(bp_filter, axis=2)
    bp_filter = bp_filter / np.max(sum_filt)
    
    # Riesz filtering
    odd_filter = (1j * Y - X) / freq
    
    return bp_filter, odd_filter, X, Y

def monogenic_shape_detection_v2(im, T, cw_monogenic):
    # Get image dimensions
    nx, nz = im.shape
    
    # Get monogenic filters
    evenfilt, oddfilt, _, _ = get_monogenic_filters([nx, nz], cw_monogenic, 'Poisson')
    
    # Apply FFT to image
    F = np.fft.fft2(im)
    
    # Apply even filter
    Ffilt = np.zeros((nz, nx, len(cw_monogenic)), dtype=complex)
    for s in range(len(cw_monogenic)):
        Ffilt[:, :, s] = F * evenfilt[:, :, s]
    
    # Compute even component
    p = np.real(np.fft.ifft2(Ffilt, axes=(0, 1)))
    
    # should be bsxfun()
    # Apply odd filter and compute odd components
    Fmodd = np.zeros_like(Ffilt, dtype=complex)
    for s in range(len(cw_monogenic)):
        Fmodd[:, :, s] = np.fft.ifft2(Ffilt[:, :, s] * oddfilt, axes=(0, 1))
    
    q1 = np.real(Fmodd)
    q2 = np.imag(Fmodd)
    
    # Get dimensions
    ysize, xsize, ssize = p.shape
    
    # Small constant to avoid division by zero
    epsilon = 0.001
    
    # Compute symmetry function
    even = np.abs(p)
    odd = np.sqrt(q1 * q1 + q2 * q2)
    
    # Compute denominator with epsilon protection
    denominator = np.maximum(np.sqrt(even * even + q1 * q1 + q2 * q2), epsilon * np.ones((ysize, xsize, ssize)))
    
    # Compute numerator with thresholding
    SFS_numerator = np.maximum(even - odd - T, np.zeros((ysize, xsize, ssize)))
    
    # Compute final SFS
    SFS = (SFS_numerator / denominator) * np.sign(p)
    
    # Median across scales
    SFS = np.median(SFS, axis=2)
    
    # Apply threshold
    SFS[SFS < T] = 0

    return SFS

def locate_cells(image, min_size=10, expected_mask_size=15, min_distance = 10, manual_threshold=-1):
    # Threshold image using Otsu's method
    if manual_threshold == -1:
        thresh = skimage.filters.threshold_otsu(image)
    else:
        thresh = manual_threshold
    
    thresh_img = image > thresh
    
    # Remove small objects
    thresh_img = skimage.morphology.remove_small_objects(thresh_img, min_size=min_size)
    
    # Label connected components
    labeled_mask = skimage.measure.label(thresh_img)
    result_coords = []
    
    for region in skimage.measure.regionprops(labeled_mask):
        mask_size = region.area
        max_allowed = 1
        
        # Determine allowed local maxima
        if mask_size > expected_mask_size:
            max_allowed = 2
        elif mask_size > 2*expected_mask_size:
            max_allowed = 3
        elif mask_size > 3*expected_mask_size:
            max_allowed = 4
        else:
            max_allowed = 5
        
        # Extract region bounding box
        minr, minc, maxr, maxc = region.bbox
        region_image = image[minr:maxr, minc:maxc] * (labeled_mask[minr:maxr, minc:maxc] == region.label)
        
        # Find local maxima within the region
        region_maxima = skimage.feature.peak_local_max(region_image, min_distance=min_distance, exclude_border=False)
        
        # Adjust coordinates to the original image space
        region_maxima = [(y + minr, x + minc) for y, x in region_maxima]
        
        if len(region_maxima) > max_allowed:
            # Compute pairwise distances
            distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(region_maxima))
            selected = [region_maxima[0]]  # Start with the first maximum
            
            while len(selected) < max_allowed and len(region_maxima) > 1:
                farthest_point = max(region_maxima, key=lambda p: min([np.linalg.norm(np.array(p) - np.array(s)) for s in selected]))
                selected.append(farthest_point)
                region_maxima.remove(farthest_point)
            
            region_maxima = selected
        
        result_coords.extend(region_maxima)
        
    return np.array(result_coords), thresh_img

def image_registration(fixed_img, moving_img, registration_channel=-1, n_processes = None):
    if n_processes is None:
        n_processes = mp.cpu_count()

    transformed_img = np.zeros_like(moving_img, dtype=moving_img.dtype)

    if fixed_img.ndim == 3:
        num_channels = fixed_img.shape[2]
    else:
        num_channels = 1

    if moving_img.ndim == 3 and moving_img.shape[2] != num_channels:
        raise ValueError("Fixed and moving images must have the same number of channels.")
    elif moving_img.ndim == 2 and num_channels != 1:
        raise ValueError("Fixed and moving images must have the same number of channels.")

    if registration_channel == -1:
        gray_fixed = skimage.color.rgb2gray(fixed_img) if fixed_img.ndim == 3 else fixed_img
        gray_moving = skimage.color.rgb2gray(moving_img) if moving_img.ndim == 3 else moving_img

        fixed_image = itk.GetImageFromArray(gray_fixed.astype(np.float32))
        moving_image = itk.GetImageFromArray(gray_moving.astype(np.float32))
    elif 0 <= registration_channel < num_channels:
        fixed_image = itk.GetImageFromArray(fixed_img[:, :, registration_channel].astype(np.float32))
        moving_image = itk.GetImageFromArray(moving_img[:, :, registration_channel].astype(np.float32))
    else:
        raise ValueError(f"Invalid registration_channel: {registration_channel}. Must be -1 or between 0 and {num_channels - 1}.")

    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
    default_affine_parameter_map = parameter_object.GetDefaultParameterMap('affine')
    parameter_object.AddParameterMap(default_rigid_parameter_map)
    parameter_object.AddParameterMap(default_affine_parameter_map)

    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object,
        number_of_threads=n_processes,
        log_to_console=False)

    if moving_img.ndim == 3:
        for c in range(moving_img.shape[2]):
            moving_channel_img = itk.GetImageFromArray(moving_img[:, :, c].astype(np.float32))
            transformed_channel_img = itk.transformix_filter(moving_channel_img, result_transform_parameters)
            transformed_img[:, :, c] = itk.GetArrayFromImage(transformed_channel_img)
    else:
        transformed_channel_img = itk.transformix_filter(itk.GetImageFromArray(moving_img.astype(np.float32)), result_transform_parameters)
        transformed_img = itk.GetArrayFromImage(transformed_channel_img)

    transformed_img[transformed_img < 0] = 0

    return transformed_img

def read_images(directory, is_gray_scale = False):
    files = glob.glob(directory, recursive=False)
    
    sorted_files = sorted(
        files)
    
    images = []
    for img_path in sorted_files:
        if is_gray_scale:
            gray_img = np.array(Image.open(img_path),dtype=np.float64)
            
            img = np.stack((gray_img,) * 3, axis=-1)
        else:
            img = np.array(Image.open(img_path),dtype=np.float64)
        
        images.append(img)
    return images

def save_images(images, directory, pattern):
    os.makedirs(directory, exist_ok=True)
    for i in range(len(images)):
        cv2.imwrite(os.path.join(directory , pattern + f"{i:03d}" + ".tif"), np.array(images[i],dtype=np.uint8))

def detect_centers(image, cell_size, min_distance):
    thresh = skimage.filters.threshold_otsu(image)
    thresh_img = image > thresh

    thresh_img = skimage.morphology.remove_small_objects(thresh_img, max_size=cell_size)

    # Label connected components
    labeled_mask = skimage.measure.label(thresh_img)
    centers = []  # list of ((y, x), area, original_label)
    for region in skimage.measure.regionprops(labeled_mask):
        y, x = region.centroid
        centers.append(((int(round(y)), int(round(x))), region.area, region.label))

    # Decide which centers to keep
    if min_distance is None or min_distance <= 0:
        kept = [(c, lbl) for (c, _a, lbl) in centers]
    else:
        # Prefer larger regions: sort by area (desc)
        centers.sort(key=lambda t: t[1], reverse=True)
        # Greedy non-maximum suppression by distance
        kept = []
        kept_coords = []
        for (cy, cx), _area, lbl in centers:
            if not kept_coords:
                kept.append(((cy, cx), lbl))
                kept_coords.append((cy, cx))
                continue
            d = np.linalg.norm(np.array(kept_coords) - np.array([cy, cx]), axis=1)
            if np.all(d >= min_distance):
                kept.append(((cy, cx), lbl))
                kept_coords.append((cy, cx))

    # Build output mask: each kept component gets a new integer label (1, 2, ...)
    out_mask = np.zeros_like(labeled_mask, dtype=np.int32)
    kept_points = []
    centroid_label_pairs = []  # list of ((y, x), new_label)
    for new_label, ((cy, cx), orig_label) in enumerate(kept, start=1):
        out_mask[labeled_mask == orig_label] = new_label
        kept_points.append((cy, cx))
        centroid_label_pairs.append(((cy, cx), new_label))

    return np.array(kept_points, dtype=int), out_mask, centroid_label_pairs

def locate_all_cells_centroids(images, cell_size, min_distance):
    edges_list = []
    mask_list = []
    labels_list = []
    for image in images:
        edges, mask, centroid_label_pairs = detect_centers(
            image[:, :, IMAGE_CHANNEL],
            cell_size=cell_size,
            min_distance=min_distance
        )

        # Swap (y, x) -> (x, y); centroid_label_pairs is in the SAME order as edges,
        # so the labels stay aligned after the swap.
        edges_copy = edges.copy()
        edges[:, 0], edges[:, 1] = edges_copy[:, 1], edges_copy[:, 0]

        labels = np.array([lbl for (_pt, lbl) in centroid_label_pairs], dtype=np.int32)

        edges_list.append(edges)
        mask_list.append(mask)
        labels_list.append(labels)

    return edges_list, mask_list, labels_list

def locate_all_cells_local_maxima(images, min_size=10, expected_mask_size=15, min_distance = 10, manual_threshold = -1):
    edges_list = []
    mask_list = []
    for image in images:
        edges, mask = locate_cells(image[:,:,IMAGE_CHANNEL], min_size=min_size, expected_mask_size=expected_mask_size, min_distance = min_distance, manual_threshold = manual_threshold)
        edges_copy = edges.copy()  # Optional: create a copy to avoid modifying in-place
        edges[:, 0], edges[:, 1] = edges_copy[:, 1], edges_copy[:, 0]
        edges_list.append(edges)
        mask_list.append(mask)
    return edges_list, mask_list

def sliding_mean_subtraction(images, window_size=5):
    num_images = len(images)
    subtracted_images = []
    sliding_mean= []
    
    images_green = [img[:, :, 1] for img in images]

    for i in range(num_images):
        # dont include the image itself in the mean determination
        start_idx = max(0, i - window_size // 2)
        end_idx = min(num_images, i + window_size // 2 + 1)
        window = images_green[start_idx:i] + images_green[i+1:end_idx]
        mean_image = skimage.filters.gaussian(np.mean(window, axis=0),2)
        subtracted_image = np.array(images_green[i],dtype=np.float64) - np.array(mean_image,dtype=np.float64)

        subtracted_image[subtracted_image<(np.std(subtracted_image)+np.mean(subtracted_image))] = 0
        
        binary_mask = subtracted_image > 0

        
        binary_mask = scipy.ndimage.binary_erosion(binary_mask,iterations=2)                
        binary_mask = scipy.ndimage.binary_closing(binary_mask)       
        binary_mask = scipy.ndimage.binary_dilation(binary_mask,iterations=2)      
        
        subtracted_images.append(np.transpose([images[i][...,0],(subtracted_image*binary_mask),images[i][...,2]],(1,2,0)))
        sliding_mean.append(np.transpose([images[i][...,0],(mean_image),images[i][...,2]],(1,2,0)))

    return subtracted_images, sliding_mean

def batch_rescale_rgb_to_255(image_list):
    return [rescale_rgb_to_255(img) for img in image_list]
    
def rescale_rgb_to_255(img_array, channel = -1):
    img_array = img_array.astype(np.float32)
    img_rescaled = img_array.copy()

    if channel ==-1:
        channel_img = IMAGE_CHANNEL
    else:
        channel_img = channel
    # Extract and rescale the chosen channel
    channel_data = img_array[..., channel_img]     
    min_val = channel_data.min()
    max_val = channel_data.max()

    if max_val > min_val:
        channel_rescaled = (channel_data - min_val) / (max_val - min_val) * 255
    else:
        channel_rescaled = np.zeros_like(channel_data)

    # Put back the rescaled channel
    img_rescaled[..., channel_img] = channel_rescaled

    return img_rescaled.astype(np.uint8)

def average_direction(points):
    # Compute displacement vectors
    displacements = np.diff(points, axis=0)
    
    # Calculate average displacement vector
    avg_vector = np.mean(displacements, axis=0)
    
    # Normalize to get unit direction vector
    norm = np.linalg.norm(avg_vector)
    avg_direction = avg_vector / norm if norm != 0 else (0, 0)
    
    # Compute the direction angle in degrees
    angle_degrees = np.degrees(np.arctan2(avg_vector[1], avg_vector[0]))
    
    return avg_direction, angle_degrees, avg_vector

#def compute_optical_flow(prev_img, next_img):
#    return skimage.registration.optical_flow_tvl1(prev_img, next_img, attachment=5

def compute_optical_flow(prev_img, next_img):
    # OpenCV requires uint8 or float32
    if prev_img.dtype not in (np.uint8, np.float32):
        # Normalize to float32 in [0, 1] to preserve dynamic range
        prev_img = prev_img.astype(np.float32)
        next_img = next_img.astype(np.float32)
        # Joint min/max so both frames share the same scale
        lo = min(prev_img.min(), next_img.min())
        hi = max(prev_img.max(), next_img.max())
        if hi > lo:
            prev_img = (prev_img - lo) / (hi - lo)
            next_img = (next_img - lo) / (hi - lo)

    flow = cv2.calcOpticalFlowFarneback(
        prev_img, next_img, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    # (H, W, 2) with (dx, dy) → (2, H, W) with (dy, dx) to match skimage
    return flow.transpose(2, 0, 1)[::-1]

def find_current_mask(point, mask):
    x, y = map(int, point[::-1])
    h, w = mask.shape

    # Define window bounds
    x0 = max(x - 3, 0)
    x1 = min(x + 4, h)
    y0 = max(y - 3, 0)
    y1 = min(y + 4, w)

    # Extract local window
    window = mask[x0:x1, y0:y1]

    max_value = window.max()

    if max_value > 0:
        return mask == max_value
    else:
        curr_mask = np.zeros(mask.shape, dtype=bool)
        curr_mask[x, y] = True
        return curr_mask

def subtract_mean_channel(images, axis=0):
    images_array = np.array(images)
    mean_img = np.mean(images_array[..., IMAGE_CHANNEL], axis=axis)

    for i in range(len(images)):
        images[i][..., IMAGE_CHANNEL] = np.abs(images[i][..., IMAGE_CHANNEL] - mean_img)

    return images, mean_img

# ---------------------------------------------------------------------------
# Geometry / flow helpers
# ---------------------------------------------------------------------------

def _advect_points(points, flow_x, flow_y):
    """Move each point by the flow vector at its (clipped, integer) pixel."""
    h, w = flow_x.shape
    xi = np.clip(points[:, 0].astype(int), 0, w - 1)
    yi = np.clip(points[:, 1].astype(int), 0, h - 1)
    return points + np.stack([flow_x[yi, xi], flow_y[yi, xi]], axis=1)


def _threshold_small_flow(flow_x, flow_y, eps=0.05):
    """Zero out flow vectors with magnitude below `eps`."""
    small = np.hypot(flow_x, flow_y) < eps
    return np.where(small, 0.0, flow_x), np.where(small, 0.0, flow_y)


def _compute_flows_parallel(green_frames):
    pairs = list(zip(green_frames[:-1], green_frames[1:]))
    with mp.Pool(mp.cpu_count()) as pool:
        return pool.starmap(compute_optical_flow, pairs)


# ---------------------------------------------------------------------------
# Track entries
# ---------------------------------------------------------------------------

def _new_track(track_id, frame_idx, point, label, parent_id=None):
    """Create a fresh track entry."""
    return {
        "ID": track_id,
        "StartIndex": frame_idx,
        "Path": [point],
        "MaskLabels": [label],
        "AverageDir": np.zeros(2),
        "Active": True,
        "Parent": parent_id,
        "Children": [],
        "OcclusionCount": 0,
        "LastSeenIndex": frame_idx,
        "GhostPos": np.asarray(point, dtype=np.float32),
    }


def _current_position(entry):
    """Where a track 'is' right now — last detection, or ghost if occluded."""
    return entry["GhostPos"] if entry["OcclusionCount"] > 0 else entry["Path"][-1]


def _paint(tracked_mask, frame_idx, labeled_mask, label, track_id):
    """Paint pixels of `label` in `frame_idx` with `track_id + 1` (IDs stay nonzero)."""
    if label is None or labeled_mask is None:
        return
    tracked_mask[frame_idx][labeled_mask == label] = track_id + 1


def _spawn_track(dict_list, tracked_mask, labeled_mask, frame_idx,
                 point, label, parent_id=None):
    """Append a new track and paint its mask. Returns the new track ID."""
    new_id = len(dict_list)
    dict_list.append(_new_track(new_id, frame_idx, point, label, parent_id))
    _paint(tracked_mask, frame_idx, labeled_mask, label, new_id)
    return new_id


def _extend_track(entry, point, label, frame_idx, tracked_mask, labeled_mask):
    """Append a real detection to a track."""
    entry["Path"].append(point)
    entry["MaskLabels"].append(label)
    entry["OcclusionCount"] = 0
    entry["LastSeenIndex"] = frame_idx
    entry["GhostPos"] = np.asarray(point, dtype=np.float32)
    _paint(tracked_mask, frame_idx, labeled_mask, label, entry["ID"])
    if len(entry["Path"]) > 1:
        _, _, entry["AverageDir"] = average_direction(np.array(entry["Path"]))


def _mark_occluded(entry, flow_x, flow_y, max_occlusion_frames):
    """No detection this frame: advect ghost, retire if past occlusion budget."""
    entry["GhostPos"] = _advect_points(entry["GhostPos"].reshape(1, 2),
                                       flow_x, flow_y)[0]
    entry["OcclusionCount"] += 1
    if entry["OcclusionCount"] > max_occlusion_frames:
        entry["Active"] = False


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _is_no_match(match):
    return np.array_equal(match, NO_MATCH)


def _round_key(point):
    """Hashable key for a 2D point, robust to float jitter."""
    return (round(float(point[0]), 6), round(float(point[1]), 6))


def _build_label_lookup(points, labels):
    """Map each detected point to its mask label."""
    return {_round_key(p): int(l) for p, l in zip(points, labels)}


def _find_collisions(matches, active_indices, label_of):
    """
    Return entry indices whose matched detections collided in the same mask
    label (multiple tracks predicted into one cell).
    """
    label_to_tracks = {}
    for k, entry_idx in enumerate(active_indices):
        if _is_no_match(matches[k]):
            continue
        label = label_of(matches[k])
        if label is None:
            continue
        label_to_tracks.setdefault(label, []).append(entry_idx)

    return {idx for tracks in label_to_tracks.values()
                if len(tracks) > 1
                for idx in tracks}


def _find_split_partner(prediction, matched_label, unmatched_points,
                        consumed, label_of, max_distance):
    """
    Find the nearest unmatched detection that is within `max_distance` of
    `prediction` and lies in a *different* mask than `matched_label`.
    Returns (unmatched_index, point) or None.
    """
    best = None  # (distance, ui, point)
    for ui, point in enumerate(unmatched_points):
        if ui in consumed or label_of(point) == matched_label:
            continue
        dist = float(np.hypot(point[0] - prediction[0], point[1] - prediction[1]))
        if dist > max_distance:
            continue
        if best is None or dist < best[0]:
            best = (dist, ui, point)

    if best is None:
        return None
    _, ui, point = best
    return ui, point


def _split_parent_into_daughters(parent, daughter_a, daughter_b, frame_idx,
                                 dict_list, tracked_mask, labeled_mask):
    """
    Roll back the parent's tentative extension into `frame_idx` and replace
    it with two new daughter tracks (mitosis-like event).
    """
    parent["Path"].pop()
    parent["MaskLabels"].pop()
    parent["Active"] = False
    tracked_mask[frame_idx][tracked_mask[frame_idx] == parent["ID"] + 1] = 0

    a_point, a_label = daughter_a
    b_point, b_label = daughter_b
    a_id = _spawn_track(dict_list, tracked_mask, labeled_mask, frame_idx,
                        a_point, a_label, parent_id=parent["ID"])
    b_id = _spawn_track(dict_list, tracked_mask, labeled_mask, frame_idx,
                        b_point, b_label, parent_id=parent["ID"])
    parent["Children"] = [a_id, b_id]


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

def track_points_optical_flow(images, all_points, all_labels, labeled_masks,
                              max_distance, max_occlusion_frames=5):
    # --- Setup ---
    green = [img[:, :, IMAGE_CHANNEL] if img.ndim == 3 else img for img in images]
    T, H, W = len(labeled_masks), *labeled_masks[0].shape
    tracked_mask = np.zeros((T, H, W), dtype=np.int32)
    average_flow = np.zeros((H, W), dtype=np.float32)
    dict_list = []

    # Seed tracks from frame 0.
    for point, label in zip(all_points[0], all_labels[0]):
        _spawn_track(dict_list, tracked_mask, labeled_masks[0],
                     frame_idx=0, point=point, label=int(label))

    flows = _compute_flows_parallel(green)

    # --- Main loop: propagate tracks frame by frame ---
    for t, (flow_y, flow_x) in enumerate(flows):
        t_next = t + 1
        print(f"Working on image: {t}")

        flow_x, flow_y = _threshold_small_flow(flow_x, flow_y)
        average_flow += np.hypot(flow_x, flow_y)

        active_indices = [i for i, e in enumerate(dict_list) if e["Active"]]
        if not active_indices:
            continue

        # 1. Predict where each active track lands in t_next.
        current_points = np.array([_current_position(dict_list[i])
                                   for i in active_indices])
        predicted = _advect_points(current_points, flow_x, flow_y)

        # 2. Match predictions to detections.
        detected = all_points[t_next]
        matches, unmatched_points = hungarian_assignment(
            predicted, detected, max_distance=max_distance,
        )
        label_lookup = _build_label_lookup(detected, all_labels[t_next])
        label_of = lambda point: label_lookup.get(_round_key(point))

        # 3. Find tracks that landed in the same mask (merge / ambiguity).
        collided = _find_collisions(matches, active_indices, label_of)

        # 4. Extend matched tracks; occlude unmatched / collided ones.
        for k, entry_idx in enumerate(active_indices):
            entry = dict_list[entry_idx]
            match = matches[k]
            if _is_no_match(match) or entry_idx in collided:
                _mark_occluded(entry, flow_x, flow_y, max_occlusion_frames)
            else:
                _extend_track(entry, match, label_of(match), t_next,
                              tracked_mask, labeled_masks[t_next])

        # 5. Detect splits: parent has a match, but another unmatched detection
        #    lies nearby in a *different* mask -> mitosis-like event.
        consumed_unmatched = set()
        for k, entry_idx in enumerate(active_indices):
            if entry_idx in collided or _is_no_match(matches[k]):
                continue

            parent = dict_list[entry_idx]
            matched_label = label_of(matches[k])
            partner = _find_split_partner(
                prediction=predicted[k],
                matched_label=matched_label,
                unmatched_points=unmatched_points,
                consumed=consumed_unmatched,
                label_of=label_of,
                max_distance=max_distance,
            )
            if partner is None:
                continue

            ui, partner_point = partner
            consumed_unmatched.add(ui)
            _split_parent_into_daughters(
                parent=parent,
                daughter_a=(matches[k], matched_label),
                daughter_b=(partner_point, label_of(partner_point)),
                frame_idx=t_next,
                dict_list=dict_list,
                tracked_mask=tracked_mask,
                labeled_mask=labeled_masks[t_next],
            )

        # 6. Spawn new tracks from any leftover unmatched detections.
        for ui, point in enumerate(unmatched_points):
            if ui in consumed_unmatched:
                continue
            _spawn_track(dict_list, tracked_mask, labeled_masks[t_next],
                         frame_idx=t_next, point=point, label=label_of(point))

    return dict_list, average_flow, tracked_mask

def track_points_nearest_neighbor(
    images, all_points, all_labels, labeled_masks, max_distance,
    mitosis_distance=None,        # max distance between two daughter candidates (defaults to max_distance)
    mitosis_area_ratio=0.6,       # each daughter should be >= this * parent area (tunable)
    mitosis_pair_area_ratio=1.5,  # combined daughter area <= this * parent area (tunable)
):
    """
    Nearest-neighbor tracker with a mitosis/split heuristic.

    Mitosis is declared at frame t_next for a parent track when:
      1. The parent matched some detection A in t_next (Hungarian result), AND
      2. There exists an unmatched detection B in t_next within `mitosis_distance`
         of the parent's last position, AND
      3. A and B are themselves within `mitosis_distance` of each other, AND
      4. The combined mask area of A and B is consistent with the parent's
         previous area (controlled by mitosis_area_ratio / mitosis_pair_area_ratio).

    On a confirmed split, the parent is deactivated and two fresh daughter tracks
    are spawned at t_next. The parent's ID is recorded on each daughter for
    lineage bookkeeping.
    """
    if mitosis_distance is None:
        mitosis_distance = max_distance

    initial_points = all_points[0]
    initial_labels = all_labels[0]

    T = len(labeled_masks)
    H, W = labeled_masks[0].shape

    tracked_mask = np.zeros((T, H, W), dtype=np.int32)

    dict_list = []
    for i in range(initial_points.shape[0]):
        item = {
            "Path": [initial_points[i]],
            "AverageDir": [0, 0],
            "StartIndex": 0,
            "Active": True,
            "ID": i,
            "MaskLabels": [int(initial_labels[i])],
            "ParentID": None,          # set on daughters
            "EndReason": None,         # "mitosis", "lost", or None while active
            "DaughterIDs": [],         # set on the parent at division time
        }
        dict_list.append(item)
        tracked_mask[0][labeled_masks[0] == initial_labels[i]] = item["ID"] + 1

    current_points = initial_points

    for t_prev in range(len(images) - 1):
        t_next = t_prev + 1
        print("Working on image: ", t_prev)

        calculated_edges = all_points[t_next]
        calculated_labels = all_labels[t_next]

        matches, unmatched_points = hungarian_assignment(
            current_points, calculated_edges, max_distance=max_distance
        )

        coord_to_label_next = {
            (float(p[0]), float(p[1])): int(l)
            for p, l in zip(calculated_edges, calculated_labels)
        }

        # Track which unmatched detections actually get consumed by mitosis
        # so we don't also spawn a brand-new singleton track from them below.
        unmatched_list = [tuple(map(float, p)) for p in unmatched_points]
        consumed_unmatched = set()

        # --- Pass 1: handle Hungarian matches, with mitosis check on each ---
        for i_dict_list, entry in enumerate(dict_list):
            if not entry["Active"]:
                continue

            if np.array_equal(matches[i_dict_list], [-1, -1]):
                continue  # handled below as "lost"

            matched_point = matches[i_dict_list]
            lbl_next = coord_to_label_next.get(
                (float(matched_point[0]), float(matched_point[1]))
            )

            # ---- Mitosis check -------------------------------------------------
            split_info = _detect_mitosis_split(
                parent_entry=entry,
                t_prev=t_prev,
                t_next=t_next,
                matched_point=matched_point,
                matched_label=lbl_next,
                unmatched_list=unmatched_list,
                consumed_unmatched=consumed_unmatched,
                coord_to_label_next=coord_to_label_next,
                labeled_masks=labeled_masks,
                mitosis_distance=mitosis_distance,
                mitosis_area_ratio=mitosis_area_ratio,
                mitosis_pair_area_ratio=mitosis_pair_area_ratio,
            )

            if split_info is not None:
                d1_pt, d1_lbl, d2_pt, d2_lbl = split_info

                # Close out the parent at t_prev (do NOT extend its path to t_next).
                entry["Active"] = False
                entry["EndReason"] = "mitosis"

                # Spawn two daughters at t_next.
                for d_pt, d_lbl in ((d1_pt, d1_lbl), (d2_pt, d2_lbl)):
                    new_id = len(dict_list)
                    dict_list.append({
                        "Path": [np.array(d_pt)],
                        "AverageDir": [0, 0],
                        "StartIndex": t_next,
                        "Active": True,
                        "ID": new_id,
                        "MaskLabels": [d_lbl],
                        "ParentID": entry["ID"],
                        "EndReason": None,
                        "DaughterIDs": [],
                    })
                    entry["DaughterIDs"].append(new_id)
                    if d_lbl is not None:
                        tracked_mask[t_next][labeled_masks[t_next] == d_lbl] = new_id + 1

                # Mark both detections as consumed so the unmatched-spawn pass skips them.
                consumed_unmatched.add(tuple(map(float, d1_pt)))
                consumed_unmatched.add(tuple(map(float, d2_pt)))
                continue
            # -------------------------------------------------------------------

            # Normal (non-mitotic) continuation.
            entry["Path"].append(matched_point)
            entry["MaskLabels"].append(lbl_next)
            if lbl_next is not None:
                tracked_mask[t_next][labeled_masks[t_next] == lbl_next] = entry["ID"] + 1
            if len(entry["Path"]) > 1:
                _, _, avg_dir = average_direction(np.array(entry["Path"]))
                entry["AverageDir"] = avg_dir

        # --- Pass 2: spawn brand-new tracks for genuinely unmatched detections ---
        for point in unmatched_points:
            key = tuple(map(float, point))
            if key in consumed_unmatched:
                continue
            lbl_next = coord_to_label_next.get(key)
            new_id = len(dict_list)
            dict_list.append({
                "Path": [point],
                "AverageDir": [0, 0],
                "StartIndex": t_next,
                "Active": True,
                "ID": new_id,
                "MaskLabels": [lbl_next],
                "ParentID": None,
                "EndReason": None,
                "DaughterIDs": [],
            })
            if lbl_next is not None:
                tracked_mask[t_next][labeled_masks[t_next] == lbl_next] = new_id + 1

        # Rebuild current_points aligned with dict_list order.
        current_points = np.array([entry["Path"][-1] for entry in dict_list])

    return dict_list, tracked_mask

def _detect_mitosis_split(
    parent_entry,
    t_prev,
    t_next,
    matched_point,
    matched_label,
    unmatched_list,
    consumed_unmatched,
    coord_to_label_next,
    labeled_masks,
    mitosis_distance,
    mitosis_area_ratio,
    mitosis_pair_area_ratio,
):
    """
    Decide whether parent_entry undergoes mitosis at t_next.

    Returns (daughter1_pt, daughter1_lbl, daughter2_pt, daughter2_lbl) on a
    confirmed split, otherwise None.
    """
    if matched_label is None:
        return None

    parent_last_pt = np.asarray(parent_entry["Path"][-1], dtype=float)
    matched_pt = np.asarray(matched_point, dtype=float)

    # Parent's mask area from the previous frame (where we know its label).
    parent_label_prev = parent_entry["MaskLabels"][-1]
    if parent_label_prev is None:
        return None
    parent_area = int(np.sum(labeled_masks[t_prev] == parent_label_prev))
    if parent_area <= 0:
        return None

    matched_area = int(np.sum(labeled_masks[t_next] == matched_label))

    # Find the best second-daughter candidate among unmatched detections in t_next.
    best = None  # (score, pt_tuple, label, area)
    for cand_key in unmatched_list:
        if cand_key in consumed_unmatched:
            continue
        cand_pt = np.asarray(cand_key, dtype=float)

        # Both daughters should be near the parent's last position...
        if np.linalg.norm(cand_pt - parent_last_pt) > mitosis_distance:
            continue
        # ...and near each other (daughters are typically adjacent post-division).
        if np.linalg.norm(cand_pt - matched_pt) > mitosis_distance:
            continue

        cand_label = coord_to_label_next.get(cand_key)
        if cand_label is None:
            continue
        cand_area = int(np.sum(labeled_masks[t_next] == cand_label))
        if cand_area <= 0:
            continue

        # Area sanity: each daughter should be a reasonable fraction of the parent,
        # and combined area shouldn't blow up.
        if matched_area < mitosis_area_ratio * parent_area:
            # matched detection alone too small relative to parent → still ok
            pass
        if cand_area < mitosis_area_ratio * 0.5 * parent_area:
            # very small fragment — likely noise, skip
            continue
        if (matched_area + cand_area) > mitosis_pair_area_ratio * parent_area:
            continue
        if (matched_area + cand_area) < mitosis_area_ratio * parent_area:
            # combined too small to plausibly be the parent splitting
            continue

        # Score: prefer pairs whose midpoint is closest to parent_last_pt and
        # whose areas are balanced.
        midpoint = 0.5 * (cand_pt + matched_pt)
        center_err = np.linalg.norm(midpoint - parent_last_pt)
        balance_err = abs(matched_area - cand_area) / float(matched_area + cand_area)
        score = center_err + 10.0 * balance_err  # weight balance modestly

        if best is None or score < best[0]:
            best = (score, cand_key, cand_label, cand_area)

    if best is None:
        return None

    _, cand_key, cand_label, _ = best
    return (matched_pt, matched_label, np.array(cand_key), cand_label)

def find_closest_points(points_A, points_B, max_distance=10):
    distances = np.linalg.norm(points_A[:, None, :] - points_B[None, :, :], axis=2)
    # Find the index of the closest point in points_B for each point in points_A
    closest_indices = np.argmin(distances, axis=1)

    # Use the indices to extract the closest points from points_B
    closest_points = points_B[closest_indices]
    # Filter out points that are farther than max_distance
    valid_mask = distances[np.arange(len(points_A)), closest_indices] <= max_distance
    closest_points = np.where(valid_mask[:, None], closest_points, [-1, -1])

    # Find points in points_A that are matched to the same closest point in points_B
    matched_to_same = {}
    for i, idx in enumerate(closest_indices):
        if valid_mask[i]:
            if idx not in matched_to_same:
                matched_to_same[idx] = []
            matched_to_same[idx].append(points_A[i])

    # Convert matched_to_same to a list of arrays
    matched_to_same = {k: np.array(v) for k, v in matched_to_same.items()}

    unmatched_points_B = np.array([point for i, point in enumerate(points_B) if i not in closest_indices])
    return closest_points, unmatched_points_B, matched_to_same

def hungarian_assignment(flow_points, edge_points, max_distance):
    if len(flow_points) == 0:
        return np.array([]), edge_points, {}
    
    if len(edge_points) == 0:
        closest_points = np.full((len(flow_points), 2), [-1, -1])
        return closest_points, np.array([]), {}
    
    # Create cost matrix (distances between all flow points and edge points)
    cost_matrix = np.zeros((len(flow_points), len(edge_points)))
    
    for i, flow_point in enumerate(flow_points):
        for j, edge_point in enumerate(edge_points):
            distance = np.linalg.norm(flow_point - edge_point)
            # Use distance as cost, but set very high cost for distances > max_distance
            cost_matrix[i, j] = distance if distance <= max_distance else 1e6
    
    # Solve assignment problem using Hungarian algorithm
    flow_indices, edge_indices = scipy.optimize.linear_sum_assignment(cost_matrix)
    
    # Initialize closest_points with [-1, -1] for all flow points
    closest_points = np.full((len(flow_points), 2), [-1, -1], dtype=float)
    matched_edge_indices = set()
    
    # Fill in the valid assignments
    for flow_idx, edge_idx in zip(flow_indices, edge_indices):
        if cost_matrix[flow_idx, edge_idx] <= max_distance:
            closest_points[flow_idx] = edge_points[edge_idx]
            matched_edge_indices.add(edge_idx)
    
    # Find unmatched edge points
    unmatched_points_B = np.array([
        edge_points[i] for i in range(len(edge_points)) 
        if i not in matched_edge_indices
    ])
    
    return closest_points, unmatched_points_B

def unsharp_single_image(image_data, radius, amount):
    i, image = image_data
    
    # Compute the unsharp mask on the green channel
    unsharp_masked = unsharp_mask(image[..., IMAGE_CHANNEL], sigma=radius, alpha=amount)
    
    # Threshold the unsharp masked image
    unsharp_masked[unsharp_masked < np.std(unsharp_masked)] = 0
    binary_mask = unsharp_masked > 0
    
    # Apply morphological operations
    binary_mask = scipy.ndimage.binary_closing(binary_mask, iterations=1)
    binary_mask = skimage.morphology.remove_small_objects(binary_mask, min_size=6)

    # Combine image back together
    result = image.copy()
    result[..., IMAGE_CHANNEL] = unsharp_masked * binary_mask

    return result
    
def median_denoise_single_image(image_data, size):
    i, image = image_data
    
    # Compute the unsharp mask on the green channel
    image_wv = scipy.ndimage.median_filter(image[..., IMAGE_CHANNEL], size=size)
    
    result = image.copy()
    result[..., IMAGE_CHANNEL] = image_wv

    return result

def log_transform_single_image(image_data):
    i, image = image_data
    
    c = 255 / np.log(1 + np.max(image[..., IMAGE_CHANNEL]))
    log_image = c * (np.log(image[..., IMAGE_CHANNEL] + 1))
    
    result = image.copy()
    result[..., IMAGE_CHANNEL] = log_image

    return result

def bm3d_denoise_single_image(image_data):
    noise_std = np.std(image_data)
    
    # Compute the unsharp mask on the green channel
    image_est = bm3d(image_data[..., IMAGE_CHANNEL], noise_std);
    
    result = image_data.copy()
    result[..., IMAGE_CHANNEL] = image_est

    return result

def wavelet_denoise_single_image(image_data):
    # Compute the unsharp mask on the green channel
    image_est = skimage.restoration.denoise_wavelet(image_data[..., IMAGE_CHANNEL]);
    
    result = image_data.copy()
    result[..., IMAGE_CHANNEL] = np.array(image_est*255,dtype = np.uint8)

    return result

def unsharp_mask_all(images, radius = 8, amount=20, n_processes=None):
    # If n_processes is not specified, use all available CPU cores
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    # Create a pool of workers
    with mp.Pool(processes=n_processes) as pool:
        # Create a list of tuples (index, image) to pass to the worker function
        image_data = [(i, img) for i, img in enumerate(images)]

        # Use partial to create a function with fixed radius parameter
        process_func = partial(unsharp_single_image, radius=radius, amount=amount)
        
        # Process images in parallel and gather results
        results = pool.map(process_func, image_data)
    
    return results

def log_transform_all(images, n_processes=None):
    # If n_processes is not specified, use all available CPU cores
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    # Create a pool of workers
    with mp.Pool(processes=n_processes) as pool:
        # Create a list of tuples (index, image) to pass to the worker function
        image_data = [(i, img) for i, img in enumerate(images)]

        # Use partial to create a function with fixed radius parameter
        process_func = partial(log_transform_single_image)
        
        # Process images in parallel and gather results
        results = pool.map(process_func, image_data)
    
    return results

def median_denoise_all(images, size, n_processes=None):
    # If n_processes is not specified, use all available CPU cores
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    # Create a pool of workers
    with mp.Pool(processes=n_processes) as pool:
        # Create a list of tuples (index, image) to pass to the worker function
        image_data = [(i, img) for i, img in enumerate(images)]

        # Use partial to create a function with fixed radius parameter
        process_func = partial(median_denoise_single_image, size = size)
        
        # Process images in parallel and gather results
        results = pool.map(process_func, image_data)
    
    return results

def bm3d_denoise_all(images, max_workers=None):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        denoised_images = list(executor.map(bm3d_denoise_single_image, images))
    
    return denoised_images

def clip_images(images, clip_percentile=95):
    for i in range(len(images)):
        img = images[i][..., IMAGE_CHANNEL].astype(np.float32)

        thresh = np.percentile(img, clip_percentile)   # clip values above this percentile
        img[img > thresh] = thresh

        img = (img - img.min()) / (img.max() - img.min()) * 255
        images[i][..., IMAGE_CHANNEL] = img.astype(np.uint8)

    return images

def wavelet_denoise_images_all(images):
    """Apply wavelet denoising to a list of images."""
    return [wavelet_denoise_single_image(img) for img in images]

def _track_frames(entry):
    """

    """
    if "Frames" in entry and len(entry["Frames"]) == len(entry.get("Path", [])):
        return [int(f) for f in entry["Frames"]]

    start = int(entry.get("StartIndex", 0))
    return list(range(start, start + len(entry.get("Path", []))))

def _advect_point_over_gap(point, flows, start_frame, end_frame):
    """
    Move a point from `start_frame` to `end_frame` using the existing TV-L1
    optical-flow fields. `flows[t]` maps frame t -> frame t+1.
    """
    p = np.asarray(point, dtype=np.float32).reshape(1, 2)

    for t in range(int(start_frame), int(end_frame)):
        if t < 0 or t >= len(flows):
            break
        flow_y, flow_x = flows[t]
        flow_x, flow_y = _threshold_small_flow(flow_x, flow_y)
        p = _advect_points(p, flow_x, flow_y)

    return p[0]

def _direction_cost_for_refinement(track_a, track_b):
    """

    """
    path_a = track_a.get("Path", [])
    path_b = track_b.get("Path", [])

    if len(path_a) < 2 or len(path_b) < 2:
        return 0.0

    va = np.asarray(path_a[-1], dtype=np.float32) - np.asarray(path_a[-2], dtype=np.float32)
    vb = np.asarray(path_b[1], dtype=np.float32) - np.asarray(path_b[0], dtype=np.float32)

    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-8 or nb < 1e-8:
        return 0.0

    va = va / na
    vb = vb / nb
    return float(1.0 - np.dot(va, vb))

def _paint_refined_tracks(dict_list, labeled_masks):
    """Recreate the tracked mask after refinement."""
    if len(labeled_masks) == 0:
        return np.zeros((0, 0, 0), dtype=np.int32)

    T = len(labeled_masks)
    H, W = labeled_masks[0].shape
    tracked_mask = np.zeros((T, H, W), dtype=np.int32)

    for tr in dict_list:
        track_id = int(tr["ID"])
        frames = _track_frames(tr)
        labels = tr.get("MaskLabels", [])

        for f, label in zip(frames, labels):
            if f < 0 or f >= T:
                continue
            if label is None or int(label) < 0:
                continue
            tracked_mask[f][labeled_masks[f] == int(label)] = track_id + 1

    return tracked_mask

def refine_tracks_by_gap_closing(
    dict_list,
    images,
    labeled_masks,
    max_distance,
    max_gap=5,
    gap_penalty=0.25,
    direction_weight=0.25,
    no_link_cost=2.5,
):
    if len(dict_list) == 0:
        if len(labeled_masks) == 0:
            return [], np.zeros((0, 0, 0), dtype=np.int32), {"accepted_links": []}
        T = len(labeled_masks)
        H, W = labeled_masks[0].shape
        return [], np.zeros((T, H, W), dtype=np.int32), {"accepted_links": []}

    green = [img[:, :, IMAGE_CHANNEL] if getattr(img, "ndim", 2) == 3 else img for img in images]
    flows = _compute_flows_parallel(green) if len(green) > 1 else []

    n = len(dict_list)
    BIG = 1e6
    cost_matrix = np.full((n, n), BIG, dtype=np.float32)
    candidate_info = {}

    for i, track_i in enumerate(dict_list):
        frames_i = _track_frames(track_i)
        if len(frames_i) == 0 or len(track_i.get("Path", [])) == 0:
            continue

        end_frame = int(frames_i[-1])
        end_point = np.asarray(track_i["Path"][-1], dtype=np.float32)

        for j, track_j in enumerate(dict_list):
            if i == j:
                continue

            frames_j = _track_frames(track_j)
            if len(frames_j) == 0 or len(track_j.get("Path", [])) == 0:
                continue

            start_frame = int(frames_j[0])
            start_point = np.asarray(track_j["Path"][0], dtype=np.float32)

            temporal_gap = start_frame - end_frame - 1
            if temporal_gap < 1 or temporal_gap > max_gap:
                continue

            predicted = _advect_point_over_gap(
                point=end_point,
                flows=flows,
                start_frame=end_frame,
                end_frame=start_frame,
            )

            spatial_distance = float(np.linalg.norm(predicted - start_point))
            allowed_distance = float(max_distance * np.sqrt(temporal_gap + 1.0))
            if spatial_distance > allowed_distance:
                continue

            spatial_cost = spatial_distance / max(allowed_distance, 1e-8)
            temporal_cost = float(gap_penalty * temporal_gap)
            dir_cost = float(direction_weight * _direction_cost_for_refinement(track_i, track_j))
            total_cost = float(spatial_cost + temporal_cost + dir_cost)

            if total_cost >= no_link_cost:
                continue

            cost_matrix[i, j] = total_cost
            candidate_info[(i, j)] = {
                "end_frame": end_frame,
                "start_frame": start_frame,
                "gap": int(temporal_gap),
                "spatial_distance": spatial_distance,
                "allowed_distance": allowed_distance,
                "cost": total_cost,
            }

    rows, cols = scipy.optimize.linear_sum_assignment(cost_matrix)

    successor = {}
    predecessor = {}
    accepted_links = []

    for i, j in zip(rows, cols):
        if cost_matrix[i, j] >= no_link_cost:
            continue
        if i in successor or j in predecessor:
            continue
        successor[int(i)] = int(j)
        predecessor[int(j)] = int(i)
        accepted_links.append({
            "from_track": int(i),
            "to_track": int(j),
            **candidate_info.get((int(i), int(j)), {}),
        })

    roots = [i for i in range(n) if i not in predecessor]
    roots = sorted(roots, key=lambda idx: (_track_frames(dict_list[idx])[0] if _track_frames(dict_list[idx]) else 0, idx))

    refined_tracks = []
    old_to_new = {}

    for root in roots:
        chain = []
        current = int(root)
        visited = set()

        while current is not None and current not in visited:
            visited.add(current)
            chain.append(current)
            current = successor.get(current)

        frames = []
        path = []
        labels = []

        for old_id in chain:
            tr = dict_list[old_id]
            tr_frames = _track_frames(tr)
            frames.extend(tr_frames)
            path.extend([np.asarray(p, dtype=np.float32) for p in tr.get("Path", [])])
            labels.extend([int(l) if l is not None else None for l in tr.get("MaskLabels", [])])
            old_to_new[old_id] = len(refined_tracks)

        if len(path) == 0:
            continue

        # Sort by frame in case a future change creates non-monotone fragments.
        order = np.argsort(np.asarray(frames, dtype=np.int32))
        frames = [int(frames[k]) for k in order]
        path = [path[k] for k in order]
        labels = [labels[k] for k in order]

        new_id = len(refined_tracks)
        refined_tracks.append({
            "ID": new_id,
            "StartIndex": int(frames[0]),
            "EndIndex": int(frames[-1]),
            "Frames": frames,
            "Path": path,
            "MaskLabels": labels,
            "AverageDir": average_direction(np.asarray(path))[2] if len(path) > 1 else np.zeros(2),
            "Active": False,
            "Parent": None,
            "ParentID": None,
            "Children": [],
            "DaughterIDs": [],
            "MergedOriginalTrackIDs": chain,
        })

    tracked_mask_refined = _paint_refined_tracks(refined_tracks, labeled_masks)

    refinement_info = {
        "accepted_links": accepted_links,
        "number_original_tracks": int(len(dict_list)),
        "number_refined_tracks": int(len(refined_tracks)),
        "number_gap_closed_links": int(len(accepted_links)),
        "successor": successor,
        "predecessor": predecessor,
    }

    return refined_tracks, tracked_mask_refined, refinement_info

def get_track_frames(track):
    """
    Return the frame indices of a track.
    Works with the refined version that has 'Frames',
    and also with older tracks using StartIndex.
    """
    if "Frames" in track and len(track["Frames"]) == len(track["Path"]):
        return np.asarray(track["Frames"], dtype=int)

    start = int(track.get("StartIndex", 0))
    return np.arange(start, start + len(track["Path"]))


def summarize_tracks(dict_list, name="tracks"):
    """
    Compute simple diagnostics for a list of ARGUS tracks.
    """
    rows = []

    for tr in dict_list:
        path = np.asarray(tr["Path"], dtype=float)
        frames = get_track_frames(tr)

        if len(path) == 0:
            continue

        if len(path) > 1:
            steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
            total_distance = float(np.sum(steps))
            mean_step = float(np.mean(steps))
            max_step = float(np.max(steps))
        else:
            total_distance = 0.0
            mean_step = 0.0
            max_step = 0.0

        duration = int(frames[-1] - frames[0] + 1)
        n_points = int(len(path))
        n_missing = int(duration - n_points)

        rows.append({
            "track_id": tr.get("ID", None),
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "duration": duration,
            "n_points": n_points,
            "n_missing_inside_track": n_missing,
            "total_distance": total_distance,
            "mean_step": mean_step,
            "max_step": max_step,
            "merged_original_ids": tr.get("MergedOriginalTrackIDs", None),
        })

    df = pd.DataFrame(rows)

    print(f"\n{name}")
    print("-" * len(name))
    print("Number of tracks:", len(df))

    if len(df) > 0:
        print("Mean duration:", df["duration"].mean())
        print("Median duration:", df["duration"].median())
        print("Max duration:", df["duration"].max())
        print("Mean number of points:", df["n_points"].mean())
        print("Number of tracks longer than 10 frames:", np.sum(df["duration"] >= 10))

    return df

def plot_track_length_comparison(df_original, df_refined):
    plt.figure(figsize=(7, 4))

    plt.hist(
        df_original["duration"],
        bins=30,
        alpha=0.5,
        label="Original"
    )

    plt.hist(
        df_refined["duration"],
        bins=30,
        alpha=0.5,
        label="Refined"
    )

    plt.xlabel("Track duration [frames]")
    plt.ylabel("Number of tracks")
    plt.title("Effect of refinement on track duration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



def plot_track_count_comparison(df_original, df_refined):
    names = ["Original", "Refined"]
    values = [len(df_original), len(df_refined)]

    plt.figure(figsize=(4, 4))
    plt.bar(names, values)
    plt.ylabel("Number of tracks")
    plt.title("Track fragmentation before and after refinement")
    plt.grid(True, axis="y", alpha=0.3)
    plt.show()

    print("Original number of tracks:", len(df_original))
    print("Refined number of tracks:", len(df_refined))
    print("Reduction:", len(df_original) - len(df_refined))


def show_merged_tracks(refined_tracks):
    """
    Print refined tracks that were created by merging multiple original tracks.
    """
    merged = []

    for tr in refined_tracks:
        old_ids = tr.get("MergedOriginalTrackIDs", [])
        if old_ids is not None and len(old_ids) > 1:
            frames = get_track_frames(tr)

            merged.append({
                "new_track_id": tr["ID"],
                "merged_original_track_ids": old_ids,
                "start_frame": int(frames[0]),
                "end_frame": int(frames[-1]),
                "duration": int(frames[-1] - frames[0] + 1),
                "n_points": len(tr["Path"]),
                "n_merged_fragments": len(old_ids),
            })

    df = pd.DataFrame(merged)

    print("Number of refined tracks created by merging:", len(df))

    if len(df) > 0:
        display(df)

    return df

def plot_single_track_on_frame(images, track, frame=None, title=None):
    """
    Plot one track over one image frame.
    """
    frames = get_track_frames(track)
    path = np.asarray(track["Path"], dtype=float)

    if frame is None:
        frame = int(frames[len(frames) // 2])

    img = images[frame]

    if img.ndim == 3:
        img_show = img[:, :, 0]
    else:
        img_show = img

    plt.figure(figsize=(6, 6))
    plt.imshow(img_show, cmap="gray")

    plt.plot(path[:, 0], path[:, 1], "-o", linewidth=2, markersize=3)
    plt.scatter(path[0, 0], path[0, 1], s=80, marker="o", label="start")
    plt.scatter(path[-1, 0], path[-1, 1], s=80, marker="x", label="end")

    for k, f in enumerate(frames):
        plt.text(path[k, 0], path[k, 1], str(f), fontsize=8)

    plt.title(title or f"Track {track.get('ID', '?')} on frame {frame}")
    plt.legend()
    plt.axis("off")
    plt.show()


def plot_accepted_gap_links(images, original_tracks, refinement_info, frame=None):
    """
    Visualize links accepted by the refinement step.
    """
    links = refinement_info.get("accepted_links", [])

    if len(links) == 0:
        print("No accepted gap-closing links.")
        return

    if frame is None:
        frame = int(np.median([
            0.5 * (link["end_frame"] + link["start_frame"])
            for link in links
        ]))

    img = images[frame]

    if img.ndim == 3:
        img_show = img[:, :, 0]
    else:
        img_show = img

    plt.figure(figsize=(7, 7))
    plt.imshow(img_show, cmap="gray")

    for link in links:
        i = link["from_track"]
        j = link["to_track"]

        tr_i = original_tracks[i]
        tr_j = original_tracks[j]

        p_end = np.asarray(tr_i["Path"][-1], dtype=float)
        p_start = np.asarray(tr_j["Path"][0], dtype=float)

        plt.plot(
            [p_end[0], p_start[0]],
            [p_end[1], p_start[1]],
            "-",
            linewidth=2
        )

        plt.scatter(p_end[0], p_end[1], s=40)
        plt.scatter(p_start[0], p_start[1], s=40)

        plt.text(
            p_end[0],
            p_end[1],
            f"{i}",
            fontsize=8
        )

        plt.text(
            p_start[0],
            p_start[1],
            f"{j}",
            fontsize=8
        )

    plt.title(f"Accepted refinement links: {len(links)}")
    plt.axis("off")
    plt.show()