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

IMAGE_CHANNEL = 0

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

def monogenic_filter_single_image(image_data):
    i, image = image_data
    
    image_ch = image[..., IMAGE_CHANNEL]
    result = image.copy()

    Y, X = image_ch.shape

    cw = min(Y, X) / 5.0   # should be optimized

    filtStruct = create_monogenic_filters_log_gabor(Y, X, wl=cw, sigma_onf=0.55)
    m1, m2, m3 = monogenic_signal(image_ch, filtStruct)
    LE = local_energy(m1, m2, m3)

    psf = gaussian_psf_2d(size=int(round(cw)), std=cw / 100.0)
    image_est = skimage.restoration.richardson_lucy(LE, psf, num_iter=20, clip=False)  # we need to change this

    result[..., IMAGE_CHANNEL] = image_est

    return result

def monogenic_filter_all_images(images, n_processes=None):
    # If n_processes is not specified, use all available CPU cores
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    # Create a pool of workers
    with mp.Pool(processes=n_processes) as pool:
        # Create a list of tuples (index, image) to pass to the worker function
        image_data = [(i, img) for i, img in enumerate(images)]

        # Use partial to create a function with fixed radius parameter
        process_func = partial(monogenic_filter_single_image)
        
        # Process images in parallel and gather results
        results = pool.map(process_func, image_data)
    
    return results

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

def detect_centers_old(image, min_size, min_distance, iterations = 3, manual_threshold = -1):
    if manual_threshold == -1:
        thresh = skimage.filters.threshold_otsu(image)
    else:
        thresh = manual_threshold
    
    thresh_img = image > thresh
    
    # Remove small objects
    if iterations>0:
        thresh_img = scipy.ndimage.binary_closing(skimage.morphology.remove_small_objects(thresh_img, min_size=min_size), iterations = iterations)
    else:
        thresh_img = skimage.morphology.remove_small_objects(thresh_img, min_size=min_size)
    
    # Label connected components
    labeled_mask = skimage.measure.label(thresh_img)

    centers = []
    for region in skimage.measure.regionprops(labeled_mask):
        y, x = region.centroid  # (row, col)
        centers.append(((int(round(y)), int(round(x))), region.area))
        
    if min_distance is None or min_distance <= 0:
        return np.array([c for (c, _) in centers], dtype=int), thresh_img

    # Prefer larger regions: sort by area (desc)
    centers.sort(key=lambda t: t[1], reverse=True)

    # Greedy non-maximum suppression by distance
    kept = []
    for (cy, cx), _area in centers:
        if not kept:
            kept.append((cy, cx))
            continue
        d = np.linalg.norm(np.array(kept) - np.array([cy, cx]), axis=1)
        if np.all(d >= min_distance):
            kept.append((cy, cx))

    return np.array(kept, dtype=int), thresh_img

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

def locate_all_cells_centroids(images, min_size, min_distance, iterations, manual_threshold = -1):
    edges_list = []
    mask_list = []
    for image in images:
        edges, mask = detect_centers(image[:,:,IMAGE_CHANNEL], min_size=min_size, min_distance = min_distance, iterations = iterations, manual_threshold = manual_threshold)
        edges_copy = edges.copy()  # Optional: create a copy to avoid modifying in-place
        edges[:, 0], edges[:, 1] = edges_copy[:, 1], edges_copy[:, 0]
        edges_list.append(edges)
        mask_list.append(mask)
    return edges_list, mask_list

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

def compute_optical_flow(prev_img, next_img):
    return skimage.registration.optical_flow_tvl1(prev_img, next_img, attachment=5)

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

def track_points_optical_flow(images, all_points, mask_list, max_distance):
    initial_points = all_points[0]
    green_images = [img[:, :, IMAGE_CHANNEL] if img.ndim == 3 else img for img in images]

    labeled_mask_list = [scipy.ndimage.label(mask)[0] for mask in mask_list]
    
    dict_list = []
    for i in range(initial_points.shape[0]):
        # (x,y) needs to be swaped to (y,x)
        curr_mask = find_current_mask(point= initial_points[i], mask = labeled_mask_list[0])
        
        item = {
            "Path": [initial_points[i]],
            "AverageDir": [0, 0],
            "StartIndex": 0,
            "Active": True,
            "ID": i,
            "Mask": [curr_mask]
        }
        dict_list.append(item)
        
    current_points = initial_points
    average_flow = np.zeros_like(green_images[0], dtype=np.float32)
    
    # Parallel computation of optical flow for all frames
    with mp.Pool(mp.cpu_count()) as pool:
        flows = pool.starmap(compute_optical_flow, [(green_images[i], green_images[i + 1]) for i in range(len(images) - 1)])
    
    for i_images, flow in enumerate(flows):
        flow_y, flow_x = flow
        flow_x[flow_x < 0.05] = 0
        flow_y[flow_y < 0.05] = 0
        average_flow += np.sqrt(flow_x**2 + flow_y**2)
        
        h, w = flow_x.shape
        flow_points = np.array([
            [
                x + flow_x[min(max(int(y), 0), h - 1), min(max(int(x), 0), w - 1)],
                y + flow_y[min(max(int(y), 0), h - 1), min(max(int(x), 0), w - 1)]
            ]
            for x, y in current_points
        ])
        
        calculated_edges = all_points[i_images + 1]
        
        matches, unmatched_points = hungarian_assignment(flow_points, calculated_edges, max_distance=max_distance)
        
        for i_dict_list, entry in enumerate(dict_list):
            if not entry["Active"]:
                continue
            
            if not np.array_equal(matches[i_dict_list], [-1, -1]):
                entry["Path"].append(matches[i_dict_list])
                
                curr_mask = find_current_mask(point = matches[i_dict_list], mask = labeled_mask_list[i_images])
                entry["Mask"].append(curr_mask)
                
                if len(entry["Path"]) > 1:
                    _, _, avg_dir = average_direction(np.array(entry["Path"]))
                    entry["AverageDir"] = avg_dir
                    
        for idx, point in enumerate(unmatched_points):
            curr_mask = find_current_mask(point= point, mask = labeled_mask_list[i_images])
                
            dict_list.append({
                "Path": [point],
                "AverageDir": [0, 0],
                "StartIndex": i_images,
                "Active": True,
                "ID": len(dict_list)+idx,
                "Mask": [curr_mask]
            })
            
        current_points = np.array([entry["Path"][-1] for entry in dict_list])
    
    return dict_list, average_flow

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
    
    # Compute the unsharp mask on the green channel
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
    i, image = image_data

    # Compute the unsharp mask on the green channel
    image_est = skimage.restoration.denoise_wavelet(image[..., IMAGE_CHANNEL]);
    
    result = image.copy()
    result[..., IMAGE_CHANNEL] = np.array(rescale_rgb_to_255(image_est, channel = None),dtype = np.uint8)

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

def wavelet_denoise_all(images, n_processes=None):
    # If n_processes is not specified, use all available CPU cores
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    # Create a pool of workers
    with mp.Pool(processes=n_processes) as pool:
        # Create a list of tuples (index, image) to pass to the worker function
        image_data = [(i, img) for i, img in enumerate(images)]

        # Use partial to create a function with fixed radius parameter
        process_func = partial(wavelet_denoise_single_image)
        
        # Process images in parallel and gather results
        results = pool.map(process_func, image_data)
    
    return results
