import glob
import os
import numpy as np
import scipy
import skimage
import cv2
import itk
import multiprocessing as mp
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from PIL import Image
from bm3d import bm3d
import pandas as pd
import matplotlib.pyplot as plt

IMAGE_CHANNEL = 0
NO_MATCH = np.array([-1, -1])


# ===========================================================================
# I/O
# ===========================================================================

def read_images(directory, is_gray_scale=False):
    files = glob.glob(directory, recursive=False)
    sorted_files = sorted(files)

    images = []
    for img_path in sorted_files:
        if is_gray_scale:
            gray_img = np.array(Image.open(img_path), dtype=np.float64)
            img = np.stack((gray_img,) * 3, axis=-1)
        else:
            img = np.array(Image.open(img_path), dtype=np.float64)
        images.append(img)
    return images


def save_images(images, directory, pattern):
    os.makedirs(directory, exist_ok=True)
    for i in range(len(images)):
        cv2.imwrite(os.path.join(directory, pattern + f"{i:03d}" + ".tif"),
                    np.array(images[i], dtype=np.uint8))


# ===========================================================================
# Rescaling / clipping
# ===========================================================================

def rescale_rgb_to_255(img_array, channel=-1):
    img_array = img_array.astype(np.float32)
    img_rescaled = img_array.copy()

    channel_img = IMAGE_CHANNEL if channel == -1 else channel

    channel_data = img_array[..., channel_img]
    min_val = channel_data.min()
    max_val = channel_data.max()

    if max_val > min_val:
        channel_rescaled = (channel_data - min_val) / (max_val - min_val) * 255
    else:
        channel_rescaled = np.zeros_like(channel_data)

    img_rescaled[..., channel_img] = channel_rescaled
    return img_rescaled.astype(np.uint8)


def batch_rescale_rgb_to_255(image_list):
    return [rescale_rgb_to_255(img) for img in image_list]


def clip_images(images, clip_percentile=95):
    for i in range(len(images)):
        img = images[i][..., IMAGE_CHANNEL].astype(np.float32)
        thresh = np.percentile(img, clip_percentile)
        img[img > thresh] = thresh
        img = (img - img.min()) / (img.max() - img.min()) * 255
        images[i][..., IMAGE_CHANNEL] = img.astype(np.uint8)
    return images


# ===========================================================================
# Registration
# ===========================================================================

def image_registration(fixed_img, moving_img, registration_channel=-1, n_processes=None):
    if n_processes is None:
        n_processes = mp.cpu_count()

    transformed_img = np.zeros_like(moving_img, dtype=moving_img.dtype)

    num_channels = fixed_img.shape[2] if fixed_img.ndim == 3 else 1

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
        raise ValueError(f"Invalid registration_channel: {registration_channel}. "
                         f"Must be -1 or between 0 and {num_channels - 1}.")

    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(parameter_object.GetDefaultParameterMap('rigid'))
    parameter_object.AddParameterMap(parameter_object.GetDefaultParameterMap('affine'))

    _, result_transform_parameters = itk.elastix_registration_method(
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
        transformed_channel_img = itk.transformix_filter(
            itk.GetImageFromArray(moving_img.astype(np.float32)), result_transform_parameters)
        transformed_img = itk.GetArrayFromImage(transformed_channel_img)

    transformed_img[transformed_img < 0] = 0
    return transformed_img


# ===========================================================================
# Mean subtraction
# ===========================================================================

def subtract_mean_channel(images, axis=0):
    images_array = np.array(images)
    mean_img = np.mean(images_array[..., IMAGE_CHANNEL], axis=axis)
    for i in range(len(images)):
        images[i][..., IMAGE_CHANNEL] = np.abs(images[i][..., IMAGE_CHANNEL] - mean_img)
    return images, mean_img


# ===========================================================================
# Denoising / enhancement (per-image worker + parallel driver)
# ===========================================================================

def unsharp_mask(image, sigma=1.0, alpha=1.5):
    """Applies unsharp masking by enhancing edges using a blurred image."""
    blurred = scipy.ndimage.gaussian_filter(image, sigma=sigma)
    return image + alpha * (image - blurred)  # high-boost filtering


def unsharp_single_image(image_data, radius, amount):
    i, image = image_data

    unsharp_masked = unsharp_mask(image[..., IMAGE_CHANNEL], sigma=radius, alpha=amount)
    unsharp_masked[unsharp_masked < np.std(unsharp_masked)] = 0
    binary_mask = unsharp_masked > 0

    binary_mask = scipy.ndimage.binary_closing(binary_mask, iterations=1)
    binary_mask = skimage.morphology.remove_small_objects(binary_mask, min_size=6)

    result = image.copy()
    result[..., IMAGE_CHANNEL] = unsharp_masked * binary_mask
    return result


def median_denoise_single_image(image_data, size):
    i, image = image_data
    result = image.copy()
    result[..., IMAGE_CHANNEL] = scipy.ndimage.median_filter(image[..., IMAGE_CHANNEL], size=size)
    return result


def log_transform_single_image(image_data):
    i, image = image_data
    c = 255 / np.log(1 + np.max(image[..., IMAGE_CHANNEL]))
    result = image.copy()
    result[..., IMAGE_CHANNEL] = c * (np.log(image[..., IMAGE_CHANNEL] + 1))
    return result


def bm3d_denoise_single_image(image_data):
    noise_std = np.std(image_data)
    result = image_data.copy()
    result[..., IMAGE_CHANNEL] = bm3d(image_data[..., IMAGE_CHANNEL], noise_std)
    return result


def wavelet_denoise_single_image(image_data):
    image_est = skimage.restoration.denoise_wavelet(image_data[..., IMAGE_CHANNEL])
    result = image_data.copy()
    result[..., IMAGE_CHANNEL] = np.array(image_est * 255, dtype=np.uint8)
    return result


def unsharp_mask_all(images, radius=8, amount=20, n_processes=None):
    n_processes = n_processes or mp.cpu_count()
    with mp.Pool(processes=n_processes) as pool:
        image_data = [(i, img) for i, img in enumerate(images)]
        process_func = partial(unsharp_single_image, radius=radius, amount=amount)
        return pool.map(process_func, image_data)


def median_denoise_all(images, size, n_processes=None):
    n_processes = n_processes or mp.cpu_count()
    with mp.Pool(processes=n_processes) as pool:
        image_data = [(i, img) for i, img in enumerate(images)]
        process_func = partial(median_denoise_single_image, size=size)
        return pool.map(process_func, image_data)


def log_transform_all(images, n_processes=None):
    n_processes = n_processes or mp.cpu_count()
    with mp.Pool(processes=n_processes) as pool:
        image_data = [(i, img) for i, img in enumerate(images)]
        return pool.map(log_transform_single_image, image_data)


def bm3d_denoise_all(images, max_workers=None):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(bm3d_denoise_single_image, images))


def wavelet_denoise_images_all(images):
    """Apply wavelet denoising to a list of images."""
    return [wavelet_denoise_single_image(img) for img in images]


# ===========================================================================
# Monogenic enhancement (log-Gabor local energy)
# ===========================================================================

def freqgrid2(ysize, xsize):
    fy = np.fft.fftfreq(ysize)
    fx = np.fft.fftfreq(xsize)
    xGrid, yGrid = np.meshgrid(fx, fy, indexing='xy')
    return yGrid, xGrid


def create_monogenic_filters_log_gabor(ysize, xsize, wl, sigma_onf=0.55):
    yGrid, xGrid = freqgrid2(ysize, xsize)
    w = np.sqrt(yGrid**2 + xGrid**2)
    w[0, 0] = 1.0  # avoid log(0) at DC during filter construction

    fo = 1.0 / wl
    bpFilt = np.exp(-(np.log(w / fo) ** 2) / (2 * (np.log(sigma_onf) ** 2)))
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


def monogenic_signal(image2d, filtStruct):
    F = np.fft.fft2(image2d)
    Ffilt = F * filtStruct["bpFilt"]

    m1 = np.real(np.fft.ifft2(Ffilt))                       # even component
    Fmodd = np.fft.ifft2(Ffilt * filtStruct["ReiszFilt"])   # odd components (Riesz)
    m2 = np.real(Fmodd)
    m3 = np.imag(Fmodd)
    return m1, m2, m3


def local_energy(m1, m2, m3):
    return m1**2 + m2**2 + m3**2


def monogenic_filter_single_image(image_data, cw=60, sigma_onf=0.05, sigma_smooth=8):
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
        x = np.clip(x, 0.0, 1.0)          # clip BEFORE casting
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


# ===========================================================================
# Cell detection: centroids
# ===========================================================================

def detect_centers(image, cell_size, min_distance, manual_threshold=-1):
    thresh = skimage.filters.threshold_otsu(image) if manual_threshold == -1 else manual_threshold
    thresh_img = image > thresh
    thresh_img = skimage.morphology.remove_small_objects(thresh_img, min_size=cell_size)

    labeled_mask = skimage.measure.label(thresh_img)
    centers = []  # ((y, x), area, original_label)
    for region in skimage.measure.regionprops(labeled_mask):
        y, x = region.centroid
        centers.append(((int(round(y)), int(round(x))), region.area, region.label))

    # Decide which centers to keep
    if min_distance is None or min_distance <= 0:
        kept = [(c, lbl) for (c, _a, lbl) in centers]
    else:
        centers.sort(key=lambda t: t[1], reverse=True)  # prefer larger regions
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

    out_mask = np.zeros_like(labeled_mask, dtype=np.int32)
    kept_points = []
    centroid_label_pairs = []
    for new_label, ((cy, cx), orig_label) in enumerate(kept, start=1):
        out_mask[labeled_mask == orig_label] = new_label
        kept_points.append((cy, cx))
        centroid_label_pairs.append(((cy, cx), new_label))

    return np.array(kept_points, dtype=int), out_mask, centroid_label_pairs


def locate_all_cells_centroids(images, cell_size, min_distance, manual_threshold=-1):
    edges_list, mask_list, labels_list = [], [], []
    for image in images:
        edges, mask, centroid_label_pairs = detect_centers(
            image[:, :, IMAGE_CHANNEL],
            cell_size=cell_size,
            min_distance=min_distance,
            manual_threshold=manual_threshold,
        )
        # Swap (y, x) -> (x, y); centroid_label_pairs is in the SAME order as edges.
        edges_copy = edges.copy()
        edges[:, 0], edges[:, 1] = edges_copy[:, 1], edges_copy[:, 0]
        labels = np.array([lbl for (_pt, lbl) in centroid_label_pairs], dtype=np.int32)

        edges_list.append(edges)
        mask_list.append(mask)
        labels_list.append(labels)

    return edges_list, mask_list, labels_list


# ===========================================================================
# Cell detection: local maxima
# ===========================================================================

def locate_cells(image, min_size=10, expected_mask_size=15, min_distance=10, manual_threshold=-1):
    thresh = skimage.filters.threshold_otsu(image) if manual_threshold == -1 else manual_threshold
    thresh_img = image > thresh
    thresh_img = skimage.morphology.remove_small_objects(thresh_img, min_size=min_size)

    labeled_mask = skimage.measure.label(thresh_img)
    result_coords = []

    for region in skimage.measure.regionprops(labeled_mask):
        mask_size = region.area
        if mask_size > expected_mask_size:
            max_allowed = 2
        elif mask_size > 2 * expected_mask_size:
            max_allowed = 3
        elif mask_size > 3 * expected_mask_size:
            max_allowed = 4
        else:
            max_allowed = 5

        minr, minc, maxr, maxc = region.bbox
        region_image = image[minr:maxr, minc:maxc] * (labeled_mask[minr:maxr, minc:maxc] == region.label)
        region_maxima = skimage.feature.peak_local_max(region_image, min_distance=min_distance, exclude_border=False)
        region_maxima = [(y + minr, x + minc) for y, x in region_maxima]

        if len(region_maxima) > max_allowed:
            selected = [region_maxima[0]]
            while len(selected) < max_allowed and len(region_maxima) > 1:
                farthest_point = max(
                    region_maxima,
                    key=lambda p: min(np.linalg.norm(np.array(p) - np.array(s)) for s in selected),
                )
                selected.append(farthest_point)
                region_maxima.remove(farthest_point)
            region_maxima = selected

        result_coords.extend(region_maxima)

    return np.array(result_coords), thresh_img


def locate_all_cells_local_maxima(images, min_size=10, expected_mask_size=15, min_distance=10, manual_threshold=-1):
    edges_list, mask_list = [], []
    for image in images:
        edges, mask = locate_cells(
            image[:, :, IMAGE_CHANNEL],
            min_size=min_size, expected_mask_size=expected_mask_size,
            min_distance=min_distance, manual_threshold=manual_threshold,
        )
        edges_copy = edges.copy()
        edges[:, 0], edges[:, 1] = edges_copy[:, 1], edges_copy[:, 0]
        edges_list.append(edges)
        mask_list.append(mask)
    return edges_list, mask_list


# ===========================================================================
# Cell detection: monogenic shape (feature symmetry) masks
# ===========================================================================

def get_monogenic_filters(sizes, scales, filt_type='Cauchy'):
    aspr = 1
    ysize, xsize = sizes[0], sizes[1]

    ymid, xmid = ysize // 2, xsize // 2
    ymax = ymid - 1 if ysize % 2 == 0 else ymid
    xmax = xmid - 1 if xsize % 2 == 0 else xmid

    Y, X = np.meshgrid(np.arange(-ymid, ymax + 1), np.arange(-xmid, xmax + 1), indexing='ij')
    Y = np.fft.ifftshift(Y) / ysize
    X = np.fft.ifftshift(X) / (xsize * aspr)

    if filt_type == 'Poisson':
        ratio = 0.9

    freq = np.sqrt(Y**2 + X**2)
    freq[0, 0] = 1  # prevent division by zero at DC

    num_scales = len(scales)
    bp_filter = np.zeros((ysize, xsize, num_scales))

    for s in range(num_scales):
        scale = scales[s]
        if filt_type == 'Poisson':
            s2 = scale / (ratio - 1) * np.log(ratio)  # difference of Poissons
            s1 = ratio * s2
            bp_filter[:, :, s] = freq * (np.exp(-freq * s1) - np.exp(-freq * s2))
        elif filt_type == 'GaussianD':
            bp_filter[:, :, s] = freq**2 * np.exp(-freq**2 * scale)  # Gaussian derivative
        else:  # Cauchy
            bp_filter[:, :, s] = freq * np.exp(-freq * scale)

        bp_filter[0, 0, s] = 0
        if ysize % 2 == 0:
            bp_filter[ysize // 2, :, s] = 0
        if xsize % 2 == 0:
            bp_filter[:, xsize // 2, s] = 0

    bp_filter = bp_filter / np.max(np.sum(bp_filter, axis=2))
    odd_filter = (1j * Y - X) / freq
    return bp_filter, odd_filter, X, Y


def monogenic_shape_detection_v2(im, T, cw_monogenic):
    nx, nz = im.shape
    evenfilt, oddfilt, _, _ = get_monogenic_filters([nx, nz], cw_monogenic, 'Poisson')

    F = np.fft.fft2(im)

    Ffilt = np.zeros((nz, nx, len(cw_monogenic)), dtype=complex)
    for s in range(len(cw_monogenic)):
        Ffilt[:, :, s] = F * evenfilt[:, :, s]

    p = np.real(np.fft.ifft2(Ffilt, axes=(0, 1)))  # even component

    Fmodd = np.zeros_like(Ffilt, dtype=complex)
    for s in range(len(cw_monogenic)):
        Fmodd[:, :, s] = np.fft.ifft2(Ffilt[:, :, s] * oddfilt, axes=(0, 1))

    q1 = np.real(Fmodd)
    q2 = np.imag(Fmodd)

    ysize, xsize, ssize = p.shape
    epsilon = 0.001

    even = np.abs(p)
    odd = np.sqrt(q1 * q1 + q2 * q2)

    denominator = np.maximum(np.sqrt(even * even + q1 * q1 + q2 * q2),
                             epsilon * np.ones((ysize, xsize, ssize)))
    SFS_numerator = np.maximum(even - odd - T, np.zeros((ysize, xsize, ssize)))

    SFS = (SFS_numerator / denominator) * np.sign(p)
    SFS = np.median(SFS, axis=2)
    SFS[SFS < T] = 0
    return SFS


def process_single_image_mask(image, T, cw_monogenic):
    I = image[:, :, 1].astype(np.float64)
    return monogenic_shape_detection_v2(I, T, cw_monogenic)


def locate_cells_monogenic_filter(image, thresh_img, min_distance=5):
    binary_mask_dilated = thresh_img > 0
    labeled_mask, num_features = skimage.measure.label(binary_mask_dilated, return_num=True)
    labeled_mask = skimage.morphology.remove_small_objects(labeled_mask, min_size=2)

    all_maxima = []
    for region_id in range(1, num_features + 1):
        region_mask = labeled_mask == region_id
        props = skimage.measure.regionprops(region_mask.astype(int))
        if not props:
            continue

        y_min, x_min, y_max, x_max = props[0].bbox
        region_image = image[y_min:y_max, x_min:x_max]
        region_binary = region_mask[y_min:y_max, x_min:x_max]
        masked_region = region_image * region_binary

        local_maxima = skimage.feature.peak_local_max(
            masked_region, min_distance=min_distance, num_peaks=2, exclude_border=1)

        if local_maxima.size > 0:
            local_maxima[:, 0] += y_min
            local_maxima[:, 1] += x_min
            all_maxima.append(local_maxima)

    if all_maxima:
        return np.vstack(all_maxima)
    return np.empty((0, 2), dtype=int)


def process_single_cell_detection(args):
    idx, image, mask, min_distance = args
    edges = locate_cells_monogenic_filter(image[:, :, 1], mask, min_distance=min_distance)
    edges[:, [0, 1]] = edges[:, [1, 0]]  # swap coordinates
    return idx, edges


def monogenic_filter(images, cw_monogenic=np.array([50.0, 60.0, 70.0]), T=0.6, num_processes=None):
    num_processes = num_processes or mp.cpu_count()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        process_func = partial(process_single_image_mask, T=T, cw_monogenic=cw_monogenic)
        return list(executor.map(process_func, images))


def locate_all_cells_monogenic_filter(images, cw_monogenic=np.array([50.0, 60.0, 70.0]),
                                      T=0.6, min_distance=5, num_processes=None):
    num_processes = num_processes or mp.cpu_count()

    all_mask = monogenic_filter(images, cw_monogenic, T, num_processes)

    edges_list = [None] * len(images)
    cell_detection_args = [(i, images[i], all_mask[i], min_distance) for i in range(len(images))]
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for idx, edges in executor.map(process_single_cell_detection, cell_detection_args):
            edges_list[idx] = edges

    return edges_list, all_mask


# ===========================================================================
# Optical flow / matching helpers
# ===========================================================================

def average_direction(points):
    displacements = np.diff(points, axis=0)
    avg_vector = np.mean(displacements, axis=0)
    norm = np.linalg.norm(avg_vector)
    avg_direction = avg_vector / norm if norm != 0 else (0, 0)
    angle_degrees = np.degrees(np.arctan2(avg_vector[1], avg_vector[0]))
    return avg_direction, angle_degrees, avg_vector


def compute_optical_flow(prev_img, next_img):
    # OpenCV requires uint8 or float32
    if prev_img.dtype not in (np.uint8, np.float32):
        prev_img = prev_img.astype(np.float32)
        next_img = next_img.astype(np.float32)
        lo = min(prev_img.min(), next_img.min())
        hi = max(prev_img.max(), next_img.max())
        if hi > lo:
            prev_img = (prev_img - lo) / (hi - lo)
            next_img = (next_img - lo) / (hi - lo)

    flow = cv2.calcOpticalFlowFarneback(
        prev_img, next_img, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    # (H, W, 2) with (dx, dy) -> (2, H, W) with (dy, dx) to match skimage
    return flow.transpose(2, 0, 1)[::-1]


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


def hungarian_assignment(flow_points, edge_points, max_distance):
    if len(flow_points) == 0:
        return np.array([]), edge_points

    if len(edge_points) == 0:
        closest_points = np.full((len(flow_points), 2), [-1, -1])
        return closest_points, np.array([])

    # Cost matrix: distances, with a prohibitive cost beyond max_distance.
    cost_matrix = np.zeros((len(flow_points), len(edge_points)))
    for i, flow_point in enumerate(flow_points):
        for j, edge_point in enumerate(edge_points):
            distance = np.linalg.norm(flow_point - edge_point)
            cost_matrix[i, j] = distance if distance <= max_distance else 1e6

    flow_indices, edge_indices = scipy.optimize.linear_sum_assignment(cost_matrix)

    closest_points = np.full((len(flow_points), 2), [-1, -1], dtype=float)
    matched_edge_indices = set()
    for flow_idx, edge_idx in zip(flow_indices, edge_indices):
        if cost_matrix[flow_idx, edge_idx] <= max_distance:
            closest_points[flow_idx] = edge_points[edge_idx]
            matched_edge_indices.add(edge_idx)

    unmatched_points_B = np.array([
        edge_points[i] for i in range(len(edge_points)) if i not in matched_edge_indices])

    return closest_points, unmatched_points_B


# ---------------------------------------------------------------------------
# Track entries
# ---------------------------------------------------------------------------

def _new_track(track_id, frame_idx, point, label, parent_id=None):
    """Create a fresh track entry."""
    return {
        "ID": track_id,
        "StartIndex": frame_idx,
        "Frames": [frame_idx],
        "Path": [point],
        "MaskLabels": [label],
        "AverageDir": np.zeros(2),
        "Active": True,
        "Parent": parent_id,
        "ParentID": parent_id,
        "Children": [],
        "DaughterIDs": [],
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


def _spawn_track(dict_list, tracked_mask, labeled_mask, frame_idx, point, label, parent_id=None):
    """Append a new track and paint its mask. Returns the new track ID."""
    new_id = len(dict_list)
    dict_list.append(_new_track(new_id, frame_idx, point, label, parent_id))
    _paint(tracked_mask, frame_idx, labeled_mask, label, new_id)
    return new_id


def _extend_track(entry, point, label, frame_idx, tracked_mask, labeled_mask):
    """Append a real detection to a track."""
    entry["Path"].append(point)
    entry.setdefault("Frames", []).append(frame_idx)
    entry["MaskLabels"].append(label)
    entry["OcclusionCount"] = 0
    entry["LastSeenIndex"] = frame_idx
    entry["GhostPos"] = np.asarray(point, dtype=np.float32)
    _paint(tracked_mask, frame_idx, labeled_mask, label, entry["ID"])
    if len(entry["Path"]) > 1:
        _, _, entry["AverageDir"] = average_direction(np.array(entry["Path"]))


def _mark_occluded(entry, flow_x, flow_y, max_occlusion_frames):
    """No detection this frame: advect ghost, retire if past occlusion budget."""
    entry["GhostPos"] = _advect_points(entry["GhostPos"].reshape(1, 2), flow_x, flow_y)[0]
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


def _find_split_partner(prediction, matched_label, unmatched_points, consumed, label_of, max_distance):
    """
    Find the nearest unmatched detection within `max_distance` of `prediction`
    that lies in a *different* mask than `matched_label`.
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
    Roll back the parent's tentative extension into `frame_idx` and replace it
    with two new daughter tracks (mitosis-like event).
    """
    parent["Path"].pop()
    parent["MaskLabels"].pop()
    if "Frames" in parent and len(parent["Frames"]) > 0:
        parent["Frames"].pop()
    parent["Active"] = False
    tracked_mask[frame_idx][tracked_mask[frame_idx] == parent["ID"] + 1] = 0

    a_point, a_label = daughter_a
    b_point, b_label = daughter_b
    a_id = _spawn_track(dict_list, tracked_mask, labeled_mask, frame_idx, a_point, a_label, parent_id=parent["ID"])
    b_id = _spawn_track(dict_list, tracked_mask, labeled_mask, frame_idx, b_point, b_label, parent_id=parent["ID"])
    parent["Children"] = [a_id, b_id]
    parent["DaughterIDs"] = [a_id, b_id]


# ===========================================================================
# Main tracker (optical flow)
# ===========================================================================

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
        _spawn_track(dict_list, tracked_mask, labeled_masks[0], frame_idx=0, point=point, label=int(label))
 
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
        current_points = np.array([_current_position(dict_list[i]) for i in active_indices])
        predicted = _advect_points(current_points, flow_x, flow_y)
 
        # 2. Match predictions to detections.
        detected = all_points[t_next]
        matches, unmatched_points = hungarian_assignment(predicted, detected, max_distance=max_distance)
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
                _extend_track(entry, match, label_of(match), t_next, tracked_mask, labeled_masks[t_next])
 
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
                max_distance=4*max_distance,
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
 
    # `flows` is returned so refinement can reuse it instead of recomputing.
    return dict_list, average_flow, tracked_mask, flows

def _track_frames(entry):
    if "Frames" in entry and len(entry["Frames"]) == len(entry.get("Path", [])):
        return [int(f) for f in entry["Frames"]]
    start = int(entry.get("StartIndex", 0))
    return list(range(start, start + len(entry.get("Path", []))))


def _advect_point_over_gap(point, flows, start_frame, end_frame):
    """
    Move a point from `start_frame` to `end_frame` using the optical-flow fields.
    `flows[t]` maps frame t -> frame t+1.
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
    path_a = track_a.get("Path", [])
    path_b = track_b.get("Path", [])
    if len(path_a) < 2 or len(path_b) < 2:
        return 0.0

    va = np.asarray(path_a[-1], dtype=np.float32) - np.asarray(path_a[-2], dtype=np.float32)
    vb = np.asarray(path_b[1], dtype=np.float32) - np.asarray(path_b[0], dtype=np.float32)

    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(1.0 - np.dot(va / na, vb / nb))


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
            if f < 0 or f >= T or label is None or int(label) < 0:
                continue
            tracked_mask[f][labeled_masks[f] == int(label)] = track_id + 1

    return tracked_mask

def refine_tracks_by_gap_closing(dict_list, images, labeled_masks, max_distance,
                                 max_gap=5, gap_penalty=0.25, direction_weight=0.25, no_link_cost=2.5,
                                 flows=None):
    if len(dict_list) == 0:
        if len(labeled_masks) == 0:
            return [], np.zeros((0, 0, 0), dtype=np.int32), {"accepted_links": []}
        T = len(labeled_masks)
        H, W = labeled_masks[0].shape
        return [], np.zeros((T, H, W), dtype=np.int32), {"accepted_links": []}

    # Reuse flows from the tracker when supplied; only compute as a fallback.
    if flows is None:
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

            predicted = _advect_point_over_gap(end_point, flows, end_frame, start_frame)
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

        frames, path, labels = [], [], []
        for old_id in chain:
            tr = dict_list[old_id]
            tr_frames = _track_frames(tr)
            frames.extend(tr_frames)
            path.extend([np.asarray(p, dtype=np.float32) for p in tr.get("Path", [])])
            labels.extend([int(l) if l is not None else None for l in tr.get("MaskLabels", [])])
            old_to_new[old_id] = len(refined_tracks)

        if len(path) == 0:
            continue

        # Sort by frame in case fragments are non-monotone.
        order = np.argsort(np.asarray(frames, dtype=np.int32))
        frames = [int(frames[k]) for k in order]
        path = [path[k] for k in order]
        labels = [labels[k] for k in order]

        # Carry over parent/lineage from the first fragment in the chain (the root).
        root_track = dict_list[chain[0]]
        parent_id_original = root_track.get("ParentID")
        if parent_id_original is None and isinstance(root_track.get("Parent"), int):
            parent_id_original = root_track["Parent"]

        # Remap parent ID to new refined ID if already processed, else defer.
        remapped_parent = old_to_new.get(parent_id_original) if parent_id_original is not None else None

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
            "Parent": remapped_parent,
            "ParentID": remapped_parent,
            "Children": [],
            "DaughterIDs": [],
            "MergedOriginalTrackIDs": chain,
            "_unresolved_parent_original_id": parent_id_original if remapped_parent is None else None,
        })

    # Second pass: resolve parent IDs whose refined counterpart wasn't yet
    # assigned when the child was processed (parent root came later in roots).
    for tr in refined_tracks:
        orig_parent = tr.pop("_unresolved_parent_original_id", None)
        if orig_parent is not None and orig_parent in old_to_new:
            tr["Parent"] = old_to_new[orig_parent]
            tr["ParentID"] = old_to_new[orig_parent]

    # Third pass: rebuild Children / DaughterIDs from the resolved ParentIDs.
    for tr in refined_tracks:
        p = tr.get("ParentID")
        if p is not None and 0 <= p < len(refined_tracks):
            if tr["ID"] not in refined_tracks[p]["Children"]:
                refined_tracks[p]["Children"].append(tr["ID"])
                refined_tracks[p]["DaughterIDs"].append(tr["ID"])

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