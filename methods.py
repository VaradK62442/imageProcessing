"""
Segmentation methods.
"""

import cv2
import numpy as np

from moments import centroid_ellipsoid


def thresholding(frame: cv2.Mat, threshold = 0.5) -> cv2.Mat:
    """
    Given a frame:
    1. Convert it to greyscale.
    2. Apply a binary threshold to the greyscale image.
    3. Apply the threshold to the original frame to get colour pixels.

    Args:
        frame: The input frame.
        threshold: The threshold value.

    Returns:
        Two by two tiling of original frame, greyscale frame,
        thresholded frame, and threshold applied to original frame.
    """

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grey, int(threshold * 255), 255, cv2.THRESH_BINARY)[1]
    colour_thresh = cv2.bitwise_and(frame, frame, mask=thresh)

    return cv2.vconcat([
        cv2.hconcat([cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), frame]),
        cv2.hconcat([colour_thresh, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)]),
    ])


def estimating_thresholding(
        frame: cv2.Mat,
        initial_threshold: float = 0.5,
        t_zero: float = 1
) -> cv2.Mat:
    """
    Given a frame:
    1. Convert it to greyscale.
    2. Apply a binary threshold to the greyscale image.
    3. Calculate mean grey value of pixels below and above the threshold.
    4. Calculate new threshold based on the mean values.
    5. Keep iterating until the successive difference is less than t_zero.
    6. Apply the threshold to the original frame to get colour pixels.

    Args:
        frame: The input frame.
        initial_threshold: The initial threshold value.
        t_zero: The threshold for stopping the iteration.

    Returns:
        Two by two tiling of original frame, greyscale frame,
        thresholded frame, and threshold applied to original frame.
    """

    def _iterate(threshold: float) -> float:
        """
        Helper function to iterate the thresholding process.
        """
        thresh = cv2.threshold(grey, int(threshold), 255, cv2.THRESH_BINARY)[1]

        # Calculate mean grey value of pixels below and above the threshold
        mean_below = np.mean(grey[thresh == 0])
        mean_above = np.mean(grey[thresh > 0])

        new_thresh = (mean_below + mean_above) / 2
        return new_thresh

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grey, int(initial_threshold * 255), 255, cv2.THRESH_BINARY)[1]

    # Calculate mean grey value of pixels below and above the threshold
    mean_below = np.mean(grey[thresh == 0])
    mean_above = np.mean(grey[thresh == 255])

    threshold = (mean_below + mean_above) / 2
    diff = np.abs(threshold - initial_threshold)

    while diff > t_zero:
        new_thresh = _iterate(threshold)
        diff = np.abs(new_thresh - threshold)
        threshold = new_thresh

    try:
        thresh = cv2.threshold(grey, int(threshold), 255, cv2.THRESH_BINARY)[1]
    except ValueError as e:
        return frame
    
    print(f"Using threshold {threshold:.2f} with diff {diff:.2f}")
    colour_thresh = cv2.bitwise_and(frame, frame, mask=thresh)

    return cv2.vconcat([
        cv2.hconcat([cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), frame]),
        cv2.hconcat([colour_thresh, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)]),
    ])


def intensity_rg_by(frame: cv2.Mat):
    """
    Given a frame, find intensity, R-G, B-Y colour space.

    Args:
        frame: The input frame.

    Returns:
        Two by two tiling of original frame, intensity frame,
        R-G colour space, B-Y colour space.
    """
    r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]

    intensity = (r + g + b) / 3
    rg = (r - g) / (r + g)
    yellow = (r + g) / 2
    by = (b - yellow) / (b + yellow)

    intensity = cv2.normalize(intensity, None, 0, 255, cv2.NORM_MINMAX)
    by = cv2.normalize(by, None, 0, 255, cv2.NORM_MINMAX)

    intensity = cv2.cvtColor(intensity.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    rg = cv2.cvtColor(rg.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    by = cv2.cvtColor(by.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return cv2.vconcat([
        cv2.hconcat([intensity, frame]),
        cv2.hconcat([rg, by]),
    ])


def moments(frame: cv2.Mat, threshold: float = 0.5) -> cv2.Mat:
    """
    Given a frame, find the centroid and ellipsoid of the image.
    Convert to greyscale and apply a binary threshold first.

    Args:
        frame: The input frame.

    Returns:
        Two by two tiling of original frame, greyscale frame,
        thresholded frame, and ellipsoid frame.s
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh_bin = cv2.threshold(grey, int(threshold * 255), 255, cv2.THRESH_BINARY_INV)[1]
    thresh = cv2.cvtColor(thresh_bin, cv2.COLOR_GRAY2BGR)

    # Calculate centroid and ellipsoid
    centroid, ellipse, _ = centroid_ellipsoid(thresh_bin)

    # Draw the ellipsoid on the original frame
    ellipse_thresh = cv2.circle(thresh, (int(centroid[0]), int(centroid[1])), 5, (255, 0, 0), -1)
    ellipse_points = np.array(ellipse.T, dtype=np.int32)
    ellipse_thresh = cv2.polylines(ellipse_thresh, [ellipse_points], isClosed=True, color=(255, 0, 0), thickness=2)

    return cv2.vconcat([
        cv2.hconcat([cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), frame]),
        cv2.hconcat([ellipse_thresh, cv2.cvtColor(thresh_bin, cv2.COLOR_GRAY2BGR)]),
    ])


def convolve(frame: cv2.Mat, kernel: np.ndarray = None) -> cv2.Mat:
    """
    Convolve the frame with a given kernel.

    Args:
        frame: The input frame.
        kernel: The kernel to convolve with.

    Returns:
        The convolved frame.
    """
    if kernel is None:
        kernel = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
    return cv2.filter2D(frame, -1, kernel)


def convolve_edges(frame: cv2.Mat) -> cv2.Mat:
    """
    Convolve the frame with horizontal, vertical, and both convolutions.

    Args:
        frame: The input frame.

    Returns:
        The convolved frame.
    """
    h_edge = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    v_edge = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    all_kernel = np.hstack([h_edge, v_edge])

    h_edge_img = cv2.filter2D(frame, -1, h_edge)
    v_edge_img = cv2.filter2D(frame, -1, v_edge)
    all_kernel_img = cv2.filter2D(frame, -1, all_kernel)

    return cv2.vconcat([
        cv2.hconcat([all_kernel_img, frame]),
        cv2.hconcat([h_edge_img, v_edge_img]),
    ])


def convolve_edges_advanced(frame: cv2.Mat) -> cv2.Mat:
    """
    Convolve the frame with advanced kernels.
    We use the
    - Sobel kernel (both horizontal and vertical): edge detection using first derivative
    - Laplacian kernel: edge detection using second derivative
    - Scharr kernel (both horizontal and vertical): better edge detection than Sobel and Laplacian

    Args:
        frame: The input frame.

    Returns:
        The convolved frame.
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    scharr_y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])

    sobel = np.hstack([sobel_x, sobel_y])
    scharr = np.hstack([scharr_x, scharr_y])
    
    sobel_img = cv2.filter2D(frame, -1, sobel)
    laplacian_img = cv2.filter2D(frame, -1, laplacian)
    scharr_img = cv2.filter2D(frame, -1, scharr)

    return cv2.vconcat([
        cv2.hconcat([sobel_img, frame]),
        cv2.hconcat([scharr_img, laplacian_img]),
    ])


_previous_frame = None

def _apply_motion_blur(frame: cv2.Mat) -> cv2.Mat:
    global _previous_frame
    alpha = 0.5

    if _previous_frame is None:
        _previous_frame = frame
        return frame
        
    new_frame = cv2.addWeighted(frame, alpha, _previous_frame, 1-alpha, 0)
    _previous_frame = frame
    return new_frame


def convolve_specialised(frame: cv2.Mat) -> cv2.Mat:
    """
    Convolve the frame with specialised kernels.
    Specifically,
    - Gabor kernel: used for texture analysis
    - Emboss kernel: used for edge detection with a 3D effect
    - Motion blur kernel: used for simulating motion blur

    Args:
        frame: The input frame.

    Returns:
        The convolved frame.
    """
    gabor = cv2.getGaborKernel(
        ksize = (21, 21),
        sigma = 2.0,
        theta = np.pi / 2,
        lambd = 12.0,
        gamma = 1.0,
        psi = np.pi / 2
    )
    emboss = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ])

    gabor_img = cv2.filter2D(frame, -1, gabor)
    emboss_img = cv2.filter2D(frame, -1, emboss)
    motion_blur_img = _apply_motion_blur(frame)

    return cv2.vconcat([
        cv2.hconcat([gabor_img, frame]),
        cv2.hconcat([motion_blur_img, emboss_img]),
    ])


def contour(frame: cv2.Mat) -> cv2.Mat:
    """
    Find contours in the frame.

    Args:
        frame: The input frame.

    Returns:
        The frame with contours drawn on it.
    """
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame
    contour_frame = frame.copy()
    cv2.drawContours(contour_frame, contours, -1, (255, 0, 0), 3)

    return cv2.vconcat([
        cv2.hconcat([cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), frame]),
        cv2.hconcat([contour_frame, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)]),
    ])


def _identity_method(frame: cv2.Mat) -> cv2.Mat:
    return cv2.vconcat([
        cv2.hconcat([frame, frame]),
        cv2.hconcat([frame, frame])
    ])  


def combination(frame: cv2.Mat, methods = None) -> cv2.Mat:
    """
    Apply all segmentation methods to the frame.
    Assumes each method returns a 2x2 grid of images.
    
    Args:
        frame: The input frame.
        methods: The list of methods to apply. If None, apply all methods.
            Default is None.

    Returns:
        Two by two tiling of original frame, greyscale frame,
        thresholded frame, and ellipsoid frame.
    """
    # selecting all methods is very slow performance
    methods_list = [
        thresholding,
        estimating_thresholding,
        intensity_rg_by,
        moments,
        convolve_edges,
        convolve_edges_advanced,
        convolve_specialised,
        contour
    ]

    if methods is None:
        methods = methods_list
    
    if len(methods) % 2 == 1:
        # add a dummy method to make it even
        methods.append(_identity_method)

    imgs = [method(frame) for method in methods][::-1] # reverse list to preserve specified order
    horizontal_concat = [cv2.hconcat([imgs[i], imgs[i+1]]) for i in range(0, len(imgs)-1, 2)]
    vertical_concat = cv2.vconcat(horizontal_concat[::-1])
    
    return vertical_concat
