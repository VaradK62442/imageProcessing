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


def all_methods(frame: cv2.Mat) -> cv2.Mat:
    """
    Apply all segmentation methods to the frame.

    Args:
        frame: The input frame.
        kwargs: Additional arguments for each method.

    Returns:
        Two by two tiling of original frame, greyscale frame,
        thresholded frame, and ellipsoid frame.
    """
    thresholding_img = thresholding(frame)
    estimating_thresholding_img = estimating_thresholding(frame)
    intensity_rg_by_img = intensity_rg_by(frame)
    moments_img = moments(frame)

    return cv2.vconcat([
        cv2.hconcat([estimating_thresholding_img, thresholding_img]),
        cv2.hconcat([moments_img, intensity_rg_by_img]),
    ])