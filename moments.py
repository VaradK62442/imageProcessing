"""
Given a binary pixel value for each pixel in an image, this program will calculate several moments in that image.
"""

import numpy as np
import matplotlib.pyplot as plt


def add_blob(image, x_left, x_right, y_left, y_right) -> np.array:
    image[x_left: x_right, y_left: y_right] = 1
    return image


def generate_image(width: int, height: int) -> np.array:
    arr = np.array([[0 for _ in range(width)] for _ in range(height)])

    arr = add_blob(arr, height // 4, int(3.5 * height // 4), width // 4, 3 * width // 4)
    arr = add_blob(arr, height // 4 - height // 5, height // 2 + height // 5, width // 4 - width // 10, width // 2)
    arr = add_blob(arr, 5 * height // 6, 11 * height // 12, width // 6, width // 3)

    return arr.T


def calculate_moment(image, p, q):
    x_id, y_id = np.nonzero(image)
    return np.sum((x_id ** p) * (y_id ** q))


def calculate_centroid(image):
    return calculate_moment(image, 1, 0) / calculate_moment(image, 0, 0), calculate_moment(image, 0, 1) / calculate_moment(image, 0, 0)


def calculate_feature_vector(image, p, q, mean_x = None, mean_y = None):
    if mean_x is None:
        mean_x = calculate_moment(image, 1, 0) / calculate_moment(image, 0, 0)

    if mean_y is None:
        mean_y = calculate_moment(image, 0, 1) / calculate_moment(image, 0, 0)

    x_id, y_id = np.nonzero(image)
    return np.sum((x_id - mean_x)**p * (y_id - mean_y)**q)


def calculate_central_moment_cov(image):
    return np.array([
        [calculate_feature_vector(image, 2, 0), calculate_feature_vector(image, 1, 1)], 
        [calculate_feature_vector(image, 1, 1), calculate_feature_vector(image, 0, 2)]
    ]) / calculate_moment(image, 0, 0)


def calculate_mu_prime(image, p, q, mean_x = None, mean_y = None):
    if mean_x is None:
        mean_x = calculate_moment(image, 1, 0) / calculate_moment(image, 0, 0)

    if mean_y is None:
        mean_y = calculate_moment(image, 0, 1) / calculate_moment(image, 0, 0)

    mu_00 = calculate_moment(image, 0, 0)
    mu_pq = calculate_moment(image, p, q)

    return mu_pq / mu_00


def calculate_orientation(image, mean_x = None, mean_y = None):
    if mean_x is None:
        mean_x = calculate_moment(image, 1, 0) / calculate_moment(image, 0, 0)

    if mean_y is None:
        mean_y = calculate_moment(image, 0, 1) / calculate_moment(image, 0, 0)

    mu_prime_11 = calculate_mu_prime(image, 1, 1, mean_x, mean_y)
    mu_prime_20 = calculate_mu_prime(image, 2, 0, mean_x, mean_y)
    mu_prime_02 = calculate_mu_prime(image, 0, 2, mean_x, mean_y)

    try:
        return 0.5 * np.arctan(2 * mu_prime_11 / (mu_prime_20 - mu_prime_02))
    except ZeroDivisionError:
        return np.inf


def calculate_elongatedness(image):
    mu_prime_20 = calculate_mu_prime(image, 2, 0)
    mu_prime_02 = calculate_mu_prime(image, 0, 2)
    mu_prime_11 = calculate_mu_prime(image, 1, 1)

    lambda_1, lambda_2 = [
        ((mu_prime_20 + mu_prime_02) + np.sqrt(4 * mu_prime_11**2 + (mu_prime_20 - mu_prime_02)**2)) / 2,
        ((mu_prime_20 + mu_prime_02) - np.sqrt(4 * mu_prime_11**2 + (mu_prime_20 - mu_prime_02)**2)) / 2
    ]

    return np.sqrt(1 - lambda_2 / lambda_1)


def calculate_rectangularity(image):
    w, h = image.shape
    return calculate_moment(image, 0, 0) / (w * h)


def calculate_perimeter(image):
    perimeter = 0
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y] == 1:
                if x == 0 or x == image.shape[0] - 1 or y == 0 or y == image.shape[1] - 1:
                    perimeter += 1
                elif image[x - 1][y] == 0 or image[x + 1][y] == 0 or image[x][y - 1] == 0 or image[x][y + 1] == 0:
                    perimeter += 1

    return perimeter


def calculate_circularity(image):
    perimeter = calculate_perimeter(image)

    return 4 * np.pi * calculate_moment(image, 0, 0) / (perimeter**2)


def plot_ellipse(ellipsoid_cov, centroid):
    eigvals, eigvecs = np.linalg.eigh(ellipsoid_cov)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    
    t = np.linspace(0, 2*np.pi, 1000)
    
    ellipse = np.array([np.cos(t), np.sin(t)])
    ellipse = 2 * np.sqrt(eigvals)[:, np.newaxis] * ellipse
    ellipse = eigvecs @ ellipse

    ellipse[0] += centroid[0]
    ellipse[1] += centroid[1]

    return ellipse


def centroid_ellipsoid(image):
    centroid = calculate_centroid(image)
    ellipsoid = calculate_central_moment_cov(image)

    ellipse = plot_ellipse(ellipsoid, centroid)

    return centroid, ellipse, ellipsoid
    

def main(image):
    # Calculate moments
    area = calculate_moment(image, 0, 0)
    centroid, ellipse, ellipsoid = centroid_ellipsoid(image)

    orientation = calculate_orientation(image)

    elongatedness = calculate_elongatedness(image)
    rectangularity = calculate_rectangularity(image)
    circularity = calculate_circularity(image)

    # print results
    print(f"Area: {area}")
    print(f"Centroid: {centroid}")
    print(f"Ellipsoid cov:\n{ellipsoid}")
    print(f"Orientation: {orientation}")
    print(f"Elongatedness: {elongatedness}")
    print(f"Rectangularity: {rectangularity}")
    print(f"Circularity: {circularity}")

    # visualise image
    plt.imshow(image.T, cmap='gray')
    plt.plot(centroid[0], centroid[1], 'rx', label='Centroid')
    plt.plot(ellipse[0], ellipse[1], 'b', label='Ellipsoid')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    width, height = 1000, 1000
    image = generate_image(width, height)

    main(image)