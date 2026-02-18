import cv2
import numpy as np


def analyze_exposure(image_path):
    """
    Analyze brightness and exposure of an image.
    Returns:
        mean_intensity,
        dark_pixel_ratio,
        bright_pixel_ratio,
        exposure_status
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mean brightness
    mean_intensity = np.mean(gray)

    # Count very dark pixels (0-50)
    dark_pixels = np.sum(gray < 50)

    # Count very bright pixels (240-255)
    bright_pixels = np.sum(gray > 240)

    total_pixels = gray.size

    dark_ratio = dark_pixels / total_pixels
    bright_ratio = bright_pixels / total_pixels

    # Decision Logic
    if mean_intensity < 60:
        exposure_status = "UNDEREXPOSED"
    elif mean_intensity > 190:
        exposure_status = "OVEREXPOSED"
    elif bright_ratio > 0.15:
        exposure_status = "OVEREXPOSED"
    elif dark_ratio > 0.40:
        exposure_status = "UNDEREXPOSED"
    else:
        exposure_status = "NORMAL"

    return mean_intensity, dark_ratio, bright_ratio, exposure_status


if __name__ == "__main__":

    image_path = "C:/Users/TEJASRI/Desktop/medical_image_validator/datasets/COVID/COVID_2.png"

    mean_intensity, dark_ratio, bright_ratio, status = analyze_exposure(image_path)

    print("Mean Intensity:", round(mean_intensity, 2))
    print("Dark Pixel Ratio:", round(dark_ratio, 3))
    print("Bright Pixel Ratio:", round(bright_ratio, 3))
    print("Exposure Status:", status)
