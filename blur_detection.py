import cv2
import numpy as np


def calculate_blur_score(image_path):
    """
    Calculate blur score using Variance of Laplacian.
    Returns blur score.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = laplacian.var()

    return blur_score


def check_blur(image_path, threshold=100):
    """
    Returns blur score and status (BLURRY/SHARP)
    """
    blur_score = calculate_blur_score(image_path)

    if blur_score < threshold:
        status = "BLURRY"
    else:
        status = "SHARP"

    return blur_score, status


# Example usage (for testing only)
if __name__ == "__main__":

    # 🔹 Give image path directly here
    image_path = "C:/Users/TEJASRI/Desktop/medical_image_validator/datasets/COVID/COVID_2.png"



    blur_score, status = check_blur(image_path)

    print("Image Path:", image_path)
    print(f"Blur Score: {blur_score:.2f}")
    print("Status:", status)
