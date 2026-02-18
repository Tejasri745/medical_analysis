import cv2
import numpy as np

# ----------------------------
# Image Path (Direct)
# ----------------------------
image_path = r"C:\Users\TEJASRI\Desktop\medical_image_validator\datasets\COVID\COVID_3.png"

# ----------------------------
# Load Image
# ----------------------------
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
    exit()

# ----------------------------
# Resize Image (For CNN Models)
# ----------------------------
image_resized = cv2.resize(image, (224, 224))

# ----------------------------
# Normalize Image (0 to 1)
# ----------------------------
image_normalized = image_resized / 255.0

# ----------------------------
# Convert to Grayscale
# ----------------------------
gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

# ----------------------------
# Extract Histogram Features
# ----------------------------
histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
histogram = histogram.flatten()

# ----------------------------
# Flatten Image (For ML Models)
# ----------------------------
flatten_features = image_normalized.flatten()

# ----------------------------
# Print Feature Info
# ----------------------------
print("----- FEATURE EXTRACTION REPORT -----")
print("Image Shape:", image_resized.shape)
print("Histogram Feature Length:", len(histogram))
print("Flatten Feature Length:", len(flatten_features))
