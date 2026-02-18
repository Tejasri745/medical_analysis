import cv2
import numpy as np
import joblib

# ==========================
# SETTINGS
# ==========================
MODEL_PATH = "medical_model.pkl"
IMAGE_SIZE = 128

# ==========================
# LOAD MODEL
# ==========================
model = joblib.load(MODEL_PATH)

# ==========================
# LABEL MAPPING
# ==========================
label_map = {
    0: "COVID",
    1: "NORMAL",
    2: "PNEUMONIA"
}

# ==========================
# FEATURE EXTRACTION
# ==========================
def extract_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found!")

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image.flatten().reshape(1, -1)

# ==========================
# PREDICTION FUNCTION
# ==========================
def predict_image(image_path):
    features = extract_features(image_path)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    confidence = round(np.max(probabilities) * 100, 2)

    print("\n----- PREDICTION REPORT -----")
    print("Predicted Class:", label_map[prediction])
    print("Confidence:", confidence, "%")

    if confidence > 90:
        print("Model Confidence Level: HIGH")
    elif confidence > 70:
        print("Model Confidence Level: MEDIUM")
    else:
        print("Model Confidence Level: LOW")

# ==========================
# TEST IMAGE PATH
# ==========================
image_path = r"C:\Users\TEJASRI\Desktop\medical_image_validator\datasets\COVID\COVID_5.png"

predict_image(image_path)
