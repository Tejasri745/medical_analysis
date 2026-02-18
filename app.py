import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# -------------------------
# Load trained model
# -------------------------
model_path = r"C:\Users\TEJASRI\Desktop\medical_image_validator\cnn_medical_model.h5"
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

model = tf.keras.models.load_model(model_path)
class_names = ["Normal", "COVID", "Pneumonia"]  # 3 classes

# -------------------------
# Preprocess image
# -------------------------
def preprocess_image(img, target_size=(128, 128)):
    img = np.array(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(img, target_size)
    img_array = np.expand_dims(img, axis=0)  # add batch dimension
    img_array = img_array / 255.0  # normalize
    return img_array

# -------------------------
# Predict
# -------------------------
def predict(img):
    img_tensor = preprocess_image(img)
    preds = model.predict(img_tensor)[0]
    pred_class = np.argmax(preds)
    confidence = preds[pred_class] * 100

    prob_text = "\n".join([f"{name}: {preds[i]*100:.2f}%" 
                           for i, name in enumerate(class_names)])
    return class_names[pred_class], confidence, prob_text

# -------------------------
# Streamlit App Layout
# -------------------------
st.title("Medical X-ray Analysis System")
st.subheader("Detect COVID, Pneumonia, or Normal from X-ray images")

uploaded_file = st.file_uploader("Upload X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Predict"):
        pred, conf, prob_text = predict(image)
        st.success(f"Predicted Class: {pred}")
        st.info(f"Confidence: {conf:.2f}%")
        st.text("Class Probabilities:")
        st.text(prob_text)
