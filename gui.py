import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os

# -------------------------
# Load trained model
# -------------------------
model_path = r"C:\Users\TEJASRI\Desktop\medical_image_validator\cnn_medical_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = tf.keras.models.load_model(model_path)
class_names = ["Normal", "COVID", "Pneumonia"]  # 3 classes

# -------------------------
# Image preprocessing
# -------------------------
def preprocess_image(img_path, target_size=(128, 128)):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img_array = np.expand_dims(img, axis=0)  # add batch dimension
    img_array = img_array / 255.0  # normalize
    return img_array

# -------------------------
# Predict function
# -------------------------
def predict_image(img_path):
    try:
        img_tensor = preprocess_image(img_path)
        preds = model.predict(img_tensor)[0]  # shape (3,)
        pred_class = np.argmax(preds)
        confidence = preds[pred_class] * 100

        # Prepare probabilities text
        prob_text = "\n".join([f"{name}: {preds[i]*100:.2f}%" 
                               for i, name in enumerate(class_names)])
        return class_names[pred_class], confidence, prob_text
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None, None, ""

# -------------------------
# GUI Functions
# -------------------------
def upload_and_predict():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )
    if file_path:
        # Display selected image
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        # Predict
        pred, conf, prob_text = predict_image(file_path)
        if pred:
            result_label.config(
                text=f"Predicted: {pred}\nConfidence: {conf:.2f}%\n\n{prob_text}"
            )

# -------------------------
# Build GUI
# -------------------------
root = tk.Tk()
root.title("COVID / Pneumonia X-ray Detection")
root.geometry("450x550")

title_label = tk.Label(root, text="COVID / Pneumonia Detection", font=("Helvetica", 16))
title_label.pack(pady=10)

upload_btn = tk.Button(root, text="Upload X-ray Image", command=upload_and_predict, font=("Helvetica", 12))
upload_btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 14))
result_label.pack(pady=10)

root.mainloop()
