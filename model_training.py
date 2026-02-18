import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ===============================
# SETTINGS
# ===============================
DATASET_PATH = r"C:\Users\TEJASRI\Desktop\medical_image_validator\datasets"
IMAGE_SIZE = 128   # Smaller size for faster training

# ===============================
# FEATURE EXTRACTION FUNCTION
# ===============================
def extract_features(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        return None
    
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Flatten image
    features = image.flatten()
    
    return features

# ===============================
# LOAD DATASET
# ===============================
X = []
y = []

labels = {
    "COVID": 0,
    "NORMAL": 1,
    "PNEUMONIA": 2
}

print("Loading dataset...")

for folder_name in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder_name)
    
    if folder_name in labels:
        print(f"Processing {folder_name}...")
        
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            
            features = extract_features(image_path)
            
            if features is not None:
                X.append(features)
                y.append(labels[folder_name])

print("Dataset loaded successfully!")

X = np.array(X)
y = np.array(y)

print("Total Samples:", len(X))
print("Feature Length:", X.shape[1])

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# TRAIN MODEL
# ===============================
print("Training Random Forest Model...")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===============================
# EVALUATE
# ===============================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n----- TRAINING REPORT -----")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(model, "medical_model.pkl")

print("\nModel saved as medical_model.pkl")
