import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# ==========================
# SETTINGS
# ==========================
DATASET_PATH = r"C:\Users\TEJASRI\Desktop\medical_image_validator\datasets"
IMAGE_SIZE = 128

label_map = {
    "COVID": 0,
    "NORMAL": 1,
    "PNEUMONIA": 2
}

# ==========================
# LOAD DATA
# ==========================
X = []
y = []

print("Loading dataset...")

for folder in os.listdir(DATASET_PATH):
    if folder in label_map:
        folder_path = os.path.join(DATASET_PATH, folder)

        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            image = cv2.imread(image_path)

            if image is not None:
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                image = image / 255.0
                X.append(image)
                y.append(label_map[folder])

X = np.array(X)
y = np.array(y)

y = to_categorical(y, num_classes=3)

print("Total Samples:", len(X))

# ==========================
# SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# BUILD CNN
# ==========================
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ==========================
# TRAIN
# ==========================
model.fit(X_train, y_train, epochs=10, batch_size=32)

# ==========================
# EVALUATE
# ==========================
loss, accuracy = model.evaluate(X_test, y_test)

print("\nCNN Accuracy:", round(accuracy * 100, 2), "%")

model.save("cnn_medical_model.h5")
print("CNN model saved.")
