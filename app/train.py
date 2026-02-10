import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from model import build_model 
from Augmentation import (
    flip_extend,
    apply_rotation,
    apply_projection_transform
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


DATASET_PATH = os.path.join(BASE_DIR, "..", "Dataset", "Images")
IMG_SIZE = 32
NUM_CLASSES = 43
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.001

def load_dataset(dataset_path):
    images = []
    labels = []

    class_folders = sorted(os.listdir(dataset_path))

    for class_name in class_folders:
        class_path = os.path.join(dataset_path, class_name)

        if not os.path.isdir(class_path):
            continue

        label = int(class_name)  # "00023" → 23

        for img_name in os.listdir(class_path):
            if not img_name.endswith(".ppm"):
                continue

            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (32, 32))
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)



print("[INFO] Loading dataset...")
X, y = load_dataset(DATASET_PATH)

X = X/255.0

X_train, X_val , y_train, y_val = train_test_split(
    X, y , test_size=0.2, random_state=42, stratify=y
)

print("[INFO] Applying flip augmentation...")
X_train, y_train = flip_extend(X_train, y_train)


print("[INFO] Applying rotation...")
X_train = apply_projection_transform(X_train, intensity=0.3)

y_train = to_categorical(y_train, NUM_CLASSES)
y_val = to_categorical(y_val, NUM_CLASSES)

model = build_model()

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

checkpoint = ModelCheckpoint(
    "traffic_sign_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


print("[INFO] Training model...")
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint],
    shuffle=True
)

print("[INFO] Training complete. Best model saved as traffic_sign_model.h5")