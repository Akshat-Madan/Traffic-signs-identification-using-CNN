import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from model import build_model

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

        label = int(class_name)

        for img_name in os.listdir(class_path):
            if not img_name.endswith(".ppm"):
                continue

            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


print("[INFO] Loading dataset...")
X, y = load_dataset(DATASET_PATH)

# Normalize
X = X.astype("float32") / 255.0

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# One-hot encoding
y_train = to_categorical(y_train, NUM_CLASSES)
y_val = to_categorical(y_val, NUM_CLASSES)

# Build model
model = build_model(input_shape=(32, 32, 3), num_classes=NUM_CLASSES)

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
