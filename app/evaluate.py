import os
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# -----------------------------
# Paths (bulletproof)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "Dataset", "Images")
MODEL_PATH = os.path.join(BASE_DIR, "..", "traffic_sign_model.h5")

IMG_SIZE = 32
NUM_CLASSES = 43


# -----------------------------
# Dataset Loader
# -----------------------------
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

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


# -----------------------------
# Main Evaluation
# -----------------------------
def main():
    print("[INFO] Loading dataset...")
    X, y = load_dataset(DATASET_PATH)

    print("[INFO] Normalizing images...")
    X = X.astype("float32") / 255.0

    print("[INFO] Splitting dataset...")
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔥 IMPORTANT: convert to one-hot (matches training)
    y_test = to_categorical(y_test, NUM_CLASSES)

    print("[INFO] Loading trained model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = load_model(MODEL_PATH)

    print("[INFO] Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

    print("\n==============================")
    print(f"Test Loss     : {loss:.4f}")
    print(f"Test Accuracy : {accuracy * 100:.2f}%")
    print("==============================")


if __name__ == "__main__":
    main()
