import numpy as np
import os
from keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import cv2

# ---------------- CONFIG ----------------
DATASET_PATH = "Dataset/Images"
IMG_SIZE = 32
NUM_CLASSES = 43
MODEL_PATH = "traffic_sign_model.h5"
# ---------------------------------------

def load_dataset(path):
    X, y = [], []

    for folder in sorted(os.listdir(path)):
        folder_path = os.path.join(path, folder)

        if not os.path.isdir(folder_path):
            continue

        # Convert "00000" → 0
        class_id = int(folder)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            X.append(img)
            y.append(class_id)

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y)

    return X, y



def main():
    print("Loading dataset...")
    X, y = load_dataset(DATASET_PATH)

    # Same split logic as training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Loading trained model...")
    model = load_model(MODEL_PATH)

    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

    print("\n✅ Evaluation Results")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
