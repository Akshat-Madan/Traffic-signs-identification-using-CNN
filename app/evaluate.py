import numpy as np
import os
from keras.models import load_model
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split


DATASET_PATH = "Dataset/Images"
IMG_SIZE = 32
NUM_CLASSES = 43
MODEL_PATH = "traffic_sign_model.h5"

def load_dataset(dataset_path):
    x = []
    y = []

    for class_id in range(NUM_CLASSES):
        class_path = os.path.join(dataset_path, str(class_id))
        if not os.path.isdir(class_path):
            continue

    
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            image = imread(img_path)
            image = resize(image, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
            image = image.astype(np.float32)

            X.append(image)
            y.append(class_id)

    X = np.array(X)
    y = np.array(y)

    # normalize
    X /= 255.0

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