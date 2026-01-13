import os
import cv2
import numpy as np

def load_data(data_dir, image_size=(32,32)):
    images = []
    labels = []

    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        label = int(class_name)
        
        for file_name in os.listdir(class_path):
            if not file_name.lower().endswith(".ppm"):
                continue
        
            img_path = os.path.join(class_path, file_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.resize(image, image_size)
            images.append(image)
            labels.append(label)
    X = np.array(images, dtype=np.uint8)
    y = np.array(labels, dtype=np.int32)

    return X, y