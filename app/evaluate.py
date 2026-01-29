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