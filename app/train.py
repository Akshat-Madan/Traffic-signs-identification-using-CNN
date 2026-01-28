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


DATASET_PATH = "Dataset/Images"
IMG_SIZE = 32
NUM_CLASSES = 43
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.001

def load_dataset()
