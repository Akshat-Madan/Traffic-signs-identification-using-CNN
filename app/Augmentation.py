import numpy as np
from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform
import random

def flip_extend(X, y):
    self_fippable_horizontally = np.array([11,12,13,15,17,18,22,26,30,35])
    self_flippable_vertically = np.array([1,5,12,15,17])
    self_flippable_both = np.array([32,40])
    cross_flippable = np.array([
        [19,20],
        [33,34],
        [36,37],
        [38,39],
        [20, 19], 
        [34, 33], 
        [37, 36], 
        [39, 38], 
    ])

    num_classes = 43

    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype = X.dtype)
    y_extended = np.empty([0], dtype = y.type)

    for c in range(num_classes):
        X_extended = np.append(X_extended, X[y == c], axis = 0)
        if c in self_fippable_horizontally:
            X_extended = np.append(X_extended, X[y == c][:, :, ::-1, :], axis = 0)
    

        if c in cross_flippable[:, 0]:
            flip_class = cross_flippable[cross_flippable[:, 0] == c][0][1]
            X_extended = np.append(X_extended, X[y == flip_class][:, :, ::-1, :], axis = 0)
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype = int))


        if c in self_flippable_vertically:
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, :, :], axis = 0)
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype = int))
        
        if c in self_flippable_both:
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, ::-1, :], axis = 0)
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype = int))
    return (X_extended, y_extended)

def rotate(X, intensity):
    for i in range(X.shape[0]):
        delta = 30 * intensity
        X[i] = rotate(X[i], random.uniform(-delta, delta), mod = 'edge')
    return X

def apply_projection_transform(X, intensity):
    image_size = X.shape[1]
    d = image_size * 0.3 * intensity
    for i in range(X.shape[0]):
        tl_top = random.uniform(-d, d)
        tl_left = random.uniform(-d, d)
        bl_bottom = random.uniform(-d, d)
        bl_left = random.uniform(-d, d)
        tr_top = random.uniform(-d, d)
        tr_right = random.uniform(-d, d)
        br_bottom = random.uniform(-d, d)
        br_right = random.uniform(-d, d)

        transform = ProjectiveTransform()
        transform.estimate(np.array((
                (tl_left, tl_top),
                (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size - br_bottom),
                (image_size - tr_right, tr_top)
            )), np.array((
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            )))
        X[i] = warp(X[i], transform, output_shape=(image_size, image_size), order = 1, mode = 'edge')
    return X