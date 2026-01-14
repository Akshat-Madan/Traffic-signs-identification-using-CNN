import numpy as np
from skimage.transform import rotate, warp, ProjectiveTransform
import random


def flip_extend(X, y):
    self_flippable_horizontally = np.array([11,12,13,15,17,18,22,26,30,35])
    self_flippable_vertically = np.array([1,5,12,15,17])
    self_flippable_both = np.array([32,40])

    cross_flippable = np.array([
        [19,20],[20,19],
        [33,34],[34,33],
        [36,37],[37,36],
        [38,39],[39,38],
    ])

    num_classes = 43

    X_ext = []
    y_ext = []

    for c in range(num_classes):
        idx = (y == c)
        X_c = X[idx]

        # Original
        X_ext.append(X_c)
        y_ext.append(np.full(len(X_c), c))

        # Horizontal flip
        if c in self_flippable_horizontally:
            X_ext.append(X_c[:, :, ::-1, :])
            y_ext.append(np.full(len(X_c), c))

        # Cross flippable
        for pair in cross_flippable:
            if c == pair[0]:
                X_pair = X[y == pair[1]]
                X_ext.append(X_pair[:, :, ::-1, :])
                y_ext.append(np.full(len(X_pair), c))

        # Vertical flip
        if c in self_flippable_vertically:
            X_v = X_c[:, ::-1, :, :]
            X_ext.append(X_v)
            y_ext.append(np.full(len(X_v), c))

        # Both flips
        if c in self_flippable_both:
            X_b = X_c[:, ::-1, ::-1, :]
            X_ext.append(X_b)
            y_ext.append(np.full(len(X_b), c))

    X_ext = np.concatenate(X_ext, axis=0)
    y_ext = np.concatenate(y_ext, axis=0)

    return X_ext, y_ext


def apply_rotation(X, intensity=0.5):
    X_rot = X.copy()
    delta = 30 * intensity

    for i in range(X.shape[0]):
        angle = random.uniform(-delta, delta)
        X_rot[i] = rotate(X_rot[i], angle, mode='edge')

    return X_rot


def apply_projection_transform(X, intensity=0.5):
    X_proj = X.copy()
    image_size = X.shape[1]
    d = image_size * 0.3 * intensity

    for i in range(X.shape[0]):
        transform = ProjectiveTransform()
        transform.estimate(
            np.array([
                (random.uniform(-d, d), random.uniform(-d, d)),
                (random.uniform(-d, d), image_size - random.uniform(-d, d)),
                (image_size - random.uniform(-d, d), image_size - random.uniform(-d, d)),
                (image_size - random.uniform(-d, d), random.uniform(-d, d))
            ]),
            np.array([
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            ])
        )

        X_proj[i] = warp(
            X_proj[i],
            transform,
            output_shape=(image_size, image_size),
            mode='edge'
        )

    return X_proj
