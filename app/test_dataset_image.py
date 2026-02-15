import cv2
import numpy as np
from keras.models import load_model
from labels import labels

# Load model
model = load_model("traffic_sign_model.h5")

# CHANGE THIS PATH to one real .ppm image from dataset
image_path = "Dataset/Images/00000/00000_00000.ppm"

img = cv2.imread(image_path)

if img is None:
    print("Failed to load image")
    exit()

# Preprocess EXACTLY like training
img = cv2.resize(img, (32, 32))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img, verbose=0)
class_id = np.argmax(prediction)
confidence = np.max(prediction)

print("Predicted class_id:", class_id)
print("Predicted label:", labels[class_id])
print("Confidence:", round(confidence * 100, 2), "%")
