import cv2
import numpy as np
from keras.models import load_model
from labels import labels

model = load_model("traffic_sign_model.h5")

image_path = "test.jpg"

img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found.")
    exit()

img_resized = cv2.resize(img, (32, 32))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_normalized = img_rgb.astype("float32") / 255.0
img_input = np.expand_dims(img_normalized, axis=0)

prediction = model.predict(img_input, verbose=0)
class_id = np.argmax(prediction)
confidence = np.max(prediction)

label_name = labels[class_id]

print("Predicted Sign:", label_name)
print("Confidence:", round(confidence * 100, 2), "%")
