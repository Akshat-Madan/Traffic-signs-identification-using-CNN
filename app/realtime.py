import cv2
import numpy as np
from keras.models import load_model
from labels import labels

# ---------------- CONFIG ----------------
MODEL_PATH = "traffic_sign_model.h5"
IMG_SIZE = 32
CONFIDENCE_THRESHOLD = 0.5  # 50%
# ---------------------------------------


print("[INFO] Loading model...")
model = load_model(MODEL_PATH)

print("[INFO] Starting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Copy frame for display
    display_frame = frame.copy()

    # Preprocess frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    preds = model.predict(img, verbose=0)[0]
    class_id = np.argmax(preds)
    confidence = preds[class_id]

    # Display result
    if confidence >= CONFIDENCE_THRESHOLD:
        label = labels[class_id]
        text = f"{label}: {confidence * 100:.2f}%"
        color = (0, 255, 0)
    else:
        text = "Unknown sign"
        color = (0, 0, 255)

    cv2.putText(
        display_frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Traffic Sign Recognition", display_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
