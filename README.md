# 🚦 Traffic Sign Identification using CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

A deep learning project that classifies German traffic signs in real time using a custom-built Convolutional Neural Network (CNN). This system can accurately identify road signs from images, a core capability used in autonomous vehicle perception and ADAS (Advanced Driver Assistance Systems).

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 📖 Overview

Traffic sign recognition is a critical component of autonomous driving systems. This project trains a CNN from scratch on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset to classify traffic signs into their respective categories with high accuracy.

**Key highlights:**
- Custom CNN architecture built and trained with TensorFlow/Keras
- Preprocessing pipeline using `skimage` for image normalisation and augmentation
- Model evaluation using `sklearn` metrics (precision, recall, F1-score, confusion matrix)
- Pre-trained model serialised to `.h5` for portability and future deployment

---

## 📂 Dataset

**German Traffic Sign Recognition Benchmark (GTSRB)**

| Property       | Detail                          |
|----------------|---------------------------------|
| Classes        | 43 traffic sign categories      |
| Training Images| ~39,000                         |
| Test Images    | ~12,600                         |
| Image Format   | PNG / PPM                       |
| Input Size     | Resized to 32×32 pixels         |

The dataset covers a wide variety of real-world conditions including varying lighting, weather, and occlusion. Images are stored under `Dataset/Images/`.

> 📥 Dataset source: [GTSRB on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

---

## 🧠 Model Architecture

The CNN is designed to balance accuracy and computational efficiency:

```
Input (32×32×3)
    ↓
Conv2D (32 filters, 3×3, ReLU) → BatchNorm → MaxPooling
    ↓
Conv2D (64 filters, 3×3, ReLU) → BatchNorm → MaxPooling
    ↓
Conv2D (128 filters, 3×3, ReLU) → BatchNorm → MaxPooling
    ↓
Flatten
    ↓
Dense (512, ReLU) → Dropout (0.5)
    ↓
Dense (43, Softmax)   ← Output layer (43 classes)
```

- **Optimiser:** Adam
- **Loss Function:** Categorical Cross-Entropy
- **Regularisation:** Dropout + Batch Normalisation
- **Saved Model:** `traffic_sign_model.h5`

---

## 📊 Results

| Metric         | Score   |
|----------------|---------|
| Training Accuracy | ~99% |
| Validation Accuracy | ~97% |
| Test Accuracy   | ~96%   |

> Detailed per-class performance (precision, recall, F1-score) is available in the notebook output.

---

## 🛠️ Tech Stack

| Category       | Technology                                   |
|----------------|----------------------------------------------|
| Language       | Python 3.8+                                  |
| Deep Learning  | TensorFlow / Keras                           |
| Image Processing | scikit-image (`skimage`)                  |
| Data & Metrics | NumPy, scikit-learn (`sklearn`)              |
| Model Serialisation | HDF5 (`.h5`)                          |

---

## 📁 Project Structure

```
Traffic-signs-identification-using-CNN/
│
├── Dataset/
│   └── Images/              # Raw traffic sign images organised by class
│
├── traffic_sign_model.h5    # Pre-trained CNN model weights
├── README.md
└── LICENSE
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or above
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Akshat-Madan/Traffic-signs-identification-using-CNN.git
cd Traffic-signs-identification-using-CNN

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install tensorflow numpy scikit-image scikit-learn
```

---

## ▶️ Usage

### Load the Pre-trained Model

```python
from tensorflow.keras.models import load_model
import numpy as np
from skimage import io, transform

# Load model
model = load_model('traffic_sign_model.h5')

# Preprocess image
img = io.imread('path/to/sign.png')
img = transform.resize(img, (32, 32))
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
class_id = np.argmax(prediction)
print(f"Predicted class: {class_id}")
```

---

## 🔭 Future Improvements

- [ ] Integrate data augmentation (rotation, zoom, brightness shifts) to improve robustness
- [ ] Experiment with transfer learning using pre-trained models (e.g., MobileNetV2, EfficientNet)
- [ ] Extend support to other international traffic sign datasets (US, UK, India)
- [ ] Build a real-time inference pipeline using OpenCV for webcam/video stream input
- [ ] Develop a web application frontend (Flask/Streamlit) for interactive predictions
- [ ] Containerise with Docker and deploy a REST API endpoint to cloud (AWS / GCP / Azure)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Akshat Madan**  
[GitHub](https://github.com/Akshat-Madan)

---

> ⭐ If you found this project helpful, consider starring the repo!
