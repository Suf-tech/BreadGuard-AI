# BreadGuard AI — AI-Based Bread Spoilage Detection Using Deep Learning

## Project Information

**Student:** Hafiz Muhammad Sufiyan Khan  
**Class:** BSCS7A  
**Department:** Computer Science  
**Instructor:** Dr. Munir

## 1. Project Overview

BreadGuard AI is a deep learning-based web application that classifies uploaded bread images into two categories:

- **Fresh (0):** No visible spoilage detected
- **Spoiled (1):** Visible spoilage patterns detected

The project compares a custom Convolutional Neural Network (CNN) trained from scratch with a transfer learning model based on MobileNetV2.

Users can upload a bread image, select a model, and receive a predicted class with confidence scores.

**Important:** This application is designed for visible spoilage screening only. It cannot guarantee food safety or detect invisible contamination.

## 2. Problem Statement

Bread is vulnerable to visible spoilage, including mold growth. Manual visual inspection can be inconsistent, especially when spoilage patterns are subtle.

This project investigates whether deep learning and computer vision can assist in identifying visible spoilage patterns from bread images.

The goal is to build an accessible image-classification system and compare the performance of a custom CNN with a pretrained MobileNetV2 architecture.

## 3. Dataset

The dataset was curated from publicly accessible online image-search results and manually reviewed.

After removing exact duplicate files using SHA-256 hashing, the dataset contained **472 unique images**.

| Class | Images |
|---|---:|
| Fresh | 374 |
| Spoiled | 98 |
| **Total** | **472** |

### Dataset Split

| Split | Fresh | Spoiled | Total |
|---|---:|---:|---:|
| Training | 261 | 68 | 329 |
| Validation | 56 | 14 | 70 |
| Testing | 57 | 16 | 73 |

Images are resized to **224 × 224 pixels** with three RGB channels.

**Dataset download:** To be added after publishing the dataset.

**Dataset note:** Images were collected from online sources, not photographed by the project author. Redistribution permissions should be verified before publicly sharing the image files.

## 4. Deep Learning Models

### Model A — Custom CNN From Scratch (V4)

A custom convolutional neural network trained on the bread dataset.

The architecture includes convolutional layers, max-pooling, image augmentation, dense layers, dropout, and a sigmoid output for binary classification.

**Saved model:** `models/bread_scratch_cnn.keras`

### Model B — MobileNetV2 Transfer Learning

A MobileNetV2 model initialized with ImageNet pretrained weights and used as a frozen feature extractor, followed by a custom classification head.

**Saved model:** `models/best_mobilenetv2.keras`

Both saved models include their required image preprocessing internally.

## 5. Model Evaluation

| Metric | Custom CNN V4 | MobileNetV2 |
|---|---:|---:|
| Training accuracy | 92.40% | 97.57% |
| Validation accuracy | 90.00% | 97.14% |
| Test accuracy | 89.04% | 94.52% |
| Spoiled recall | 75.00% | 81.25% |

### Test Confusion Matrices

**Custom CNN V4**

| Actual / Predicted | Fresh | Spoiled |
|---|---:|---:|
| Fresh | 53 | 4 |
| Spoiled | 4 | 12 |

**MobileNetV2**

| Actual / Predicted | Fresh | Spoiled |
|---|---:|---:|
| Fresh | 56 | 1 |
| Spoiled | 3 | 13 |

MobileNetV2 achieved the higher test accuracy.

**Evaluation limitation:** The held-out test set contains only 73 images, including 16 Spoiled examples. Results may vary on new images. Exact duplicates were removed, but visually similar online images may still exist across splits.

## 6. Technology Stack

- Python 3.12
- TensorFlow / Keras
- NumPy
- Pillow
- FastAPI
- Uvicorn
- HTML, CSS and JavaScript
- MobileNetV2 transfer learning

## 7. Project Structure

```text
BreadDetectionProject/
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── assets/
├── backend/
│   ├── main.py
│   └── requirements.txt
├── models/
│   ├── bread_scratch_cnn.keras
│   └── best_mobilenetv2.keras
├── dataset_raw/
├── dataset_split/
├── results/
├── BreadDetectionProject.ipynb
└── README.md
```

## 8. How to Run Locally

### Requirements

Install Python 3.12.

Open a terminal in the project root directory.

### Install Backend Dependencies

```bash
py -3.12 -m pip install -r backend/requirements.txt
```

### Start the Backend

```bash
py -3.12 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Backend API documentation:

`http://127.0.0.1:8000/docs`

### Start the Frontend

Open another terminal:

```bash
cd frontend
py -3.12 -m http.server 5500
```

Open the application:

`http://127.0.0.1:5500`

Both servers must remain running during local testing.

## 9. Prediction Workflow

1. Upload a JPG, PNG, or WEBP bread image.
2. Select MobileNetV2 or Custom CNN.
3. The image is converted to RGB and resized to 224 × 224 pixels.
4. The selected trained model generates a sigmoid probability.
5. The application displays Fresh or Spoiled with confidence.

Class mapping:

- `0 = Fresh`
- `1 = Spoiled`

The saved models perform their own preprocessing. Input images should not be normalized again before inference.

## 10. Model Weights and Downloads

The trained model files are:

- `models/best_mobilenetv2.keras`
- `models/bread_scratch_cnn.keras`

**Public download links:** To be added after hosting the model files.

These are saved Keras models containing the trained network parameters required for inference.

## 11. Project Resources

- **GitHub Repository:** To be added
- **Live Vercel Website:** To be added
- **Dataset Download:** To be added
- **MobileNetV2 Model Download:** To be added
- **Custom CNN Model Download:** To be added
- **Project Report:** To be added
- **Presentation Slides:** To be added
- **YouTube Presentation:** To be added
- **LinkedIn Project Post:** To be added

## 12. Responsible Use

BreadGuard AI is an academic computer vision project.

A Fresh prediction does not mean the bread is safe to consume. The system cannot detect all contaminants or invisible microbial hazards.

Predictions should not replace appropriate food-safety practices.

## Author

**Hafiz Muhammad Sufiyan Khan**  
BSCS7A — Department of Computer Science  
Instructor: Dr. Munir