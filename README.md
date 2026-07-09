---
title: Diabetic Retinopathy Detection
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: streamlit
pinned: false
---

# Diabetic Retinopathy Detection

## Overview

Diabetic Retinopathy Detection is a deep learning-based web application that analyzes retinal fundus images and predicts whether the eye is affected by Diabetic Retinopathy or is Normal. This project aims to assist in the early detection of diabetic eye disease using image classification techniques.

This repository is updated with a clinical portal interface (with patient history, logins, and reporting) and is containerized for easy cloud deployment.

## Features

* Upload retinal fundus images
* Deep learning-based prediction
* User-friendly clinical portal web interface
* Doctor login authentication
* Patient scans history logging and reporting
* Automated disease detection
* Accuracy evaluation script
* Grad-CAM visualization support to highlight lesions

## Technologies Used

* Python 3.11.0
* TensorFlow / Keras
* Flask
* OpenCV
* NumPy
* HTML / CSS
* Docker & Gunicorn (for deployment)

## Python Version

⚠️ This project was developed and tested using **Python 3.11.0**.
Check your Python version before running the project:
```bash
python --version
```

## Project Structure

```text
Source Code/
│
├── app.py                  # Main Flask application
├── check_accuracy.py       # Accuracy evaluation script
├── model_binary.keras      # Trained deep learning model
├── requirements.txt        # Python package dependencies
├── Dockerfile              # Docker container deployment configuration
├── patients.json           # Local database for patient records
├── README.md               # Project documentation & HF Space configuration
├── static/                 # Static assets (uploads, Grad-CAM, sample images)
└── templates/              # HTML templates (index, login, predict)
```

## Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/1-vijayan/DR-Detection.git
cd DR-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Flask application:
```bash
python app.py
```
Open your browser and visit:
```text
http://127.0.0.1:5000
```

---

## Deployment & Hosting (Publishing)

This project contains configuration for Docker-based deployment. It is fully ready to be published and hosted.

### Run with Docker Locally
```bash
docker build -t dr-detection .
docker run -p 7860:7860 dr-detection
```

### Deploy to Hugging Face Spaces (Recommended Free Host)
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Set SDK to **Docker** (choose the **Blank** template).
3. Set Space Hardware to **CPU basic (Free, 16GB RAM)**.
4. Upload all project files. Hugging Face will automatically read the `Dockerfile` and publish your app online!

### Deploy to Render
1. Create a new **Web Service** on [Render.com](https://render.com/).
2. Connect your GitHub repository.
3. Select **Docker** as the runtime.
4. Use a paid plan (Starter tier or higher) to avoid Out Of Memory crashes.

---

## Model Information

The trained deep learning model is stored in `model_binary.keras` and classifies retinal images into:
* Non-Referable DR (Healthy/Mild)
* Referable DR (Moderate/Severe/Proliferative)

## Author

**Vijayan S**  
Final Year Project  

## License

This project is intended for educational and research purposes.
