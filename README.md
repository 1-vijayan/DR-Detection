# Diabetic Retinopathy Detection

## Overview

Diabetic Retinopathy Detection is a deep learning-based web application that analyzes retinal fundus images and predicts whether the eye is affected by Diabetic Retinopathy or is Normal. This project aims to assist in the early detection of diabetic eye disease using image classification techniques.

## Features

* Upload retinal fundus images
* Deep learning-based prediction
* User-friendly web interface
* Automated disease detection
* Accuracy evaluation script
* Grad-CAM visualization support

## Technologies Used

* Python 3.11.0
* TensorFlow / Keras
* Flask
* OpenCV
* NumPy
* HTML
* CSS

## Python Version

⚠️ This project was developed and tested using **Python 3.11.0**.

Check your Python version before running the project:

```bash
python --version
```

Expected output:

```text
Python 3.11.0
```

Using a different Python version may lead to dependency compatibility issues.

## Project Structure

```text
Diabetic Retinopathy Detection/
│
├── app.py
├── check_accuracy.py
├── model_binary.keras
├── requirements.txt
├── static/
│   ├── uploads/
│   └── gradcam/
│
├── templates/
│   ├── index.html
│   └── predict.html
│
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/1-vijayan/DR-Detection.git
```

### 2. Navigate to the Project Directory

```bash
cd DR-Detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Flask application:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## How to Use

1. Launch the application.
2. Upload a retinal fundus image.
3. Click the Predict button.
4. View the prediction result.
5. Analyze the generated output and visualization.

## Model Information

The trained deep learning model is stored in:

```text
model_binary.keras
```

The model is used to classify retinal images into:

* Normal
* Diabetic Retinopathy

## Requirements

Install all required libraries using:

```bash
pip install -r requirements.txt
```

## Future Improvements

* Multi-class Diabetic Retinopathy classification
* Improved model accuracy
* Cloud deployment
* Mobile application integration
* Enhanced visualization techniques

## Author

**Vijayan S**

Final Year Project

## License

This project is intended for educational and research purposes.
