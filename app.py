import tensorflow as tf
from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = 'static/uploads'
GRADCAM_FOLDER = 'static/gradcam'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("model_binary.keras")

CLASS_NAMES = ['Non-Referable DR (Healthy/Mild)', 'Referable DR (Moderate/Severe/Proliferative)']

# ---------------- CHECK FILE ----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- PREPROCESS ----------------
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # CLAHE (same as training)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    return img

# ---------------- CONFIDENCE LOGIC ----------------
def analyze_prediction(preds):
    preds = preds[0]

    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))

    predicted_label = CLASS_NAMES[class_index]

    # 🔥 Confidence Handling
    if confidence < 0.6:
        final_label = "Uncertain Diagnosis"
        status = "warning"
        message = "⚠ Low confidence. Please upload a clearer image or consult a doctor."
    elif class_index == 1:
        final_label = predicted_label
        status = "danger"
        message = "⚠️ Model detected signs of Referable DR. Please consult an ophthalmologist."
    else:
        final_label = predicted_label
        status = "success"
        message = "✅ Model did not detect signs of Referable DR. Regular checkups are still recommended."

    # 🔥 All class probabilities
    probabilities = []
    for i in range(len(CLASS_NAMES)):
        probabilities.append({
            "label": CLASS_NAMES[i],
            "value": round(float(preds[i] * 100), 2)
        })

    return final_label, confidence, class_index, probabilities, status, message

# ---------------- GRAD-CAM ----------------
def generate_gradcam(image_path, class_index):
    # 1. Prepare image
    img_tensor = preprocess_image(image_path)

    # 2. Find the last 4D convolutional layer automatically
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            last_conv_layer_name = layer.name
            break

    # 3. Build Gradient Model
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 4. Record Gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_index]

    # 5. Calculate Weights (Global Average Pooling of Gradients)
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 6. Generate Weighted Heatmap
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 7. 🔥 SHARPENING & REFINEMENT
    # Apply ReLU (remove negative importance) and Normalize to 0-1 range
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Apply Power Transform (Heatmap^2) to tighten the focus on lesions
    # This removes the "blurry" low-importance areas
    heatmap = np.power(heatmap, 2) 

    # 8. Visual Processing & Overlay
    original = cv2.imread(image_path)
    original = cv2.resize(original, (224, 224))
    
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Combine: 0.6 Original + 0.4 Heatmap for better anatomical alignment
    superimposed = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

    # 9. Save and Return Path
    filename = os.path.basename(image_path)
    save_path = os.path.join(GRADCAM_FOLDER, filename)
    cv2.imwrite(save_path, superimposed)

    return save_path
# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message="No file uploaded")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message="No file selected")

    if file and allowed_file(file.filename):
        # 1. Secure and save the original file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 2. Run the AI Prediction
        img = preprocess_image(file_path)
        preds = model.predict(img)

        # 3. Analyze confidence and labels
        label, confidence, class_index, probabilities, status, message = analyze_prediction(preds)

        # 4. Generate Grad-CAM (This saves the heatmap into static/gradcam)
        # We need to capture the filename, not just the path
        generate_gradcam(file_path, class_index)
        
        # 5. Pass ONLY the filename to the template
        # The template uses url_for('static', filename='uploads/' + image)
        return render_template(
            'predict.html',
            diagnosis=label,
            confidence=round(confidence * 100, 2),
            image=filename,           # Just 'filename.jpg'
            heatmap=filename,         # Grad-CAM uses the same filename in a different folder
            probabilities=probabilities,
            status=status,
            message=message
        )

    return render_template('index.html', message="Invalid file")
# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)