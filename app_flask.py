import tensorflow as tf
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import cv2
import numpy as np
import json
import shutil
import datetime
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
app.secret_key = 'optiscan-clinical-portal-secret-key-2026'

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
def preprocess_image(path, save_filename=None):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # CLAHE (same as training)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    if save_filename:
        os.makedirs('static/preprocessed', exist_ok=True)
        cv2.imwrite(os.path.join('static/preprocessed', save_filename), cv2.cvtColor(img_clahe, cv2.COLOR_RGB2BGR))

    img = preprocess_input(img_clahe)
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
# ---------------- PATIENT DB HELPERS ----------------
PATIENTS_FILE = 'patients.json'

def load_patients():
    if not os.path.exists(PATIENTS_FILE):
        return []
    try:
        with open(PATIENTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print("Error loading patients:", e)
        return []

def save_patients(patients):
    try:
        with open(PATIENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(patients, f, indent=2)
        return True
    except Exception as e:
        print("Error saving patients:", e)
        return False

# ---------------- AUTH CHECK ----------------
def is_authenticated():
    return session.get('logged_in') == True

# ---------------- AUTH ROUTES ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if is_authenticated():
        return redirect(url_for('index'))
        
    error = None
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email == 'doctor@optiscan.ai' and password == 'clinical2026':
            session['logged_in'] = True
            session['user_email'] = email
            return redirect(url_for('index'))
        else:
            error = 'Invalid clinical credentials. Please try again.'
            
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('user_email', None)
    return redirect(url_for('login'))

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    if not is_authenticated():
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/api/patients', methods=['GET'])
def get_patients():
    if not is_authenticated():
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    return jsonify(load_patients())

@app.route('/api/patients/add', methods=['POST'])
def add_patient():
    if not is_authenticated():
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    data = request.json or request.form
    name = data.get('name')
    age = data.get('age')
    gender = data.get('gender')
    medical_history = data.get('medical_history', '')
    
    if not name or not age or not gender:
        return jsonify({"success": False, "message": "Missing required fields"}), 400
        
    patients = load_patients()
    patient_id = f"PAT-{np.random.randint(1000, 9999)}"
    while any(p['id'] == patient_id for p in patients):
        patient_id = f"PAT-{np.random.randint(1000, 9999)}"
        
    new_patient = {
        "id": patient_id,
        "name": name,
        "age": int(age),
        "gender": gender,
        "medical_history": medical_history,
        "scans": []
    }
    patients.append(new_patient)
    save_patients(patients)
    
    return jsonify({"success": True, "patient": new_patient})

@app.route('/api/patients/delete/<patient_id>', methods=['POST'])
def delete_patient(patient_id):
    if not is_authenticated():
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    patients = load_patients()
    patients = [p for p in patients if p['id'] != patient_id]
    save_patients(patients)
    return jsonify({"success": True})

@app.route('/predict', methods=['POST'])
def predict():
    if not is_authenticated():
        return redirect(url_for('login'))
    if 'file' not in request.files:
        return render_template('index.html', message="No file uploaded")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message="No file selected")

    if file and allowed_file(file.filename):
        # 1. Secure and save the original file
        filename = f"scan_{int(datetime.datetime.now().timestamp())}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 2. Run the AI Prediction (saving the preprocessed image too)
        img = preprocess_image(file_path, save_filename=filename)
        preds = model.predict(img)

        # 3. Analyze confidence and labels
        label, confidence, class_index, probabilities, status, message = analyze_prediction(preds)

        # 4. Generate Grad-CAM (saves heatmap into static/gradcam)
        generate_gradcam(file_path, class_index)
        
        # 5. Link to patient history if patient_id is provided
        patient_id = request.form.get('patient_id')
        eye_side = request.form.get('eye_side', 'Not Specified')
        patient_name = None
        
        if patient_id and patient_id != 'none':
            patients = load_patients()
            for p in patients:
                if p['id'] == patient_id:
                    scan_id = f"SCN-{datetime.datetime.now().strftime('%Y%m%d')}-{np.random.randint(10, 99)}"
                    scan_entry = {
                        "scan_id": scan_id,
                        "date": datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                        "eye": eye_side,
                        "diagnosis": label,
                        "confidence": round(confidence * 100, 2),
                        "image": filename,
                        "heatmap": filename,
                        "status": status,
                        "message": message,
                        "probabilities": probabilities
                    }
                    p['scans'].append(scan_entry)
                    patient_name = p['name']
                    save_patients(patients)
                    break
        
        return render_template(
            'predict.html',
            diagnosis=label,
            confidence=round(confidence * 100, 2),
            image=filename,
            heatmap=filename,
            probabilities=probabilities,
            status=status,
            message=message,
            patient_id=patient_id,
            patient_name=patient_name,
            eye_side=eye_side
        )

    return render_template('index.html', message="Invalid file")

@app.route('/predict_sample', methods=['POST'])
def predict_sample():
    if not is_authenticated():
        return redirect(url_for('login'))
    sample_type = request.form.get('sample_type') # 'healthy' or 'referable'
    patient_id = request.form.get('patient_id')
    eye_side = request.form.get('eye_side', 'Not Specified')
    
    if sample_type == 'healthy':
        sample_path = 'static/samples/healthy.png'
    elif sample_type == 'referable':
        sample_path = 'static/samples/referable.png'
    else:
        return render_template('index.html', message="Invalid sample selection")
        
    filename = f"sample_{sample_type}_{int(datetime.datetime.now().timestamp())}.png"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    shutil.copy(sample_path, file_path)
    
    # 2. Run the AI Prediction (saving the preprocessed image too)
    img = preprocess_image(file_path, save_filename=filename)
    preds = model.predict(img)

    # 3. Analyze confidence and labels
    label, confidence, class_index, probabilities, status, message = analyze_prediction(preds)

    # 4. Generate Grad-CAM
    generate_gradcam(file_path, class_index)
    
    # 5. Link to patient history
    patient_name = None
    if patient_id and patient_id != 'none':
        patients = load_patients()
        for p in patients:
            if p['id'] == patient_id:
                scan_id = f"SCN-{datetime.datetime.now().strftime('%Y%m%d')}-{np.random.randint(10, 99)}"
                scan_entry = {
                    "scan_id": scan_id,
                    "date": datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                    "eye": eye_side,
                    "diagnosis": label,
                    "confidence": round(confidence * 100, 2),
                    "image": filename,
                    "heatmap": filename,
                    "status": status,
                    "message": message,
                    "probabilities": probabilities
                }
                p['scans'].append(scan_entry)
                patient_name = p['name']
                save_patients(patients)
                break
                
    return render_template(
        'predict.html',
        diagnosis=label,
        confidence=round(confidence * 100, 2),
        image=filename,
        heatmap=filename,
        probabilities=probabilities,
        status=status,
        message=message,
        patient_id=patient_id,
        patient_name=patient_name,
        eye_side=eye_side
    )
@app.route('/report/<patient_id>/<scan_id>')
def view_report(patient_id, scan_id):
    if not is_authenticated():
        return redirect(url_for('login'))
    patients = load_patients()
    for p in patients:
        if p['id'] == patient_id:
            for s in p['scans']:
                if s['scan_id'] == scan_id:
                    return render_template(
                        'predict.html',
                        diagnosis=s['diagnosis'],
                        confidence=s['confidence'],
                        image=s['image'],
                        heatmap=s['heatmap'],
                        probabilities=s.get('probabilities', []),
                        status=s['status'],
                        message=s['message'],
                        patient_id=p['id'],
                        patient_name=p['name'],
                        eye_side=s['eye']
                    )
    return "Report not found", 404

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)