import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)

# Load the trained model
# IMPORTANT: Ensure 'model.keras' is in the same directory as this script,
# or provide the full absolute path to the model file.
MODEL_PATH = r"model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    # Exit or handle the error appropriately if the model cannot be loaded
    model = None # Set model to None to prevent further errors if loading failed

# Configurations
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Ensure the upload directory exists

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    # Load image, resize to target_size, convert to array, and normalize
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0 # Normalize pixel values to [0, 1]
    return np.expand_dims(img_array, axis=0) # Add batch dimension

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/tool')
def tool():
    return render_template('tool.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('file')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400

    results = []
    for file in files:
        file_path = None # Initialize file_path to None
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path) # Save the file temporarily

            try:
                img_array = preprocess_image(file_path)
                prediction = model.predict(img_array)[0][0] # Get the prediction value
                label = "Dirty" if prediction > 0.5 else "Clean"
                confidence = prediction if label == "Dirty" else 1 - prediction

                results.append({
                    'label': label,
                    'confidence': f"{confidence:.2%}",
                    'image_url': f"/static/uploads/{unique_filename}" # URL for displaying the image
                })
            except Exception as e:
                # If an error occurs during processing, record it
                results.append({
                    'label': 'Error',
                    'confidence': 'N/A',
                    'image_url': None, # No image URL if processing failed
                    'error': str(e)
                })
            finally:
                # ALWAYS attempt to delete the file, regardless of prediction success or failure
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Deleted uploaded file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting file {file_path}: {e}")
        else:
            # Handle cases where the file itself is invalid
            results.append({
                'label': 'Error',
                'confidence': 'N/A',
                'image_url': None,
                'error': f"Invalid file type or no file: {file.filename}"
            })

    # Render a results page and pass results into it
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)