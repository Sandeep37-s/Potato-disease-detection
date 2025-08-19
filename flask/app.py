# app.py
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64
from datetime import datetime
import os

# Initialize the Flask application
app = Flask(__name__)

# --- Define the path to your model ---
# Using the absolute path you provided.
MODEL_PATH = "1.keras"

# --- Load the Trained Keras Model ---
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model = None

# --- Data for Features ---
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# In-memory storage for prediction history (for demonstration)
# In a real application, you would use a database.
prediction_history = []

# Detailed information about each disease
DISEASE_INFO = {
    "Early Blight": {
        "description": "A common fungal disease that affects tomatoes and potatoes. It first appears on lower, older leaves as small, brown lesions.",
        "symptoms": "Dark spots with concentric rings (target spots), yellowing around spots, leaf drop.",
        "treatment": "Use fungicides containing mancozeb or chlorothalonil. Ensure good air circulation and avoid overhead watering.",
        "prevention": "Plant resistant varieties, rotate crops, and maintain garden hygiene by removing infected plant debris."
    },
    "Late Blight": {
        "description": "A devastating fungal disease caused by Phytophthora infestans. It can rapidly destroy entire crops of potatoes and tomatoes.",
        "symptoms": "Large, dark, water-soaked spots on leaves and stems. A white moldy growth may appear on the underside of leaves in humid conditions.",
        "treatment": "Apply fungicides proactively, especially during cool, wet weather. Copper-based fungicides are often used.",
        "prevention": "Ensure good drainage, space plants for air circulation, and monitor weather forecasts for conditions favorable to blight."
    },
    "Healthy": {
        "description": "The plant shows no visible signs of disease. Leaves are vibrant and well-formed.",
        "symptoms": "No spots, lesions, or discoloration. Strong, vigorous growth.",
        "treatment": "None required. Continue good care practices.",
        "prevention": "Maintain optimal watering, sunlight, and nutrient levels to keep the plant resilient to potential diseases."
    }
}


# --- Preprocessing Function ---
def preprocess_image(image_data):
    """Prepares image data for the model."""
    try:
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# --- Routes ---

@app.route('/')
def index():
    """Renders the main prediction page."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template('about.html')

@app.route('/history')
def history():
    """Renders the prediction history page."""
    # Pass the history in reverse order (newest first)
    return render_template('history.html', history=prediction_history[::-1])

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image prediction and returns JSON."""
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image_data = file.read()
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            return jsonify({'error': 'Could not process image'}), 400

        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = float(np.max(predictions[0]))
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        
        # --- Add to history ---
        prediction_record = {
            "image": f'data:image/jpeg;base64,{encoded_image}',
            "prediction": predicted_class_name,
            "confidence": f'{confidence:.2%}',
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        prediction_history.append(prediction_record)

        return jsonify({
            'prediction': predicted_class_name,
            'confidence': f'{confidence:.2%}',
            'image': f'data:image/jpeg;base64,{encoded_image}',
            'info': DISEASE_INFO.get(predicted_class_name, {})
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# --- Main execution block ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
