import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
IMG_TARGET_SIZE = (224, 224)
CLASS_LABELS = ['Biodegradable Images (0)', 'Recyclable Images (1)', 'Trash Images (2)']

# Initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model
model_path = "vgg16.h5"
model = None
try:
    model = tf.keras.models.load_model(model_path)
    print("✅ Deep learning model loaded successfully.")
except Exception as e:
    print("❌ ERROR loading model:", e)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_waste():
    file = request.files.get('pc_image')
   
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img = load_img(filepath, target_size=IMG_TARGET_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction[0])
        predicted_label = CLASS_LABELS[class_index]
    except Exception as e:
        return render_template("index.html", predict=f"❌ Prediction error: {e}")

    return render_template("index.html", predict=predicted_label, uploaded_image=filename)

# ✅ MAKE SURE THIS BLOCK EXISTS
if __name__ == '__main__':
    app.run(debug=True, port=2222)
