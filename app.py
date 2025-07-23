from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/skin_model.h5')  # Adjust this if your filename is different

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Change size if needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded."

    file = request.files['file']
    if file.filename == '':
        return "No selected file."

    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    img = preprocess_image(filepath)
    prediction = model.predict(img)

    result = 'Disease Detected' if prediction[0][0] > 0.5 else 'No Disease Detected'

    return render_template('index.html', prediction=result, user_image=filepath)

if __name__ == '__main__':
    app.run(debug=True)
