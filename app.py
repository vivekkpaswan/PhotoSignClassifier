from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.secret_key = "secret_key"

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained CNN model
model = tf.keras.models.load_model('photo_signature_classifier.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_image(image_path):
    img = Image.open(image_path)
    img = img.resize((300, 300))
    img = img.convert('RGB')
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction[0][0]

@app.route('/',methods=['GET'])
def upload_file():
    return render_template('upload.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        photos = request.files.getlist('photos[]')
        signatures = request.files.getlist('signatures[]')
        results = []
        
        for file in photos:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                classification = classify_image(file_path)
                if classification > 0.5:
                    flash(f"Error: {filename} is a signature, not a photo.")
                    os.remove(file_path)
                    return redirect(url_for('upload_file'))
                else:
                    results.append((filename, "Photo"))

        for file in signatures:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                classification = classify_image(file_path)
                if classification <= 0.5:
                    flash(f"Error: {filename} is a photo, not a signature.")
                    os.remove(file_path)
                    return redirect(url_for('upload_file'))
                else:
                    results.append((filename, "Signature"))
        
        return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)