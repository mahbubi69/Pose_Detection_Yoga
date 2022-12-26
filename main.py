#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims
import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/files/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

class_dict = {0 : 'Downdog' , 1: 'Goddess', 2: 'Plank', 3 :'Tree', 4:'Warrior_2', 5 : 'tidak ada pose'}
valueInt = 0
model = load_model('static/model/model.h5')
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Home
@app.route('/index.html')
def home():
    return render_template('index.html')

# Classification      
@app.route('/classification.html')
def classificasion():
    return render_template('classification.html')

# About 
@app.route('/about.html')
def about():
    return render_template('about.html')
 
#  Updload Image
@app.route('/readImage', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)
        
        prediction = get_output(img_path)

        # valueInt = get_output_model(img_path)

        flash('Pose Gerakan : ' + prediction,)
        # flash('Value Accurasi : ' + predition,)
        print('Pose Gerak : ', prediction)
        # print('value Model : ', valueInt)
        return render_template('classification.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

def get_output(img_path): 
    loaded_img = load_img(img_path, target_size = (150, 150))
    img_array = img_to_array(loaded_img) / 255.0
    img_array = expand_dims(img_array, 0)
    predicted_bit = np.argmax(model.predict(img_array))
    return class_dict[predicted_bit]
    
# Read Image
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='files/' + filename), code=301)

# Contact
@app.route('/contact.html')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.debug = True
    app.run()