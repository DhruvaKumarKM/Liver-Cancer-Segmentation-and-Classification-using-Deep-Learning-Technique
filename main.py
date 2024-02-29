from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow import keras
from tensorflow.keras import layers
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads/'


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = '1a2b3c4d5e'


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','jfif'])
class_names = ['non-cancerous','cancerous']
img_height = 300
img_width = 300
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# http://localhost:5000/pythinlogin/home - this will be the home page, only accessible for loggedin users
@app.route('/',methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        
        if username == 'admin' and password == 'admin':
            session['loggedin'] = True
            session['id'] = 1
            session['username'] = "admin"
            return render_template('index.html')
            #return redirect(url_for('upload_image'))
        else:
            msg = 'Incorrect username/password!'
            
    return render_template('login.html', msg=msg)

@app.route('/home') #end-point(starting or entering point)
def home():
    return render_template('index.html')
    # User is not loggedin redirect to login page

@app.route('/about')
def about():
    # Check if user is loggedin
        
        # User is loggedin show them the home page
    return render_template('about.html')
    # User is not loggedin redirect to login page

@app.route('/home1')
def home1():
    # Check if user is loggedin
        
        # User is loggedin show them the home page
    return render_template('home.html')

    # User is not loggedin redirect to login page
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    print(file)
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(path)

        num_classes = 2
        model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
        ])
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        model.load_weights("liver_cancer.h5")

        test_data_path = path

        img = keras.preprocessing.image.load_img(
            test_data_path, target_size=(img_height, img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(np.argmax(score), 100 * np.max(score))
        )
        print(np.argmax(score))
        return render_template('result.html', filename=filename,aclass=class_names[np.argmax(score)],ascore=100 * np.max(score),res=1)
        
@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

    
if __name__ =='__main__':
	app.run()
