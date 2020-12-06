from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import load_model
import os
from PIL import Image
import io, sys

model = load_model('model/chest_xray_cnn_100_801010.h5')

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

app = Flask(__name__, static_url_path="/static")

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# limit upload size upto 8mb
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024


def allowed_file(filename):
    # make sure it is only an allowed extension as we defined in ALLOWED_EXTENSIONS set
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # next line prevents hacking tricks with uploaded files accessing bash (The more you know***)
            filename = secure_filename(file.filename)

            requested_image = request.files['file'].read()


            test_me = Image.open(io.BytesIO(requested_image))
            if test_me.mode != 'RGB':
                test_me = test_me.convert('RGB')

            print('IMG TYPE!', type(test_me), file=sys.stderr)

            test_me = test_me.resize((150, 150))  # image is from keras

            test_me = image.img_to_array(test_me)
            print('IMG SIZE!', test_me.shape, file=sys.stderr)


            test_me = np.expand_dims(test_me, axis=0)

            inputs = preprocess_input(test_me)
            result = model.predict(inputs)
            if result[0][0] >= 0.5:
                pred = 'Pneumonia'
            else:
                pred = 'Normal'

            result = "{:.2f}".format(result[0][0])

            return render_template('index.html', filename=filename, pred=pred, result=result)  # pass whatever we need to populate index

    return render_template('index.html')  # pass whatever we need to populate index



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



if __name__ == '__main__':
    app.run()