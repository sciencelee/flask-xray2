'''
Flask web app for taking in medical chest xrays and returning Pneumonia or Normal result to user
- single page 'index.html'
- image POST from form in index.html
- file is processed and used for prediction (also routed for display back to index.html)
- data routed to root index.html template
'''

from flask import Flask, request, redirect, url_for, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import load_model
import os
from PIL import Image
import io, sys
import random

model = load_model('model/chest_xray_cnn_100_801010.h5')  # model is CNN trained with 5k+ images
image_list = []



app = Flask(__name__, static_url_path="/static")

# configure my app
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # attempt to keep persistant images from appearing (clear cache)
# limit upload size upto 8mb
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024


def allowed_file(filename):
    # boilerplate function from flask
    # make sure it is only an allowed extension as we defined in ALLOWED_EXTENSIONS set
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/<id>')  # route is to an image name which we will add file to
def serve_image(id):
    # create file-object in memory only
    file_object = io.BytesIO()

    # write PNG in file-object
    # my images are stored in a list which is accessible by scope rules for python
    image_list[-1].save(file_object, 'PNG')

    # move to beginning of file so `send_file()` it will read from start
    file_object.seek(0)

    # send back an actual png image to display in HTML
    return send_file(file_object, mimetype='image/PNG', cache_timeout=0) # cache timeout or the image won't change


@app.route('/', methods=['GET', 'POST'])
def index():
    '''
    Main function for my app at the root.
    Takes in form POST data from index.html (uploaded image file)
    :return: rendered_template index.html (webpage for the app):
                                 filename(str): the name of image loaded
                                 result(str):  a string numerical between 0 and 1 that the model predicted
                                 pred(str): either "Normal" or 'Pneumonia"
                                 id(str): the file route/location for the image
    '''

    if request.method == 'POST':
        if 'file' not in request.files:
            # there was no file posted, so go back to original URL
            return redirect(request.url)

        # Okay, there was a file, and now we grab it as an object called file
        file = request.files['file']

        # if there is no filename with it, we don't do anything
        if file.filename == '':
            return redirect(request.url)

        # if we have a file and it is named properly, then we can do the processing and model part
        if file and allowed_file(file.filename):
            # next line prevents hacking tricks with uploaded files accessing bash (The more you know***)
            filename = secure_filename(file.filename)

            # pull the data from the POST data
            requested_image = request.files['file'].read()
            print(type(requested_image), file=sys.stderr)

            # BytesIO makes an object in memory, and Image makes a PIL object out of it
            pil_img = Image.open(io.BytesIO(requested_image))

            # should be 3 channel because that's what we trained on
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            # dump the PIL format image into my list
            image_list.append(pil_img)  # list has scope!!

            # Had a lot of difficutly here, so I will explain this solution
            # turns out you can't overwrite at the same img location (route), so I will make a unique id for each
            # image_served is a list to avoid scope issues and inabiltiy to pass file object
            # I will use the image_list to grab the object from that route/function
            id = str(len(image_list)) + '.png'  # this will be my file name for HTML to show xray image
            serve_image(len(image_list))

            # Data Science part
            # now do my preprocessing for prediction
            test_me = pil_img.resize((150, 150))  # image is from keras, we need 150x150 for model
            test_me = image.img_to_array(test_me) # got two lines from a medium article of guy doing similar thing
            test_me = np.expand_dims(test_me, axis=0)
            inputs = preprocess_input(test_me)  # from vgg16 module in keras
            result = model.predict(inputs) # [[float]] between 0 and 1
            result = result[0][0]

            if result >= 0.5:
                pred = 'Pneumonia'
            else:
                pred = 'Normal'

            result = "{:.2f}".format(result)

            # We have results, now pass them back into the template to display
            return render_template('index.html', filename=filename, pred=pred, result=result, id=id)  # pass whatever we need to populate index

    return render_template('index.html')  # show the template even if we got nothing from POST


if __name__ == '__main__':
    app.run()