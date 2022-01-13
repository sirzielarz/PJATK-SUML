from __future__ import division
from flask import Flask, request, render_template
from fastai.vision.all import *
from fastai.vision.widgets import *
import base64


app = Flask(__name__)

model = 'model.pkl'

def model_predict(img_path, model_path):

    learn_inf = load_learner(model_path)
    pred , pred_idx , probs = learn_inf.predict(img_path)
    prob_value = probs[pred_idx] * 100
    out = f'Na {prob_value:.02f} % na tym zdjęciu znajdują się {pred}.👨‍🔬.'
    return out

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        f = request.files['file']
        basepath = Path(__file__).parent
        file_path = basepath.joinpath('uploads')
        filename = 'test.jpg'
        file_path = file_path.joinpath(filename)
        f.save(file_path)
        os.system("python yolov5/detect.py --weights best.pt --name ../../../output --img 416 --conf 0.50 --source uploads --exist-ok")
        # path = Path()
        # model_path = (path/model)
        # out = model_predict(file_path, model_path)
        with open("output/test.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            out = encoded_string
            return out
    return None

if __name__ == '__main__':
    app.run(debug=True)
