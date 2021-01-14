from flask import Flask, Response, __version__, request, render_template, abort, jsonify
from io import BytesIO
from PIL import Image
import medscan
import numpy as np
from flask_cors import CORS, cross_origin
import pdf2image

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

ok_exts = {"png", "jpg","jpeg", "pdf"}
get_ext = lambda x: x.split('.')[-1].lower()

#TODO remove copy?
def pil2cv(image):
    temp = np.asarray(image)
    return temp[:, :, ::-1].copy()

def predict_from_bytes(bytes, pdf=False, debug=False):
    if pdf:
        image_list = pdf2image.convert_from_bytes(bytes)
    else:
        image_list = [Image.open(BytesIO(bytes))]

    image_list = [pil2cv(img.convert('RGB')) for img in image_list]

    return medscan.predict(image_list, debug=debug)
    

@app.route("/")
def index():
    return render_template("index.html", text=False)


@app.route("/debug_upload", methods=["POST"])
def manual_upload():
    file_ = request.files["file"]
    bytes = file_.read()
    result = predict_from_bytes(bytes, debug=True, pdf=get_ext(file_.filename)=='pdf')
    return render_template("index.html", text=result)

@app.route("/upload", methods=["POST"])
@cross_origin()
def upload():
    file_ = request.files["file"]
    bytes = file_.read()
    if not file_ or get_ext(file_.filename) not in ok_exts:
        return {'error': True}
    result = predict_from_bytes(bytes, pdf=get_ext(file_.filename)=='pdf')
    return result
