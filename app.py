from flask import Flask, Response, __version__, request, render_template, abort, jsonify
from io import BytesIO
from PIL import Image
import medscan
import numpy as np
app = Flask(__name__)

ext_list = ["png", "jpg","jpeg"]
ext_list += [i.upper() for i in ext_list]
ALLOWED_EXTENSIONS = set(ext_list)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXTENSIONS

def predict_image_from_bytes_debug(bytes):
    pil_img = Image.open(BytesIO(bytes)).convert('RGB')
    cv_img = np.array(pil_img)
    cv_img = cv_img[:, :, ::-1].copy() 
    return medscan.predict_debug(cv_img)

def predict_image_from_bytes(bytes):
    pil_img = Image.open(BytesIO(bytes)).convert('RGB')
    cv_img = np.array(pil_img)
    cv_img = cv_img[:, :, ::-1].copy() 
    return medscan.predict(cv_img) 
    

@app.route("/")
def index():
    return render_template("index.html", text=False)

@app.route("/manual_upload", methods=["POST"])
def manual_upload():
    file_ = request.files["file"]
    bytes = file_.read()
    if not file_:
        return render_template("index.html", text="could you please upload a file?")
    if not allowed_file(file_.filename):
        return render_template("index.html", text="unsupported format.")
    result = predict_image_from_bytes_debug(bytes)
    return render_template("index.html", text=result)

@app.route("/upload", methods=["POST"])
def upload():
    file_ = request.files["file"]
    bytes = file_.read()
    if not file_ or not allowed_file(file_.filename):
        return {'error': 'error'}
    result = predict_image_from_bytes(bytes)
    return result
