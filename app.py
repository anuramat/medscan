from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from io import BytesIO
from PIL import Image
import medscan
import numpy as np
import pdf2image

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    

@app.get("/")
def index():
    return render_template("index.html", text=False)


@app.post("/manual_upload")
async def manual_upload():
    file_ = request.files["file"]
    bytes = file_.read()
    loop = asyncio.get_event_loop()
    with concurrent.futures.ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, predict_from_bytes(bytes, debug=True, pdf=get_ext(file_.filename)=='pdf'))
    return render_template("index.html", text=result)

@app.post("/upload")
async def upload():
    file_ = request.files["file"]
    bytes = file_.read()
    if not file_ or get_ext(file_.filename) not in ok_exts:
        return {'error': True}
    loop = asyncio.get_event_loop()
    with concurrent.futures.ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, predict_from_bytes(bytes, pdf=get_ext(file_.filename)=='pdf'))
    return result
