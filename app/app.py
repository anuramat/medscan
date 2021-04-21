from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import concurrent
from io import BytesIO
from PIL import Image, ImageOps
import medscan
import numpy as np
import pdf2image
import time

# temp 
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
templates = Jinja2Templates(directory='templates')

verbosity = 1

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ok_exts = {"png", "jpg","jpeg", "pdf"}
doc_types = {"snils", "discharge", "insurance"}
get_ext = lambda x: x.split('.')[-1].lower()

#TODO remove copy?
def pil2cv(image):
    temp = np.asarray(image)
    return temp[:, :, ::-1].copy()

def predict_from_bytes(data, doc_type, pdf=False):
    if pdf:
        image_list = pdf2image.convert_from_bytes(data)
    else:
        image_list = [Image.open(BytesIO(data))]

    image_list = [pil2cv(ImageOps.exif_transpose(img.convert('RGB'))) for img in image_list]
    if verbosity>=1:
        print(f'{len(image_list)} pages')
    return medscan.text_recognition(image_list, doc_type)

@app.post("/upload")
async def upload(file = File(...), doc_type = 'discharge'):
    if verbosity>=1:
        print('Upload incoming'
    start_time = time.time() 
    if get_ext(file.filename) not in ok_exts or doc_type not in doc_types:
        return {'error': True}
    data = await file.read()
    loop = asyncio.get_event_loop()
    with concurrent.futures.ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, predict_from_bytes, data, doc_type, get_ext(file.filename)=='pdf')
    if verbosity>=1:
        print(f'Elapsed time:{time.time()-start_time}')
    return result

# temp
@app.get("/", response_class = HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {'request': request, 'text': ''})

if __name__ == "__main__":
    uvicorn.run("app:app")
