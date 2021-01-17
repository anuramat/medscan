from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import concurrent

# for manual_upload path
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
templates = Jinja2Templates(directory="templates")

from io import BytesIO
from PIL import Image
import medscan
import numpy as np
import pdf2image

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
get_ext = lambda x: x.split('.')[-1].lower()

#TODO remove copy?
def pil2cv(image):
    temp = np.asarray(image)
    return temp[:, :, ::-1].copy()

def predict_from_bytes(data, pdf=False, debug=False):
    if pdf:
        image_list = pdf2image.convert_from_bytes(data)
    else:
        image_list = [Image.open(BytesIO(data))]

    image_list = [pil2cv(img.convert('RGB')) for img in image_list]

    return medscan.predict(image_list, debug=debug)
    

@app.get("/")
def index():
    return render_template("index.html", text=False)


@app.post("/manual_upload", response_class=HTMLResponse)
async def manual_upload(file: UploadFile = File(...)):
    data = file.read()
    loop = asyncio.get_event_loop()
    with concurrent.futures.ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, predict_from_bytes(file, debug=True, pdf=get_ext(file.filename)=='pdf'))
    return templates.TemplateResponse("index.html", {"text": result})

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file or get_ext(file.filename) not in ok_exts:
        return {'error': True}
    data = await file.read()
    loop = asyncio.get_event_loop()
    with concurrent.futures.ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, predict_from_bytes, data, get_ext(file.filename)=='pdf')
    return result

if __name__ == "__main__":
    uvicorn.run("app:app") # filename:app
    # to run: # uvicorn run fastapi_code:app (--reload for fast reloading)
