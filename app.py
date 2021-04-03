from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import concurrent
from io import BytesIO
from PIL import Image, ImageOps
import medscan
import numpy as np
import pdf2image
import time

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
get_ext = lambda x: x.split('.')[-1].lower()

#TODO remove copy?
def pil2cv(image):
    temp = np.asarray(image)
    ## saving image for debug
    #with open('array', 'wb') as file:
    #    np.save(file, temp)
    return temp[:, :, ::-1].copy()

def predict_from_bytes(data, pdf=False):
    if pdf:
        image_list = pdf2image.convert_from_bytes(data)
    else:
        image_list = [Image.open(BytesIO(data))]
    

    image_list = [pil2cv(ImageOps.exif_transpose(img.convert('RGB'))) for img in image_list]
    if verbosity>=1:
        print(f'{len(image_list)} pages')
    return medscan.predict(image_list)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if verbosity>=1:
        print('Upload incoming')
    start_time = time.time() 
    if not file or get_ext(file.filename) not in ok_exts:
        return {'error': True}
    data = await file.read()
    loop = asyncio.get_event_loop()
    with concurrent.futures.ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, predict_from_bytes, data, get_ext(file.filename)=='pdf')
    if verbosity>=1:
        print(f'Elapsed time:{time.time()-start_time}')
    return result

if __name__ == "__main__":
    uvicorn.run("app:app")
