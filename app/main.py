import os

from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from ultralytics import YOLO

app = FastAPI(
    title="YOLO Document Layout Analysis server",
    version="0.0.1"
)

model_path = 'models'

text_block_model = YOLO(os.path.join(model_path, 'e50_aug.pt'))
text_line_model = YOLO(os.path.join(model_path, 'yolov8n-yiddish-detect-lines-1280.pt'))
word_model = YOLO(os.path.join(model_path, 'yolov8n-yiddish-detect-words-1280.pt'))
glyph_model = YOLO(os.path.join(model_path, 'yolov8n-yiddish-detect-glyphs-1280.pt'))

@app.post("/analyze_blocks")
async def analyze_blocks(imageFile: UploadFile = File(...)):
    original_image = Image.open(imageFile.file)
    try:
        results = text_block_model(original_image, imgsz=640, conf=0.25, retina_masks=True)
    except:
        results = text_block_model(original_image, imgsz=640, conf=0.25, retina_masks=False)

    response = result_to_response(results)

    return JSONResponse(content=jsonable_encoder(response))

@app.post("/analyze_lines")
async def analyze_lines(imageFile: UploadFile = File(...)):
    original_image = Image.open(imageFile.file)

    results = text_line_model(original_image, imgsz=1280, conf=0.25)

    response = result_to_response(results)

    return JSONResponse(content=jsonable_encoder(response))

@app.post("/analyze_words")
async def analyze_words(imageFile: UploadFile = File(...)):
    original_image = Image.open(imageFile.file)

    results = word_model(original_image, imgsz=1280, conf=0.25, max_det=1500)

    response = result_to_response(results)

    return JSONResponse(content=jsonable_encoder(response))


@app.post("/analyze_glyphs")
async def analyze_glyphs(imageFile: UploadFile = File(...)):
    original_image = Image.open(imageFile.file)

    results = glyph_model(original_image, imgsz=1280, conf=0.25, max_det=6000)

    response = result_to_response(results)

    return JSONResponse(content=jsonable_encoder(response))


def result_to_response(results):
    result = results[0]
    boxes = result.boxes.xywh.numpy()
    classes = result.boxes.cls.numpy()
    confidences = result.boxes.conf.numpy()

    response = []
    for i in range(len(classes)):
        box = boxes[i].tolist()
        box = list(map(lambda x: round(x), box))
        clazz = classes[i]
        class_name = text_block_model.names[clazz]
        confidence = round(confidences[i].item(), 2)
        response.append({
            "box": box,
            "class": class_name,
            "conf": confidence
        })

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8444)