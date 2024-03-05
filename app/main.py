import os
from typing import Annotated, Union

from PIL import Image
from fastapi import FastAPI, Query, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from ultralytics import YOLO

app = FastAPI(
    title="YOLO Document Layout Analysis server",
    version="1.0.0"
)

model_path = 'models'

text_block_model = YOLO(os.path.join(model_path, 'yolov8n-yiddish-detect-blocks-1280.pt'))
text_line_model = YOLO(os.path.join(model_path, 'yolov8n-yiddish-detect-lines-1280.pt'))
word_model = YOLO(os.path.join(model_path, 'yolov8n-yiddish-detect-words-1280.pt'))
glyph_model = YOLO(os.path.join(model_path, 'yolov8n-yiddish-detect-glyphs-tiled-1280-tsvey-yudn.pt'))
glyph_model_simple = YOLO(os.path.join(model_path, 'yolov8n-yiddish-detect-glyphs-tiled-1280.pt'))

glyph_image_size = 1280

@app.post("/analyze-blocks")
async def analyze_blocks(
    min_confidence: Annotated[Union[float, None], Query(alias="min-confidence")] = None,
    max_items: Annotated[Union[int, None], Query(alias="max-items")] = None,
    imageFile: UploadFile = File(...)
):
    original_image = Image.open(imageFile.file)

    confidence = 0.25
    if (min_confidence):
        confidence = min_confidence

    max_items_to_predict = 300
    if (max_items):
        max_items_to_predict = max_items

    try:
        results = text_block_model(original_image, imgsz=640, conf=confidence, retina_masks=True, max_det=max_items_to_predict)
    except:
        results = text_block_model(original_image, imgsz=640, conf=confidence, retina_masks=False, max_det=max_items_to_predict)

    response = result_to_response(results)

    return JSONResponse(content=jsonable_encoder(response))

@app.post("/analyze-lines")
async def analyze_lines(
    min_confidence: Annotated[Union[float, None], Query(alias="min-confidence")] = None,
    max_items: Annotated[Union[int, None], Query(alias="max-items")] = None,
    imageFile: UploadFile = File(...)
):
    original_image = Image.open(imageFile.file)

    confidence = 0.25
    if (min_confidence):
        confidence = min_confidence

    max_items_to_predict = 300
    if (max_items):
        max_items_to_predict = max_items

    results = text_line_model(original_image, imgsz=1280, conf=confidence, max_det=max_items_to_predict)

    response = result_to_response(results)

    return JSONResponse(content=jsonable_encoder(response))

@app.post("/analyze-words")
async def analyze_words(
    min_confidence: Annotated[Union[float, None], Query(alias="min-confidence")] = None,
    max_items: Annotated[Union[int, None], Query(alias="max-items")] = None,
    imageFile: UploadFile = File(...)
):
    original_image = Image.open(imageFile.file)

    confidence = 0.25
    if (min_confidence):
        confidence = min_confidence

    max_items_to_predict = 1500
    if (max_items):
        max_items_to_predict = max_items

    results = word_model(original_image, imgsz=1280, conf=confidence, max_det=max_items_to_predict)

    response = result_to_response(results)

    return JSONResponse(content=jsonable_encoder(response))


@app.post("/analyze-glyphs")
async def analyze_glyphs(
    min_confidence: Annotated[Union[float, None], Query(alias="min-confidence")] = None,
    max_items: Annotated[Union[int, None], Query(alias="max-items")] = None,
    imageFile: UploadFile = File(...)
):
    original_image = Image.open(imageFile.file)

    confidence = 0.25
    if (min_confidence):
        confidence = min_confidence

    max_items_to_predict = 6000
    if (max_items):
        max_items_to_predict = max_items

    results = glyph_model(original_image, imgsz=glyph_image_size, conf=confidence, max_det=max_items_to_predict)

    response = result_to_response(results)

    return JSONResponse(content=jsonable_encoder(response))

@app.post("/analyze-glyphs-simple")
async def analyze_glyphs_simple(
        min_confidence: Annotated[Union[float, None], Query(alias="min-confidence")] = None,
        max_items: Annotated[Union[int, None], Query(alias="max-items")] = None,
        imageFile: UploadFile = File(...)
):
    original_image = Image.open(imageFile.file)

    confidence = 0.25
    if (min_confidence):
        confidence = min_confidence

    max_items_to_predict = 6000
    if (max_items):
        max_items_to_predict = max_items

    results = glyph_model_simple(original_image, imgsz=glyph_image_size, conf=confidence, max_det=max_items_to_predict)

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