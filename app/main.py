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

glyph_image_size = 1280
# Strangely, blocks predict better at 640 although trained at 1280! (discovered by mistake)
block_image_size = 640
#block_image_size = 1280

word_to_glyph_image_size = 128

class Models(object):
    def __init__(self):
        self._text_block_model = None
        self._text_line_model = None
        self._word_model = None
        self._glyph_model = None
        self._glyph_model_simple = None
        self._word_to_glyph_model = None

    def get_text_block_model(self):
        if self._text_block_model is None:
            self._text_block_model = YOLO(os.path.join(model_path, 'yolov8n-yiddish-detect-blocks-1280.pt'))
        return self._text_block_model

    def get_text_line_model(self):
        if self._text_line_model is None:
            self._text_line_model = YOLO(os.path.join(model_path, 'yolov8n-yiddish-detect-lines-1280.pt'))
        return self._text_line_model

    def get_word_model(self):
        if self._word_model is None:
            self._word_model = YOLO(os.path.join(model_path, 'yolov8n-yiddish-detect-words-1280.pt'))
        return self._word_model

    def get_glyph_model(self):
        if self._glyph_model is None:
            self._glyph_model = YOLO(os.path.join(model_path, 'yolov8n-yiddish-detect-glyphs-tiled-1280-pasekh-tsvey-yudn.pt'))
        return self._glyph_model

    def get_glyph_model_simple(self):
        if self._glyph_model_simple is None:
            self._glyph_model_simple = YOLO(os.path.join(model_path, 'yolov8n-yiddish-detect-glyphs-tiled-1280.pt'))
        return self._glyph_model_simple

    def get_word_to_glyph_model(self):
        if self._word_to_glyph_model is None:
            self._word_to_glyph_model = YOLO(os.path.join(model_path, 'yolov8n-yiddish-detect-word-to-glyph.pt'))
        return self._word_to_glyph_model

models = Models()

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

    text_block_model = models.get_text_block_model()
    try:
        results = text_block_model(original_image, imgsz=block_image_size, conf=confidence, retina_masks=True, max_det=max_items_to_predict)
    except:
        results = text_block_model(original_image, imgsz=block_image_size, conf=confidence, retina_masks=False, max_det=max_items_to_predict)

    response = result_to_response(results, text_block_model)

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

    text_line_model = models.get_text_line_model()
    results = text_line_model(original_image, imgsz=1280, conf=confidence, max_det=max_items_to_predict)

    response = result_to_response(results, text_line_model)

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

    word_model = models.get_word_model()
    results = word_model(original_image, imgsz=1280, conf=confidence, max_det=max_items_to_predict)

    response = result_to_response(results, word_model)

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

    glyph_model = models.get_glyph_model()
    results = glyph_model(original_image, imgsz=glyph_image_size, conf=confidence, max_det=max_items_to_predict)

    response = result_to_response(results, glyph_model)

    return JSONResponse(content=jsonable_encoder(response))

@app.post("/analyze-glyphs-simple")
async def analyze_glyphs_simple(
        min_confidence: Annotated[Union[float, None], Query(alias="min-confidence")] = None,
        max_items: Annotated[Union[int, None], Query(alias="max-items")] = None,
        imageFile: UploadFile = File(...)
):
    original_image = Image.open(imageFile.file)

    confidence = 0.25
    if min_confidence is not None:
        confidence = min_confidence

    max_items_to_predict = 6000
    if max_items is not None:
        max_items_to_predict = max_items

    glyph_model_simple = models.get_glyph_model_simple()

    results = glyph_model_simple(original_image, imgsz=glyph_image_size, conf=confidence, max_det=max_items_to_predict)

    response = result_to_response(results, glyph_model_simple)

    return JSONResponse(content=jsonable_encoder(response))

@app.post("/word-to-glyph")
async def word_to_glyph(
        min_confidence: Annotated[Union[float, None], Query(alias="min-confidence")] = None,
        max_items: Annotated[Union[int, None], Query(alias="max-items")] = None,
        imageFile: UploadFile = File(...)
):
    original_image = Image.open(imageFile.file)

    confidence = 0.25
    if min_confidence is not None:
        confidence = min_confidence

    max_items_to_predict = 100
    if max_items is not None:
        max_items_to_predict = max_items

    word_to_glyph_model = models.get_word_to_glyph_model()

    results = word_to_glyph_model(original_image, imgsz=word_to_glyph_image_size, conf=confidence, max_det=max_items_to_predict)

    response = result_to_response(results, word_to_glyph_model)

    return JSONResponse(content=jsonable_encoder(response))


def result_to_response(results, model):
    result = results[0]
    boxes = result.boxes.xywh.numpy()
    classes = result.boxes.cls.numpy()
    confidences = result.boxes.conf.numpy()

    response = []
    for i in range(len(classes)):
        box = boxes[i].tolist()
        box = list(map(lambda x: round(x), box))
        clazz = classes[i]
        class_name = model.names[clazz]
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