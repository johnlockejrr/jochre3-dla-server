import io

from PIL import Image
from fastapi import FastAPI, File, UploadFile, Response
from pydantic import BaseModel, Field

from app.Analyzer import Analyzer

class TextAnalysisRequest(BaseModel):
    text: str = Field(example="The cat ate the mouse. Then it slept.")
    language: str = Field(example="en")

app = FastAPI(
    title="YOLO Document Layout Analysis server",
    version="0.0.1-SNAPSHOT"
)

analyzer = Analyzer()

@app.post("/analyze")
async def analyze(imageFile: UploadFile = File(...)):
    original_image = Image.open(imageFile.file)
    stacked_mask_image = analyzer.generate_stacked_mask(original_image)

    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        stacked_mask_image.save(buf, format='PNG')
        im_bytes = buf.getvalue()

    headers = {'Content-Disposition': 'inline; filename="masks.png"'}
    return Response(im_bytes, headers=headers, media_type='image/png')

@app.post("/display")
async def display(imageFile: UploadFile = File(...)):
    original_image = Image.open(imageFile.file)
    pretty_image = analyzer.generate_pretty_image(original_image)

    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        pretty_image.save(buf, format='PNG')
        im_bytes = buf.getvalue()

    headers = {'Content-Disposition': 'inline; filename="image_with_masks.png"'}
    return Response(im_bytes, headers=headers, media_type='image/png')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8444)