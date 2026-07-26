from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File
from fastapi import Request

import tempfile
import os

from src.inference.preprocess import ImagePreprocessor
from src.inference.predictor import Predictor
from src.inference.postprocess import PostProcessor

router = APIRouter()

preprocess = ImagePreprocessor()
postprocess = PostProcessor()


@router.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...)
):

    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp:

        temp.write(await file.read())

        image_path = temp.name

    image = preprocess(image_path)

    predictor = Predictor(
        request.app.state.model,
        request.app.state.model.device
        if hasattr(request.app.state.model, "device")
        else "cpu",
    )

    prediction = predictor.predict(image)

    result = postprocess(prediction)

    os.remove(image_path)

    return result