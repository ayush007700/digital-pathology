import os
import tempfile

from fastapi import APIRouter, File, Request, UploadFile

from src.pipeline.pathology_pipeline import DigitalPathologyPipeline

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded",
        "device": "cpu",
        "version": "1.0.0",
    }


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".png"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp:
        temp.write(await file.read())
        image_path = temp.name

    try:
        model = request.app.state.model
        device = next(model.parameters()).device

        pipeline = DigitalPathologyPipeline(model, device)
        result = pipeline.run(
            image_path=image_path,
            clinical_question=(
                "Explain the pathological significance of "
                "HER2-positive breast cancer and summarize relevant evidence."
            ),
        )
    finally:
        os.remove(image_path)

    return result
