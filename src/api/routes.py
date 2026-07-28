import os
import tempfile

from fastapi import APIRouter, File, Request, UploadFile

from src.pipeline.factory import build_pipeline

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "pipeline": "loaded",
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
        pipeline = build_pipeline(model, device)

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
