from fastapi import FastAPI

from src.api.routes import router
from src.api.model_loader import load_model

app = FastAPI(
    title="Digital Pathology Copilot"
)

app.state.model = load_model()

app.include_router(router)