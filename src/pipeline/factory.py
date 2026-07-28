"""
Pipeline Factory
"""

from src.inference.preprocess import ImagePreprocessor
from src.pipeline.pathology_pipeline import DigitalPathologyPipeline
from src.services.explainability_service import ExplainabilityService
from src.services.model_service import ModelService
from src.services.rag_service import RagService
from src.services.report_service import ReportService


def build_pipeline(model, device):
    preprocess = ImagePreprocessor()
    model_service = ModelService(model, device)
    explainability_service = ExplainabilityService(model)
    rag_service = RagService()
    report_service = ReportService()

    return DigitalPathologyPipeline(
        preprocess,
        model_service,
        explainability_service,
        rag_service,
        report_service,
    )
