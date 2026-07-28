"""
Report Service
"""

from src.rag.report.report_generator import ReportGenerator


class ReportService:

    def __init__(self):
        self.generator = ReportGenerator()

    def generate(
        self,
        prediction: str,
        confidence: float,
        llm_answer: str,
        model_name: str = "ResNet18",
    ) -> str:
        path = self.generator.generate(
            prediction=prediction,
            confidence=confidence,
            llm_answer=llm_answer,
            model_name=model_name,
        )
        return str(path)
