"""
End-to-End Digital Pathology Pipeline

Author: Ayush Raj
"""

import time


class DigitalPathologyPipeline:

    def __init__(
        self,
        preprocess,
        model_service,
        explainability_service,
        rag_service,
        report_service,
    ):
        self.preprocess = preprocess
        self.model_service = model_service
        self.explainability_service = explainability_service
        self.rag_service = rag_service
        self.report_service = report_service

    def run(self, image_path, clinical_question, return_timings: bool = False):
        timings = {}

        t0 = time.perf_counter()
        image_tensor = self.preprocess(image_path)
        result = self.model_service.predict(image_tensor)
        timings["prediction_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        overlay_path = self.explainability_service.explain(
            image_tensor,
            image_path,
        )
        timings["gradcam_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        state = self.rag_service.ask(
            question=clinical_question,
            prediction=result["prediction"],
            confidence=result["confidence"],
        )
        timings["rag_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        report_path = self.report_service.generate(
            prediction=result["prediction"],
            confidence=result["confidence"],
            llm_answer=state.get("answer", ""),
        )
        timings["report_ms"] = (time.perf_counter() - t0) * 1000

        timings["total_ms"] = (
            timings["prediction_ms"]
            + timings["gradcam_ms"]
            + timings["rag_ms"]
            + timings["report_ms"]
        )

        output = {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "overlay": overlay_path,
            "report": report_path,
        }
        if return_timings:
            output["timings"] = timings
        return output
