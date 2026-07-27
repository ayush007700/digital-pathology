"""
End-to-End Digital Pathology Pipeline

Author: Ayush Raj
"""

import torch

from src.inference.predictor import Predictor
from src.inference.preprocess import ImagePreprocessor
from src.inference.postprocess import PostProcessor

from src.explainability.gradcam import GradCAM
from src.explainability.heatmap import generate_heatmap
from src.explainability.overlay import overlay_heatmap

from src.rag.agents.graph import graph

import cv2
from pathlib import Path
import uuid


class DigitalPathologyPipeline:

    def __init__(

        self,

        model,

        device,

    ):

        self.device = device

        self.predictor = Predictor(
            model,
            device,
        )

        self.preprocess = ImagePreprocessor()

        self.postprocess = PostProcessor()

        self.gradcam = GradCAM(
            model,
            model.model.layer4[-1],
        )

    def run(

        self,

        image_path,

        clinical_question,

    ):

        image_tensor = self.preprocess(
            image_path
        )

        prediction = self.predictor.predict(
            image_tensor
        )

        result = self.postprocess(
            prediction
        )

        image_tensor = image_tensor.to(
            self.device
        )

        cam = self.gradcam.generate(
            image_tensor
        )

        heatmap = generate_heatmap(cam)

        original = cv2.imread(image_path)

        original = cv2.resize(
            original,
            (96,96),
        )

        overlay = overlay_heatmap(
            original / 255,
            heatmap,
        )

        Path(
            "outputs/overlays"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = f"{uuid.uuid4()}.png"

        overlay_path = (
            Path("outputs/overlays")
            / filename
        )

        cv2.imwrite(
            str(overlay_path),
            overlay,
        )

        thread_id = str(uuid.uuid4())

        state = graph.invoke(
            {
                "question": clinical_question,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        return {

            "prediction":
                result["prediction"],

            "confidence":
                result["confidence"],

            "probabilities":
                result["probabilities"],

            "overlay":
                str(overlay_path),

            "report":
                state["report_path"],

        }