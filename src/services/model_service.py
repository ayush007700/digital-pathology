"""
Model Service
"""

from src.inference.predictor import Predictor
from src.inference.postprocess import PostProcessor


class ModelService:

    def __init__(self, model, device):

        self.predictor = Predictor(model, device)
        self.postprocess = PostProcessor()

    def predict(self, image):

        prediction = self.predictor.predict(image)

        return self.postprocess(prediction)