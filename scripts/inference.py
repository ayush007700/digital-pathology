import torch

from src.inference.predictor import Predictor
from src.inference.preprocess import ImagePreprocessor
from src.inference.postprocess import PostProcessor

from src.models.resnet18 import MedicalResNet18


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = MedicalResNet18(
    pretrained=False,
)

# TODO:
# model.load_state_dict(torch.load(...))

predictor = Predictor(
    model,
    device,
)

preprocess = ImagePreprocessor()

postprocess = PostProcessor()

image = preprocess(
    "sample.png"
)

prediction = predictor.predict(
    image,
)

result = postprocess(
    prediction,
)

print(result)