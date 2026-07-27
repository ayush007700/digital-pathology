import torch

from src.inference.predictor import Predictor


class DummyModel(torch.nn.Module):

    def forward(self, x):

        return torch.tensor([[0.2, 1.8]])


def test_predictor():

    predictor = Predictor(
        DummyModel(),
        "cpu",
    )

    image = torch.randn(
        1,
        3,
        96,
        96,
    )

    result = predictor.predict(image)

    assert result["prediction"].item() == 1