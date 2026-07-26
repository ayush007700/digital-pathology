"""
Postprocessing

Author: Ayush Raj
"""


class PostProcessor:

    CLASS_NAMES = {
        0: "Normal",
        1: "Tumor",
    }

    def __call__(self, prediction):

        pred = prediction["prediction"].item()

        return {
            "class": self.CLASS_NAMES[pred],
            "confidence": round(
                prediction["confidence"].item() * 100,
                2,
            ),
        }