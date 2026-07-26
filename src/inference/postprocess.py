class PostProcessor:

    CLASS_NAMES = {
        0: "Normal",
        1: "Tumor",
    }

    def __call__(self, prediction):

        probs = prediction["probabilities"][0]

        return {

            "prediction":
                self.CLASS_NAMES[
                    prediction["prediction"].item()
                ],

            "confidence":
                round(
                    prediction["confidence"].item() * 100,
                    2,
                ),

            "probabilities":{

                "Normal":
                    round(
                        probs[0].item() * 100,
                        2,
                    ),

                "Tumor":
                    round(
                        probs[1].item() * 100,
                        2,
                    )

            },

            "model":"ResNet18"

        }