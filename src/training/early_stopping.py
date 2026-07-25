"""
Early Stopping
"""


class EarlyStopping:

    def __init__(

        self,

        patience=5,

    ):

        self.patience = patience

        self.counter = 0

        self.best_loss = float("inf")

        self.should_stop = False

    def step(

        self,

        validation_loss,

    ):

        if validation_loss < self.best_loss:

            self.best_loss = validation_loss

            self.counter = 0

        else:

            self.counter += 1

        if self.counter >= self.patience:

            self.should_stop = True