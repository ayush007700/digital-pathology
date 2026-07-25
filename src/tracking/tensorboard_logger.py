"""
TensorBoard Logger

Author: Ayush Raj
"""

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:

    def __init__(self, log_dir="runs"):

        self.writer = SummaryWriter(log_dir)

    def log_metrics(
        self,
        train_loss,
        val_loss,
        train_acc,
        val_acc,
        lr,
        epoch,
    ):

        self.writer.add_scalar(
            "Loss/Train",
            train_loss,
            epoch,
        )

        self.writer.add_scalar(
            "Loss/Validation",
            val_loss,
            epoch,
        )

        self.writer.add_scalar(
            "Accuracy/Train",
            train_acc,
            epoch,
        )

        self.writer.add_scalar(
            "Accuracy/Validation",
            val_acc,
            epoch,
        )

        self.writer.add_scalar(
            "LearningRate",
            lr,
            epoch,
        )

    def close(self):

        self.writer.close()