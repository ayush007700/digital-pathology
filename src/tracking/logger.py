"""
Unified Logger

Author: Ayush Raj
"""

from src.tracking.mlflow_logger import MLflowLogger
from src.tracking.tensorboard_logger import TensorBoardLogger


class Logger:

    def __init__(self, experiment_name):

        self.mlflow = MLflowLogger(experiment_name)

        self.tensorboard = TensorBoardLogger()

    def log_params(self, params):

        self.mlflow.log_params(params)

    def log_metrics(
        self,
        metrics,
        step,
    ):

        self.mlflow.log_metrics(
            metrics,
            step,
        )

        self.tensorboard.log_metrics(
            train_loss=metrics["train_loss"],
            val_loss=metrics["valid_loss"],
            train_acc=metrics["train_accuracy"],
            val_acc=metrics["valid_accuracy"],
            lr=metrics["learning_rate"],
            epoch=step,
        )

    def close(self):

        self.mlflow.end()

        self.tensorboard.close()