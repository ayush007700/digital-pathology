"""
Trainer

Author: Ayush Raj
"""

import torch

from src.training.losses import get_loss
from src.training.optimizer import get_optimizer
from src.training.metrics import accuracy
from src.utils.device import get_device
from src.training.checkpoint import CheckpointManager
# from src.tracking.mlflow_logger import MLflowLogger
from src.training.scheduler import get_scheduler
from src.training.early_stopping import EarlyStopping
from torch.amp import GradScaler
from torch.amp import autocast
# from src.tracking.tensorboard_logger import TensorBoardLogger
from src.tracking.logger import Logger


class Trainer:

    def __init__(
        self,
        model,
        datamodule,
        config,
    ):

        self.model = model

        self.datamodule = datamodule

        self.config = config

        self.device = get_device()

        self.model.to(self.device)

        self.checkpoint = CheckpointManager()

        self.loss_fn = get_loss(
            config.get("training", "loss")
        )

        self.optimizer = get_optimizer(
            model=self.model,
            optimizer_name=config.get(
                "training",
                "optimizer",
            ),
            learning_rate=config.get(
                "training",
                "learning_rate",
            ),
        )

        self.train_loader = datamodule.train_dataloader()

        self.valid_loader = datamodule.valid_dataloader()

    #    # MLflow Logger Initialization
    #     self.logger = MLflowLogger("Digital Pathology")

        # Log training hyperparameters
        # self.logger.log_params(
        #     {
        #         "epochs": config.get(
        #             "training",
        #             "epochs",
        #         ),
        #         "learning_rate": config.get(
        #             "training",
        #             "learning_rate",
        #         ),
        #         "optimizer": config.get(
        #             "training",
        #             "optimizer",
        #         ),
        #     }
        # )

        self.logger.log_params(
            {
                "epochs": config.get(
                    "training",
                    "epochs",
                ),
                "learning_rate": config.get(
                    "training",
                    "learning_rate",
                ),
                "optimizer": config.get(
                    "training",
                    "optimizer",
                ),
                "scheduler": config.get(
                    "training",
                    "scheduler",
                ),
                "batch_size": config.get(
                    "dataset",
                    "batch_size",
                ),
            }
        )

        self.scheduler = get_scheduler(
            self.optimizer,
            config,
        )

        self.early_stopping = EarlyStopping(

            patience=config.get(

                "training",

                "patience",

            )

        )

        self.use_amp = (
            self.config.get("training", "mixed_precision")
            and self.device.type == "cuda"
        )

        self.scaler = GradScaler(
            enabled=self.use_amp
        )

        # self.tb_logger = TensorBoardLogger()
        self.logger = Logger(
            experiment_name="Digital Pathology"
        )

    def train_one_epoch(self):

        self.model.train()

        total_loss = 0

        total_acc = 0

        for images, labels in self.train_loader:

            images = images.to(self.device)

            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):

                outputs = self.model(images)

                loss = self.loss_fn(
                    outputs,
                    labels,
                )

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.get(
                    "training",
                    "gradient_clip",
                ),
            )

            self.scaler.step(self.optimizer)

            self.scaler.update()

            total_loss += loss.item()

            total_acc += accuracy(
                outputs,
                labels,
            )

        return (
            total_loss / len(self.train_loader),
            total_acc / len(self.train_loader),
        )
        
    @torch.no_grad()
    def validate(self):

        self.model.eval()

        total_loss = 0

        total_acc = 0

        for images, labels in self.valid_loader:

            images = images.to(self.device)

            labels = labels.to(self.device)

            outputs = self.model(images)

            loss = self.loss_fn(
                outputs,
                labels,
            )

            total_loss += loss.item()

            total_acc += accuracy(
                outputs,
                labels,
            )

        return (
            total_loss / len(self.valid_loader),
            total_acc / len(self.valid_loader),
        )
        
    def train(self):

        epochs = self.config.get(
            "training",
            "epochs",
        )

        for epoch in range(epochs):

            train_loss, train_acc = self.train_one_epoch()

            val_loss, val_acc = self.validate()

            self.checkpoint.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                accuracy=val_acc,
            )

            # # Log metrics to MLflow
            # self.logger.log_metrics(
            #     {
            #         "train_loss": train_loss,
            #         "train_accuracy": train_acc,
            #         "valid_loss": val_loss,
            #         "valid_accuracy": val_acc,
            #     },
            #     step=epoch,
            # )

            if self.scheduler is not None:
                self.scheduler.step()

            self.early_stopping.step(
                val_loss
            )

            if self.early_stopping.should_stop:

                print()

                print("Early stopping triggered.")

                break

            current_lr = self.optimizer.param_groups[0]["lr"]

            # self.tb_logger.log_metrics(
            #     train_loss=train_loss,
            #     val_loss=val_loss,
            #     train_acc=train_acc,
            #     val_acc=val_acc,
            #     lr=current_lr,
            #     epoch=epoch,
            # )

            self.logger.log_metrics(
                metrics={
                    "train_loss": train_loss,
                    "valid_loss": val_loss,
                    "train_accuracy": train_acc,
                    "valid_accuracy": val_acc,
                    "learning_rate": current_lr,
                },
                step=epoch,
            )

            print(f"Learning Rate : {current_lr:.6f}")

            print()

            print("=" * 60)

            print(f"Epoch {epoch+1}/{epochs}")

            print(f"Train Loss : {train_loss:.4f}")

            print(f"Train Acc  : {train_acc:.4f}")

            print(f"Valid Loss : {val_loss:.4f}")

            print(f"Valid Acc  : {val_acc:.4f}")

        # End logger session after all epochs
        # self.logger.end()
        # self.tb_logger.close()
        self.logger.close()

    # def load(self, model, optimizer, checkpoint_path):

    #     checkpoint = torch.load(
    #         checkpoint_path,
    #         map_location="cpu",
    #     )

    #     model.load_state_dict(
    #         checkpoint["model_state_dict"]
    #     )

    #     optimizer.load_state_dict(
    #         checkpoint["optimizer_state_dict"]
    #     )

    #     print(
    #         f"Loaded checkpoint from epoch "
    #         f"{checkpoint['epoch']}"
    #     )

    #     return checkpoint["epoch"]