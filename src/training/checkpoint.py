"""
Checkpoint Manager

Author: Ayush Raj
"""

from pathlib import Path
import torch


class CheckpointManager:

    def __init__(self, save_dir="checkpoints"):

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_accuracy = 0.0

    def save(
        self,
        model,
        optimizer,
        epoch,
        accuracy,
    ):

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accuracy": accuracy,
        }

        # Always save latest model
        torch.save(
            checkpoint,
            self.save_dir / "latest_model.pth",
        )

        # Save best model
        if accuracy > self.best_accuracy:

            self.best_accuracy = accuracy

            torch.save(
                checkpoint,
                self.save_dir / "best_model.pth",
            )

            print("✅ Best model updated.")