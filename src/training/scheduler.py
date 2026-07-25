"""
Learning Rate Scheduler Factory
"""

from torch.optim.lr_scheduler import StepLR


def get_scheduler(optimizer, config):

    scheduler_name = config.get("training", "scheduler")

    if scheduler_name == "step":

        return StepLR(
            optimizer,
            step_size=config.get("scheduler", "step_size"),
            gamma=config.get("scheduler", "gamma"),
        )

    return None