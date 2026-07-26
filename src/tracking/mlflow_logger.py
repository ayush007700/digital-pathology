"""
MLflow Logger

Author: Ayush Raj
"""

import mlflow


class MLflowLogger:

    def __init__(

        self,

        experiment_name,

    ):

        mlflow.set_experiment(
            experiment_name
        )

        mlflow.start_run()

    def log_params(self, params):

        mlflow.log_params(params)

    def log_metrics(

        self,

        metrics,

        step,

    ):

        mlflow.log_metrics(

            metrics,

            step=step,

        )

    def end(self):

        mlflow.end_run()