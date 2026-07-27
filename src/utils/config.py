from pathlib import Path

import yaml


class Config:

    def __init__(self, config_path="configs/config.yaml"):

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, *keys):

        value = self.config

        for key in keys:
            value = value[key]

        return value

    def get_path(self, key):

        return self.config["paths"][key]

    def get_model(self, key):

        return self.config["model"][key]

    def get_training(self, key):

        return self.config["training"][key]

    def get_deployment(self, key):

        return self.config["deployment"][key]


config = Config()
