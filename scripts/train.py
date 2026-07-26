from src.utils.config import config

# from src.models.simple_cnn import SimpleCNN
# from src.models.resnet import MiniResNet
# from src.models.resnet18 import MedicalResNet18
# from src.models.efficientnet import MedicalEfficientNet
from src.models.vit import VisionTransformer

from src.data.datamodule import PCamDataModule

from src.training.trainer import Trainer

dm = PCamDataModule()

# model = MiniResNet()
# model = MedicalResNet18()
# model = MedicalEfficientNet()
model = VisionTransformer()

trainer = Trainer(

    model=model,

    datamodule=dm,

    config=config,

)

trainer.train()