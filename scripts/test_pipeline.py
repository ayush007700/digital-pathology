import torch

from src.models.resnet18 import MedicalResNet18

from src.pipeline.pathology_pipeline import (
    DigitalPathologyPipeline,
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = MedicalResNet18(
    pretrained=False,
)

checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location=device,
)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)

pipeline = DigitalPathologyPipeline(

    model,

    device,

)

result = pipeline.run(

    image_path=r"C:\Users\ayush\Downloads\pcam.jpg",

    clinical_question="""
Explain the pathological significance of HER2-positive breast cancer and summarize relevant evidence.
""",

)

print()

print("="*80)

for k,v in result.items():

    print(f"{k}: {v}")