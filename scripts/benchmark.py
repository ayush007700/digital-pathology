"""
Pipeline performance benchmark.

Measures average prediction, GradCAM, RAG, and report generation latency,
then writes results/benchmark.md.
"""

from pathlib import Path

import torch

from src.models.resnet18 import MedicalResNet18
from src.pipeline.factory import build_pipeline

IMAGE_PATH = r"C:\Users\ayush\Downloads\pcam.jpg"
QUESTION = "Explain the pathological significance of HER2-positive breast cancer."
N_WARMUP = 1
N_RUNS = 5
OUTPUT_PATH = Path("results/benchmark.md")


def load_model(device: torch.device) -> MedicalResNet18:
    model = MedicalResNet18(pretrained=False)
    checkpoint = torch.load(
        "checkpoints/best_model.pth",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    pipeline = build_pipeline(model, device)

    # Warmup (embeddings / CUDA kernels / first LLM call)
    for _ in range(N_WARMUP):
        pipeline.run(IMAGE_PATH, QUESTION, return_timings=True)

    buckets = {
        "prediction_ms": [],
        "gradcam_ms": [],
        "rag_ms": [],
        "report_ms": [],
        "total_ms": [],
    }

    for i in range(N_RUNS):
        result = pipeline.run(IMAGE_PATH, QUESTION, return_timings=True)
        timings = result["timings"]
        for key in buckets:
            buckets[key].append(timings[key])
        print(
            f"run {i + 1}/{N_RUNS}: "
            f"pred={timings['prediction_ms']:.0f}ms "
            f"gradcam={timings['gradcam_ms']:.0f}ms "
            f"rag={timings['rag_ms']:.0f}ms "
            f"report={timings['report_ms']:.0f}ms "
            f"total={timings['total_ms']:.0f}ms"
        )

    device_label = "CUDA" if device.type == "cuda" else "CPU"
    lines = [
        "# Performance Benchmark",
        "",
        device_label,
        "",
        f"Average Prediction Time: {avg(buckets['prediction_ms']):.0f} ms",
        "",
        f"Average GradCAM: {avg(buckets['gradcam_ms']):.0f} ms",
        "",
        f"Average RAG: {avg(buckets['rag_ms']):.0f} ms",
        "",
        f"Average Report Generation: {avg(buckets['report_ms']):.0f} ms",
        "",
        f"Total Pipeline: {avg(buckets['total_ms']):.0f} ms",
        "",
        f"_Runs: {N_RUNS} (warmup: {N_WARMUP}). Image: `{IMAGE_PATH}`_",
        "",
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(OUTPUT_PATH.read_text(encoding="utf-8"))
    print(f"Wrote {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
