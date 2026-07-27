# Digital Pathology Copilot

Production-ready AI system for histopathology image analysis.

## Features

- Deep Learning
- Computer Vision
- Tumor Detection
- Tumor Segmentation
- Explainable AI
- RAG
- LLM
- AWS Deployment

## Tech Stack

- PyTorch
- OpenCV
- FastAPI
- LangGraph
- Docker
- AWS

Run
uvicorn src.api.app:app --reload

Open

http://127.0.0.1:8000/docs

Build

docker build -t pathology-copilot .

Run

docker run -p 8000:8000 pathology-copilot

Rebuild the Docker image

Since requirements changed, you must rebuild.

docker compose -f docker/docker-compose.yml down

Then

docker compose -f docker/docker-compose.yml build --no-cache

Then

docker compose -f docker/docker-compose.yml up

Final Architecture:

                 Histopathology Image
                         │
                  Preprocess Image
                         │
                  ResNet18 / ViT
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
 Prediction + Confidence              GradCAM
        │                                 │
        └────────────────┬────────────────┘
                         ▼
                 LangGraph Supervisor
        ┌───────────────┼─────────────────┐
        ▼               ▼                 ▼
   Retriever       PubMed Tool      General Agent
        │               │                 │
        └───────────────┴─────────────────┘
                         ▼
                 Pathology Agent
                         ▼
               Clinical Report Agent
                         ▼
                 Markdown/PDF Report