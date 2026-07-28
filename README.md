# Digital Pathology Copilot

Production-ready AI system for histopathology image analysis.

## Features

- Image Classification
- Explainability (GradCAM)
- FastAPI Backend
- LangGraph Multi-Agent Workflow
- RAG
- GPT-4.1-mini
- PubMed Tool
- Clinical Report Generation
- Docker
- MLflow
- TensorBoard

---

## Pipeline

Image

↓

Prediction

↓

GradCAM

↓

LangGraph

↓

Clinical Report

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
                 

# Performance Benchmark

CPU

Average Prediction Time: 54 ms

Average GradCAM: 111 ms

Average RAG: 8946 ms

Average Report Generation: 1 ms

Total Pipeline: 9112 ms