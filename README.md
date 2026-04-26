# Anime YOLO AI

YOLOv8-based anime character detection project focused on dataset preparation, training, evaluation, API delivery, and MLOps-style experimentation.

## Project Summary

This repository explores anime character detection as an end-to-end computer vision workflow rather than just a single training notebook.

The public repo is structured around:

- dataset collection and filtering
- pseudo-label generation
- YOLOv8 training and evaluation
- experiment tracking
- API deployment
- load testing and batch inference

## Core Problem

The project targets detection of five anime characters:

- Naruto
- Luffy
- Gojo
- Goku
- Sukuna

The workflow is built around a SafeBooru-derived dataset and a transfer-learning setup using YOLOv8 pretrained weights.

## What The Repo Demonstrates

- automated dataset preparation scripts
- image validation and filtering
- pseudo-label creation
- YOLOv8 training flow
- evaluation and benchmarking utilities
- FastAPI serving path
- Docker and CI-oriented structure

## Main Repo Areas

```text
.github/workflows/   CI and automation
api/                 FastAPI inference service
src/                 dataset, training, evaluation, and inference scripts
data/                dataset staging
runs/                experiment and inference outputs
```

## Why This Repo Is Useful

This repo is a strong supporting computer vision project because it shows more than model training alone. It demonstrates practical work on the full pipeline around an object detection system.

## Important Notes

- Some large generated or environment-related files existed during development; those are not the core value of the project.
- The best signal for recruiters here is the workflow design, automation intent, and deployment awareness around the model.

## Suggested Recruiter Framing

This project is best described as a computer vision and MLOps portfolio piece that demonstrates:

- object detection fundamentals
- dataset engineering
- training and evaluation workflows
- production-minded API packaging

## Related Focus Areas

This project complements my LLM, MLOps, and backend work by showing applied ML breadth in computer vision.
