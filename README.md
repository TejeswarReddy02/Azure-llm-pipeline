# 🚀 LLM Workload Predictor & Automated MLOps Pipeline

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/GitHub_Actions-Automated-2088FF.svg?logo=github-actions)](https://github.com/features/actions)

> **Live Demo:** [https://llm-workload-router-latest.onrender.com]

## 📌 Project Overview
Large Language Models (LLMs) experience severe latency and memory crashes when massive conversational workloads hit standard compute servers. This project is a **Full-Stack Machine Learning solution** that dynamically predicts the token output size of an incoming LLM request based on input context size, time of day, and workload type (Code vs. Chat). 

It utilizes a **Random Forest Regressor ($R^2$ = 0.81)** to instantly route heavy generation tasks to memory-optimized GPUs and lightweight code completions to fast-compute clusters.

Beyond the model, this project features a **fully automated CI/CD MLOps pipeline**. Pushing new CSV trace logs to GitHub automatically triggers data validation, model retraining, Docker container rebuilding, and live cloud deployment with zero downtime.

---

## 🏗️ Architecture & CI/CD Pipeline

This project implements a complete closed-loop MLOps architecture:

1. **Continuous Integration (Data & Training):** - New `AzureLLMInferenceTrace` logs are pushed to the `data/` directory.
   - **GitHub Actions** detects the push and spins up an Ubuntu runner.
   - A **Gatekeeper Script** (`validate_data.py`) strictly checks data integrity (missing values, negative tokens, schema matching).
   - If validation passes, `train_model.py` trains a fresh `.pkl` model.
2. **Continuous Deployment (Docker & Render):**
   - The pipeline immediately executes a `docker build`, baking the newly trained model into a `python:3.11-slim` container along with the FastAPI backend.
   - The image is pushed to the **GitHub Container Registry (GHCR)**.
   - **Render** automatically detects the new image, pulls it, and executes a zero-downtime rolling deployment.

---

## 🛠️ Tech Stack

- **Machine Learning:** Scikit-Learn (Random Forest Regressor), Pandas, NumPy
- **Backend API:** FastAPI, Uvicorn
- **Frontend Dashboard:** HTML5, CSS3, Vanilla JS (Fetch API)
- **DevOps / MLOps:** Docker, GitHub Actions, GitHub Container Registry (GHCR)
- **Cloud Deployment:** Render (Web Services)

---

## 📂 Repository Structure

```text
├── .github/workflows/
│   └── mlops-pipeline.yml     # The automated CI/CD pipeline instructions
├── data/
│   └── *.csv                  # Raw inference trace datasets (trigger pipeline on update)
├── models/
│   └── llm_workload_predictor.pkl # The trained Random Forest model (Auto-generated)
├── scripts/
│   ├── merge_data.py          # Data engineering: joins multiple CSVs
│   ├── validate_data.py       # Gatekeeper: asserts data quality before training
│   └── train_model.py         # ML script: trains model and evaluates R2 score
├── static/
│   └── index.html             # The interactive frontend dashboard
├── app.py                     # FastAPI server hosting the model and static files
├── Dockerfile                 # Container blueprint
├── requirements.txt           # Version-pinned dependencies
└── README.md                  # Project documentation
