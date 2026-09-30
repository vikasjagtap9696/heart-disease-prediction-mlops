# Heart Disease Prediction — MLops Pipeline

> **MLops learning project** — The main goal of this project is to practice core MLops concepts by building a complete ML pipeline, from training to production serving. The heart disease dataset is just a vehicle: what matters here is understanding how **Docker**, **MLflow**, and **FastAPI** fit together in a real ML workflow.

## What this project covers

**Docker & Docker Compose**
- Containerizing multiple independent Python services, each with its own `Dockerfile`
- Orchestrating services with ordered dependencies (healthchecks, `depends_on`, `condition: service_completed_successfully`)
- Sharing volumes between containers to persist MLflow artifacts

**MLflow**
- Deploying an MLflow tracking server as a standalone service
- Logging experiments: parameters, metrics, and serialized models
- Using the **Model Registry** to version and promote the best model
- Loading a registered model from another service via its URI (`models:/name/version`)

**FastAPI**
- Building a REST API for ML serving with input validation via Pydantic
- Managing the model lifecycle (load on startup, release on shutdown) with `lifespan`
- Exposing `/health` and `/predict` endpoints ready for real deployment

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                         DOCKER COMPOSE                               ║
║                                                                      ║
║   ┌─────────────┐     ┌──────────────────────┐     ┌─────────────┐   ║
║   │    DATA     │     │  (2) Training Script │     │ (3) FastAPI │   ║
║   │  heart.csv  │────>│  8 sklearn models    │     │    :8000    │   ║
║   │ UCI Dataset │     │  config.yaml driven  │     │             │   ║
║   └─────────────┘     │  one-shot container  │     │ GET  /health│   ║
║                       └──────────┬──^────────┘     │ POST/predict│   ║
║                         log runs |  | register     │             │   ║
║                                  |  | best model   └──────^──────┘   ║
║                       ┌──────────v──┴───────────┐         |          ║
║                       │   (1) MLflow Server     │─────────┘          ║
║                       │         :5000           │  load model        ║
║                       │  Experiment tracking    │  models:/name/1    ║
║                       │  Metrics & params logs  │                    ║
║                       │  Model Registry         │                    ║
║                       │  SQLite + mlruns volume │                    ║
║                       └─────────────────────────┘                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

```

**Startup order** — enforced by Docker Compose healthchecks:
1. **MLflow** starts first and must pass its healthcheck before anything else runs
2. **Training script** runs once MLflow is healthy — trains all models, logs to MLflow, registers the best one
3. **FastAPI** starts only after MLflow is healthy *and* the training script has completed successfully — then loads the registered model and serves predictions

---

## Project Structure

```
MLops_project_1/
├── data/
│   └── heart.csv                  # Heart disease dataset (UCI)
├── scripts/
│   ├── train.py                   # Training logic + MLflow logging
│   ├── register_model.py          # Selects & registers the best model
│   ├── constant.py                # Model registry & feature definitions
│   ├── config.yaml                # Experiment config (models & hyperparams)
│   └── Dockerfile
├── API/
│   ├── server.py                  # FastAPI app with /predict endpoint
│   ├── test_request.py            # Manual test script
│   └── Dockerfile
├── mlflow-data/                   # MLflow SQLite database (persisted)
├── Docker-compose.yaml
├── Dockerfile                     # MLflow server image
├── environment.yml                # Conda environment
├── requirements_mlflow.txt
├── requirements_script_training.txt
└── requirements_fasapi.txt
```

---

## Models Trained

The pipeline trains the following classifiers in parallel, all tracked in MLflow:

| Model | Preprocessing strategy |
|---|---|
| Decision Tree | Ordinal encoding, no scaling |
| Random Forest | Ordinal encoding, no scaling |
| Gradient Boosting | Ordinal encoding, no scaling |
| XGBoost | Ordinal encoding, no scaling |
| Logistic Regression | One-hot encoding + StandardScaler |
| SVM (RBF kernel) | One-hot encoding + StandardScaler |
| K-Nearest Neighbors | One-hot encoding + StandardScaler |
| MLP Neural Network | One-hot encoding + StandardScaler |

The best model (ranked by **F1-score**) is automatically registered in the MLflow Model Registry as `heart_disease_prediction`.

---

## Dataset

The project uses the [UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) (`data/heart.csv`).

**Features:** `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

**Target:** `target` — binary (0 = no disease, 1 = disease)

---

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/) and Docker Compose

### Run the full pipeline

```bash
docker compose up --build
```

This single command will:
- Start the MLflow tracking server on `http://localhost:5000`
- Train all models and log results to MLflow
- Register the best model
- Start the prediction API on `http://localhost:8000`

---

## API Usage

### Health check

```bash
curl http://localhost:8000/health
```

### Predict heart disease

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "sex": 1,
    "cp": 0,
    "trestbps": 160,
    "chol": 289,
    "fbs": 0,
    "restecg": 0,
    "thalach": 145,
    "exang": 1,
    "oldpeak": 0.8,
    "slope": 1,
    "ca": 1,
    "thal": 3
  }'
```

**Response:**

```json
{"prediction": 1}
```

`0` = no heart disease, `1` = heart disease

---

## MLflow UI

Once the stack is running, open [http://localhost:5000](http://localhost:5000) to explore:
- All training runs with their metrics (accuracy, precision, recall, F1, ROC-AUC)
- Logged hyperparameters for each model
- The registered model and its versions

---

## Configuration

Edit `scripts/config.yaml` to add models, change hyperparameters, or rename the experiment:

```yaml
name_experiment: heart_disease_prediction

model:
  RandomForestClassifier_v1:
    model_name: 'RandomForestClassifier'
    preproc_strategy: 'tree'
    hyperparameters:
      n_estimators: 100
      max_depth: 5
  # add more models here...
```

---

## Tech Stack

- **Scikit-learn** — model training & preprocessing pipelines
- **XGBoost** — gradient boosting
- **MLflow** — experiment tracking & model registry
- **FastAPI** — REST API for serving predictions
- **Docker Compose** — service orchestration
- **Pydantic** — input validation
#   h e a r t - d i s e a s e - p r e d i c t i o n - m l o p s  
 