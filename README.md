# ❤️ Heart Disease Prediction – MLOps

End-to-End Machine Learning and MLOps project using **Python, Scikit-learn, MLflow, FastAPI, Docker, AWS ECR, AWS EC2 and GitHub Actions CI/CD**.

---

## 🚀 Project Architecture

![Heart Disease Prediction MLOps Architecture](docs/architecture.png)

### Architecture Flow

```text
Developer
    │
    │ Git Push
    ▼
GitHub Repository
    │
    ▼
GitHub Actions
    │
    ├── Checkout Code
    ├── Build Docker Images
    └── Push Images
          │
          ▼
      Amazon ECR
          │
          │ Pull Images
          ▼
       AWS EC2
    Self-Hosted Runner
          │
          ▼
    Docker Compose
      ┌────┼────┐
      ▼    ▼    ▼
   MLflow Training FastAPI
   :5000   Script  :8000
```

---

## 📌 Project Overview

This project implements a complete MLOps pipeline for a **Heart Disease Prediction Machine Learning model**.

### Main Components

- Machine Learning model training
- MLflow experiment tracking
- MLflow model registration
- FastAPI prediction API
- Docker containerization
- Docker Compose
- Amazon ECR
- AWS EC2
- AWS IAM
- GitHub Actions CI/CD
- Self-hosted GitHub Actions runner
- Automated deployment
- API health verification

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Machine Learning |
| Scikit-learn | Model Training |
| Pandas | Data Processing |
| MLflow | Experiment Tracking & Model Registry |
| FastAPI | Prediction REST API |
| Docker | Containerization |
| Docker Compose | Multi-container deployment |
| Git | Version Control |
| GitHub | Source Code Repository |
| GitHub Actions | CI/CD Automation |
| AWS ECR | Docker Image Registry |
| AWS EC2 | Application Deployment |
| AWS IAM | AWS Authentication |

---

## 📂 Project Structure

```text
heart-disease-prediction-mlops/
│
├── API/
│   ├── server.py
│   ├── requirements_fasapi.txt
│   └── Dockerfile
│
├── scripts/
│   ├── train.py
│   ├── register_model.py
│   ├── constant.py
│   ├── config.yaml
│   └── Dockerfile
│
├── data/
│   └── heart.csv
│
├── mlruns/
│
├── mlflow-data/
│
├── Dockerfile
│
├── Docker-compose.yaml
│
├── Docker-compose.prod.yaml
│
├── requirements_mlflow.txt
│
├── requirements_script_training.txt
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── docs/
│   └── architecture.png
│
└── README.md
```

---

## 📊 Dataset

Dataset:

```text
data/heart.csv
```

Dataset shape:

```text
1025 rows × 14 columns
```

### Features

```text
age
sex
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
target
```

### Target

```text
0 → No Heart Disease
1 → Heart Disease
```

---

## 🤖 Machine Learning Models

The training pipeline evaluates multiple classification models:

```text
DecisionTreeClassifier
RandomForestClassifier
LogisticRegression
SVC
GradientBoostingClassifier
XGBClassifier
KNN
MLPClassifier
```

The models are trained, evaluated and tracked using MLflow.

The best-performing model is selected based on the evaluation metric and registered in MLflow.

---

## 📈 MLflow

MLflow is used for:

```text
Experiment Tracking
       ↓
Parameters
       ↓
Metrics
       ↓
Model Artifacts
       ↓
Model Registration
```

Registered model:

```text
heart_disease_prediction
```

Example model URI:

```text
models:/heart_disease_prediction/1
```

MLflow:

```text
http://<EC2_PUBLIC_IP>:5000
```

---

## ⚡ FastAPI

FastAPI provides the prediction REST API.

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/predict` | Heart disease prediction |

### Health Check

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "healthy": "true"
}
```

### Root Endpoint

```bash
curl http://localhost:8000/
```

Response:

```json
{
  "message": "FastAPI server is up."
}
```

### Prediction API

Endpoint:

```text
POST /predict
```

Example request:

```json
{
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
}
```

Example response:

```json
{
  "prediction": 0
}
```

---

## 🐳 Docker Services

The project contains three Docker services.

### MLflow

```text
Port: 5000
```

Purpose:

```text
Experiment Tracking
Model Registry
Model Storage
```

### Training

```text
Training Script
```

Flow:

```text
Load Dataset
     ↓
Train Models
     ↓
Evaluate Models
     ↓
Log Metrics
     ↓
Log Models
     ↓
Register Model
```

### FastAPI

```text
Port: 8000
```

Flow:

```text
Load Registered Model
        ↓
Receive API Request
        ↓
Prediction
        ↓
Return Response
```

---

## 💻 Local Development

Clone repository:

```bash
git clone https://github.com/vikasjagtap9696/heart-disease-prediction-mlops.git
```

Go to project:

```bash
cd heart-disease-prediction-mlops
```

Build Docker services:

```bash
docker-compose -f Docker-compose.yaml build
```

Start services:

```bash
docker-compose -f Docker-compose.yaml up -d
```

Check containers:

```bash
docker-compose -f Docker-compose.yaml ps
```

Check FastAPI:

```bash
curl http://localhost:8000/health
```

---

## ☁️ AWS Deployment

AWS services used:

```text
AWS IAM
AWS ECR
AWS EC2
```

AWS Region:

```text
ap-south-1
```

ECR Repository:

```text
heart-disease-prediction-mlops
```

---

## 🔐 AWS IAM

EC2 uses an IAM Role for AWS authentication.

Role:

```text
MLOpsEC2ECRRole
```

Permission:

```text
AmazonEC2ContainerRegistryPowerUser
```

Verify IAM:

```bash
aws sts get-caller-identity
```

No long-lived AWS access keys are required on the EC2 server.

---

## 🐳 Amazon ECR

Three Docker images are pushed to ECR:

```text
heart-disease-prediction-mlops:fastapi
heart-disease-prediction-mlops:mlflow
heart-disease-prediction-mlops:training
```

The ECR registry URI is automatically obtained by GitHub Actions.

---

## 🏃 GitHub Self-Hosted Runner

The GitHub Actions runner is installed on the AWS EC2 instance.

Runner user:

```text
githubrunner
```

Runner name:

```text
heart-disease-ec2-runner
```

Runner labels:

```text
self-hosted
Linux
X64
```

Runner directory:

```text
/home/ec2-user/heart-disease-prediction-mlops/actions-runner
```

Workflow configuration:

```yaml
runs-on: [self-hosted, Linux, X64]
```

---

## 🔄 CI/CD Pipeline

```text
Developer
    │
    │ git push origin main
    ▼
GitHub Repository
    │
    ▼
GitHub Actions
    │
    ├── Checkout Code
    ├── Verify AWS IAM
    ├── Login to ECR
    ├── Build FastAPI Image
    ├── Build MLflow Image
    ├── Build Training Image
    ├── Push Images to ECR
    ├── Verify ECR Images
    ├── Pull Images on EC2
    ├── Deploy using Docker Compose
    └── Health Check
```

---

## 🔁 Automatic ECR URI

`Docker-compose.prod.yaml` does not contain a hardcoded AWS account ID.

FastAPI:

```yaml
image: ${ECR_URI}/heart-disease-prediction-mlops:fastapi
```

MLflow:

```yaml
image: ${ECR_URI}/heart-disease-prediction-mlops:mlflow
```

Training:

```yaml
image: ${ECR_URI}/heart-disease-prediction-mlops:training
```

GitHub Actions automatically sets:

```bash
export ECR_URI=${{ steps.login-ecr.outputs.registry }}
```

Therefore, the AWS account-specific ECR URI does not need to be manually changed in the Compose file.

---

## 🚀 Production Deployment

Production Compose file:

```text
Docker-compose.prod.yaml
```

Deployment:

```bash
cd /home/ec2-user/heart-disease-prediction-mlops

export ECR_URI=${{ steps.login-ecr.outputs.registry }}

docker-compose -f Docker-compose.prod.yaml pull

docker-compose -f Docker-compose.prod.yaml up -d --force-recreate
```

---

## ❤️ Application Flow

```text
Heart Disease Dataset
        │
        ▼
Model Training
        │
        ▼
MLflow Experiment Tracking
        │
        ▼
Best Model
        │
        ▼
MLflow Model Registry
        │
        ▼
FastAPI
        │
        ▼
Docker
        │
        ▼
Amazon ECR
        │
        ▼
AWS EC2
        │
        ▼
Prediction API
```

---

## 🔍 Verification

Check Docker containers:

```bash
docker-compose -f Docker-compose.prod.yaml ps
```

Check FastAPI:

```bash
curl http://localhost:8000/health
```

Expected:

```json
{
  "healthy": "true"
}
```

FastAPI:

```text
http://<EC2_PUBLIC_IP>:8000
```

MLflow:

```text
http://<EC2_PUBLIC_IP>:5000
```

---

## 🧪 CI/CD Test

Make a project change:

```bash
git add .
```

Commit:

```bash
git commit -m "Update project"
```

Push:

```bash
git push origin main
```

GitHub Actions automatically performs:

```text
Checkout
    ↓
Docker Build
    ↓
ECR Push
    ↓
ECR Pull
    ↓
Docker Compose Deployment
    ↓
Health Check
```

After CI/CD is working, manual Docker build, push and deployment commands are not required for normal updates.

---

## 📌 Final Result

```text
Code Push
    ↓
GitHub
    ↓
GitHub Actions
    ↓
Self-Hosted EC2 Runner
    ↓
Docker Build
    ↓
Amazon ECR
    ↓
AWS EC2
    ↓
Docker Compose
    ↓
MLflow + Training + FastAPI
    ↓
Heart Disease Prediction API
```

---

## 👨‍💻 Author

**Vikas Jagtap**

GitHub:

https://github.com/vikasjagtap9696

Project Repository:

https://github.com/vikasjagtap9696/heart-disease-prediction-mlops

