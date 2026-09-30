FROM python:3.12.13-slim

ENV PYTHONBUFFERED=1
ENV PYTHONBYTECODE=1

WORKDIR /app

COPY requirements_script_training.txt .

RUN pip install --no-cache-dir -r requirements_script_training.txt

COPY scripts/train.py ./scripts/train.py
COPY scripts/constant.py ./scripts/constant.py
COPY scripts/register_model.py ./scripts/register_model.py
COPY scripts/config.yaml ./scripts/config.yaml

COPY data/heart.csv ./data/heart.csv

CMD ["bash", "-c", "python scripts/train.py && python scripts/register_model.py"]
