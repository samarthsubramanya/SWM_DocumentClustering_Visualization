FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install \
    --default-timeout=300 \
    --retries=10 \
    --no-cache-dir \
    -r requirements.txt

COPY src ./src
COPY swm ./swm
COPY ontology ./ontology
COPY ui ./ui

EXPOSE 8000 8501
