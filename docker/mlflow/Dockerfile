# Mlflow tracking server + model registry server
# Pinned to the same version as the client in requirements.txt

FROM python:3.11-slim

RUN pip install --no-cache-dir \ 
    mlflow==3.16.1 \ 
    "boto3>=1.34"

EXPOSE 5000 
