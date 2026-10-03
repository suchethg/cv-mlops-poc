# Runtime image for Metaflow @kubernetes steps.
# Python must match your local version (3.11), because Metaflow passes data
# between your Mac and pods as Python pickles.
FROM python:3.11-slim

RUN pip install --no-cache-dir \
      mlflow-skinny==3.16.1 \
      "boto3>=1.34" \
      requests \
      "pillow>=10.0"