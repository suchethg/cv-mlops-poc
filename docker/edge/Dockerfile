# Edge inference server: ONNX Runtime + a small HTTP API. No PyTorch, no
# training code: the device only needs what it takes to run the model.
FROM python:3.11-slim

RUN pip install --no-cache-dir \
      onnxruntime==1.30.0 \
      mlflow-skinny==3.16.1 \
      "fastapi>=0.115" \
      "uvicorn>=0.30" \
      "pillow>=10.0" \
      numpy

WORKDIR /app
COPY server.py /app/server.py
EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]