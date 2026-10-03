# Training image: PyTorch (CPU build) + MLflow + data tools.
FROM python:3.11-slim

# CPU-only PyTorch from PyTorch's own package index (much smaller than the
# GPU build). Production would use a GPU image on the B200 nodes instead.
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 \
      --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
      mlflow==3.16.1 \
      "boto3>=1.34" \
      "pillow>=10.0" \
      requests

# Bake the pretrained backbone weights into the image, so training pods never
# download anything at runtime: reproducible, and works without internet.
ENV TORCH_HOME=/opt/torch
RUN python -c "from torchvision.models import MobileNet_V3_Large_Weights as W; W.IMAGENET1K_V1.get_state_dict(progress=False)"