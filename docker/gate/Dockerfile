# Release-gate image: the training image plus ONNX export and runtime.
# Built on top of surgseg-train:0.1 so PyTorch and MLflow versions match the
# model being checked exactly.
FROM surgseg-train:0.1

RUN pip install --no-cache-dir \
      onnx==1.23.1 \
      onnxscript==0.7.2 \
      onnxruntime==1.30.0