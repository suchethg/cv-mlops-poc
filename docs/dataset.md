# Dataset

**CholecSeg8k**: 8,080 laparoscopic cholecystectomy frames from 17 Cholec80
videos, pixel-labeled for 13 classes.

- Source: https://www.kaggle.com/datasets/newslab/cholecseg8k
  (mirror used: Hugging Face `minwoosun/CholecSeg8k`)
- Paper: Hong et al., *CholecSeg8k: A Semantic Segmentation Dataset for
  Laparoscopic Cholecystectomy Based on Cholec80*, arXiv:2012.12453
- License: **CC BY-NC-SA 4.0** (non-commercial, attribution, share-alike)

The data is downloaded locally into `data/` and is **never committed**.

## How the POC uses it

- Each video is treated as one surgical **case** (one patient).
- Each 80-frame clip is sent as one simulated **device upload**
  (`scripts/simulate_device_upload.py`), with fake patient details burned
  into the frames and listed in the manifest, so the ingest flow has real
  PHI-handling work to do.
- Train/test splits are made **by case**, never by frame, to avoid leakage.