"""Send a surgical frame to an edge site and save the segmentation overlay.

Usage (Mac, tunnels open):
  python scripts/edge_predict.py --site boston
  python scripts/edge_predict.py --site denver --frame data/raw/CholecSeg8k/video12/.../frame_X_endo.png
Writes out/<site>-overlay.png: the frame with the predicted classes blended on top.
"""
import argparse
import base64
import glob
import io
import os

import requests
from PIL import Image

PORTS = {"boston": 8081, "denver": 8082}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", choices=PORTS, default="boston")
    p.add_argument("--frame", help="image file (default: a frame from the dataset)")
    args = p.parse_args()

    frame = args.frame or sorted(glob.glob("data/raw/CholecSeg8k/*/*/*_endo.png"))[0]
    url = f"http://localhost:{PORTS[args.site]}"
    with open(frame, "rb") as f:
        r = requests.post(f"{url}/predict", data=f.read(), timeout=30)
    r.raise_for_status()
    result = r.json()

    image = Image.open(frame).convert("RGB")
    mask = Image.open(io.BytesIO(base64.b64decode(result["mask_png_base64"]))).convert("RGB")
    os.makedirs("out", exist_ok=True)
    out = f"out/{args.site}-overlay.png"
    Image.blend(image, mask.resize(image.size), 0.45).save(out)

    print(f"site {result['site']}  model v{result['model_version']}  "
          f"inference {result['latency_ms']} ms")
    for name, share in sorted(result["class_shares"].items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {name:24} {share:6.1%}")
    print(f"overlay saved to {out}  (frame: {frame})")


if __name__ == "__main__":
    main()