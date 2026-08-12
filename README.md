# Brain-Tumor Detection & Automated Screening — YOLO11 Capstone

An end-to-end computer-vision application built with **Ultralytics YOLO11** for one medical
use case: **detecting brain tumors in MRI scans** and running an **automated screening pipeline**
that streams scans, detects tumors, and counts them. Every stage uses the same auto-downloaded
**Brain-Tumor** dataset — no API key, no external service.

## What it does

| Capability | Model / API | Task |
|------------|-------------|------|
| Tumor detection | fine-tuned `best.pt` + `model.predict` | Detection |
| Instance segmentation | `yolo11n-seg.pt` + `model.predict` | Segmentation (beyond detection) |
| Automated screening | `best.pt` + `model.track` over an MRI stream | Tracking + counting |
| Custom training | `yolo11n.pt` + `model.train` | Fine-tuning on Brain-Tumor |
| Evaluation | `model.val` | mAP50 / mAP50-95 / P / R + confusion matrix |
| Deployment | `model.export` | ONNX export + `onnxruntime` check |

## Pipeline
Setup → Train (best.pt) → Detect + Segment → Screen & count (OpenCV) → Evaluate → Export ONNX
- **Model:** YOLO11-nano (detection); YOLO11-nano-seg for the segmentation task.
- **Dataset:** `brain-tumor.yaml` — real MRI tumor-detection data, auto-downloaded by Ultralytics.
- **Training config:** `epochs=30`, `imgsz=640`, `batch=16`, `patience=15`, `seed=0`.

## Results (validation)

| Metric | Value |
|--------|-------|
| mAP50 | 0.44 |
| mAP50-95 | 0.32 |
| Precision | 0.47 |
| Recall | 0.80 |

Recall (0.80) > Precision (0.47) — the model is sensitive and catches most tumors at the cost of
some false alarms, which is the right trade-off for a medical triage system.

## How to run

1. Open `capstone_ultralytics_yolo11.ipynb` in [Google Colab](https://colab.research.google.com/).
2. **Runtime → Run all** (≈ 10–15 minutes). Models and dataset download automatically.

## Training program

Completed under **Computer Vision for Developers with Ultralytics** — **SDAIA Academy**


- Cohort / session dates: _<Aug 09,2026>_
- SDAIA Academy on GitHub: https://github.com/SDAIAAcademy
