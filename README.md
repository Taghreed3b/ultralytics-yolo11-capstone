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
