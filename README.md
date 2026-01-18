# Mole Risk Screening Demo (ML Prototype)

This is a portfolio ML project that estimates a **risk percentage** for skin lesions using models trained on **public dermatology datasets**.
It is a **screening prototype** and **not medical advice**.

## Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Quick checks
```powershell
python training\check_manifests.py
```

## Planned Features
- Photo upload or camera capture (Streamlit)
- Image quality checks (blur/brightness)
- Risk % output (binary classifier)
- Explainability heatmap (Grad-CAM)

## Status
Day 1: Project setup, environment, dataset manifest scaffold, cloud GPU workspace
Day 2: Created Kaggle manifests (train/val/test) with binary melanoma label and committed them.
Day 3: Added baseline EfficientNet training pipeline with metrics + confusion matrix outputs.
