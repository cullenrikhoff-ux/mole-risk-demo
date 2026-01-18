# Day 3: Training Baseline Classifier (Colab)

These steps train a binary melanoma classifier from manifest CSVs with columns
`filepath,label`.

## Install dependencies
```bash
pip install torch torchvision timm scikit-learn matplotlib pillow pandas
```

## Run training
```bash
python training/train_classifier.py \
  --train_csv docs/manifests/train_manifest.csv \
  --val_csv docs/manifests/val_manifest.csv \
  --test_csv docs/manifests/test_manifest.csv \
  --epochs 8 \
  --batch_size 32 \
  --img_size 224 \
  --model_name efficientnet_b0 \
  --out_dir results/day3 \
  --model_out models/classifier_v1.pt
```

## Outputs
- `results/day3/metrics.json` (AUC + confusion matrix stats)
- `results/day3/confusion_matrix.png`
- `models/classifier_v1.pt` (best model by val AUC)
