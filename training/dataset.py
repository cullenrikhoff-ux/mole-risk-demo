from __future__ import annotations

from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ManifestDataset(Dataset):
    """Dataset backed by a CSV with columns: filepath,label."""

    def __init__(
        self,
        csv_path: str,
        transform: transforms.Compose | None = None,
        img_size: int = 224,
    ) -> None:
        self.csv_path = csv_path
        self.transform = transform
        self.img_size = img_size

        df = pd.read_csv(csv_path)
        if "filepath" not in df.columns or "label" not in df.columns:
            raise ValueError("Manifest must include columns: filepath,label")

        labels = pd.to_numeric(df["label"], errors="raise").astype(int)
        self.filepaths = df["filepath"].astype(str).tolist()
        self.labels = labels.tolist()

        self._default_to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.filepaths)

    def _load_image(self, path: str) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"Warning: failed to open {path}: {exc}. Using black image.")
            return Image.new("RGB", (self.img_size, self.img_size), color=(0, 0, 0))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.filepaths[idx]
        label = self.labels[idx]

        image = self._load_image(path)
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = self._default_to_tensor(image)

        label_tensor = torch.tensor([label], dtype=torch.float32)
        return image_tensor, label_tensor
