from pathlib import Path
from typing import Callable, List

from PIL import Image
import torch
from torch.utils.data import Dataset


class PotatoDataset(Dataset):

    def __init__(self, root: Path, split: str, transforms: Callable = None):
        split_dir = root / split          # ör. data/Potato Disease/train
        if not split_dir.exists():
            raise FileNotFoundError(
                f"{split_dir} bulunamadı.\n"
                "‣ prepare_dataset.py betiğini çalıştırıp "
                "train/ ve valid/ klasörlerini oluşturduğundan emin ol."
            )

        # Tüm görselleri bul
        self.items: List[Path] = list(split_dir.glob("*/*.jpg"))
        self.transforms = transforms

        # Sınıf adlarını klasörlerden çıkar
        classes = sorted({p.parent.name for p in self.items})
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[img_path.parent.name]
        if self.transforms:
            img = self.transforms(img)
        return img, label
