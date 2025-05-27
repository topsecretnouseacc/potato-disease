import random, shutil
from pathlib import Path

random.seed(42)


SRC = Path("data") / "archive"


DST = Path("data") / "Potato Disease"

TRAIN, VAL = DST / "train", DST / "valid"
VAL_RATIO = 0.20     # %20 validation

for cls_dir in SRC.iterdir():
    if not cls_dir.is_dir():
        continue
    (TRAIN / cls_dir.name).mkdir(parents=True, exist_ok=True)
    (VAL   / cls_dir.name).mkdir(parents=True, exist_ok=True)

    imgs = list(cls_dir.glob("*"))
    random.shuffle(imgs)
    split = int(len(imgs) * (1 - VAL_RATIO))
    for img in imgs[:split]:
        shutil.copy(img, TRAIN / cls_dir.name / img.name)
    for img in imgs[split:]:
        shutil.copy(img, VAL / cls_dir.name / img.name)

print("Tamam: ",
      sum(1 for _ in TRAIN.rglob('*.*')), "train   |",
      sum(1 for _ in VAL.rglob('*.*')),   "valid örnek kopyalandı.")
