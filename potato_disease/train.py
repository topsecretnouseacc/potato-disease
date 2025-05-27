from pathlib import Path
import torch
from torch.utils.data import DataLoader

from .datasets import PotatoDataset
from .transforms import train_transforms, val_transforms
from .models.cnn import build_model
from .engine import run_epoch


def main() -> None:
    # ----------------------------------------------------------
    # 1. Veri kümesini yükle
    # ----------------------------------------------------------
    data_root = Path("./data") / "Potato Disease"   # prepare_dataset.py sonucu
    train_ds = PotatoDataset(data_root, "train", train_transforms())
    val_ds   = PotatoDataset(data_root, "valid", val_transforms())

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

    # ----------------------------------------------------------
    # 2. Model, optimizör, cihaz
    # ----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(train_ds.class_to_idx)

    model = build_model(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # ----------------------------------------------------------
    # 3. Eğitim döngüsü
    # ----------------------------------------------------------
    best_acc = 0.0
    epochs = 10

    for epoch in range(epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader, None,      device, train=False)

        print(f"E{epoch:02d} | "
              f"train_acc={tr_acc:.3%}  val_acc={vl_acc:.3%}  "
              f"best={best_acc:.3%}")

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), "best.pt")

    print(f"✅  Eğitim bitti • En iyi doğruluk: {best_acc:.2%}")


if __name__ == "__main__":
    main()

