from tqdm import tqdm
import torch
import torch.nn.functional as F


def run_epoch(model, loader, optimizer, device, train: bool = True):
    """
    Tek bir eğitim veya doğrulama döngüsü çalıştırır.

    Returns
    -------
    avg_loss : float
    accuracy : float   (0-1 arası)
    """
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in tqdm(loader, desc="train" if train else "val", leave=False):
        x, y = x.to(device), y.to(device)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_sum += loss.item() * x.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += x.size(0)

    return loss_sum / total, correct / total
