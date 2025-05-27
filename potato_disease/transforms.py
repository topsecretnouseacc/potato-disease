from torchvision import transforms as T


def train_transforms(img_size: int = 224):
    """Veri büyütme + normalizasyon (Eğitim)."""
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])


def val_transforms(img_size: int = 224):
    """Doğrulama / test dönüşümleri (sabit)."""
    return T.Compose([
        T.Resize(img_size + 32),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])
