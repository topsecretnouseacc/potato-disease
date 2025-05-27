import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 7, pretrained: bool = True):
    """
    EfficientNet-B0 üzerine son katmanı değiştirerek sınıflandırıcı kurar.

    Parameters
    ----------
    num_classes : int
        Çıkıştaki sınıf sayısı.
    pretrained : bool
        ImageNet ağırlıklarını kullanılsın mı?

    Returns
    -------
    torch.nn.Module
    """
    model = models.efficientnet_b0(
        weights="IMAGENET1K_V1" if pretrained else None
    )

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
