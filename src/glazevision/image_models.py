import torch
from torch import nn

from .config import CONFIG


def image_encoder(name, pretrained=True):
    if name == "tiny":
        return nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        ), 64
    import timm

    encoder = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                **({"img_size": CONFIG["image_size"]} if "dinov2" in name else {}))
    return encoder, encoder.num_features


class ImageClassifier(nn.Module):
    def __init__(self, name="tiny", pretrained=True, freeze=False):
        super().__init__()
        self.encoder, width = image_encoder(name, pretrained)
        self.encoder.requires_grad_(not freeze)
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(width, len(CONFIG["classes"]["surface"])))

    def forward(self, image):
        return self.head(self.encoder(image))


def load_image_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ImageClassifier(checkpoint["backbone"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()
