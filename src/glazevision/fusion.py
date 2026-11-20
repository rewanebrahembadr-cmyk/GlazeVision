import torch
from torch import nn

from .config import CONFIG
from .image_models import image_encoder


class FusionModel(nn.Module):
    def __init__(self, recipe_width, backbone="tiny", mode="early", width=128, pretrained=True):
        super().__init__()
        if mode not in {"early", "late", "attention", "image", "tabular"}:
            raise ValueError(f"Unknown fusion mode: {mode}")
        self.mode = mode
        self.encoder, image_width = image_encoder(backbone, pretrained)
        self.visual = nn.Linear(image_width, width)
        self.recipe = nn.Sequential(nn.Linear(recipe_width, width), nn.ReLU(), nn.Dropout(.2))
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.heads = nn.ModuleDict({task: nn.Linear(width * (2 if mode == "early" else 1), len(classes))
                                    for task, classes in CONFIG["classes"].items()})
        self.image_heads = nn.ModuleDict({task: nn.Linear(width, len(classes))
                                          for task, classes in CONFIG["classes"].items()})
        self.recipe_heads = nn.ModuleDict({task: nn.Linear(width, len(classes))
                                           for task, classes in CONFIG["classes"].items()})

    def forward(self, image=None, recipe=None):
        visual = self.visual(self.encoder(image)) if image is not None and self.mode != "tabular" else None
        tabular = self.recipe(recipe) if recipe is not None and self.mode != "image" else None
        if visual is None and tabular is None:
            raise ValueError("Provide an input supported by this model")
        if self.mode == "image":
            return {key: head(visual) for key, head in self.heads.items()}
        if self.mode == "tabular":
            return {key: head(tabular) for key, head in self.heads.items()}
        if visual is None:
            return {key: head(tabular) for key, head in self.recipe_heads.items()}
        if tabular is None:
            return {key: head(visual) for key, head in self.image_heads.items()}
        if self.mode == "late":
            return {key: (self.image_heads[key](visual) + self.recipe_heads[key](tabular)) / 2
                    for key in self.heads}
        if self.mode == "early":
            combined = torch.cat((visual, tabular), dim=1)
        else:
            tokens = torch.stack((visual, tabular), dim=1)
            combined = self.attention(tokens, tokens, tokens, need_weights=False)[0].mean(1)
        return {key: head(combined) for key, head in self.heads.items()}
