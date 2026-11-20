from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import CONFIG


def image_transform(training=False, size=CONFIG["image_size"]):
    spatial = [transforms.RandomHorizontalFlip(), transforms.RandomRotation(12)] if training else []
    return transforms.Compose([
        transforms.Resize((size, size)), *spatial, transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])


class GlazeImages(Dataset):
    def __init__(self, frame, training=False, recipe_matrix=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = image_transform(training)
        self.recipe_matrix = recipe_matrix
        missing = [path for path in self.frame.image_path if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[0]}")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        labels = {}
        for task, classes in CONFIG["classes"].items():
            label = row.get(f"image_{task}")
            labels[task] = classes.index(label) if label in classes else -100
        item = {"image": tensor, "labels": labels, "id": row.sample_id}
        if self.recipe_matrix is not None:
            item["recipe"] = torch.as_tensor(self.recipe_matrix[index], dtype=torch.float32)
        return item


def loader(frame, training=False, recipe_matrix=None, batch_size=16):
    return DataLoader(GlazeImages(frame, training, recipe_matrix), batch_size=batch_size,
                      shuffle=training, generator=torch.Generator().manual_seed(CONFIG["seed"]))
