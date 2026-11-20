import torch
from torch import nn


def logits_for(model, batch, device):
    image = batch["image"].to(device)
    if "recipe" in batch:
        return model(image, batch["recipe"].to(device))
    return {"surface": model(image)}


def train_epoch(model, batches, optimizer, device, surface_weights=None):
    model.train()
    for batch in batches:
        optimizer.zero_grad()
        outputs = logits_for(model, batch, device)
        predictions = [outputs]
        if getattr(model, "mode", None) in {"early", "attention"}:
            predictions.extend([model(batch["image"].to(device)),
                                model(recipe=batch["recipe"].to(device))])
        losses = []
        for logits in predictions:
            for task, values in logits.items():
                labels = batch["labels"][task].to(device)
                present = labels != -100
                if present.any():
                    weights = surface_weights if task == "surface" else None
                    losses.append(nn.functional.cross_entropy(values[present], labels[present],
                                                              weight=weights))
        if losses:
            torch.stack(losses).mean().backward()
            optimizer.step()


@torch.no_grad()
def collect(model, batches, device, task="surface"):
    model.eval()
    truth, probabilities, ids = [], [], []
    for batch in batches:
        order = batch["labels"][task] != -100
        output = logits_for(model, batch, device)[task].softmax(1).cpu()
        probabilities.extend(output[order].tolist())
        truth.extend(batch["labels"][task][order].tolist())
        ids.extend(sample_id for sample_id, include in zip(batch["id"], order) if include)
    return ids, truth, probabilities
