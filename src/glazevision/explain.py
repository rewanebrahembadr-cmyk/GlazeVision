import torch
from torch import nn


def gradcam(model, image):
    from pytorch_grad_cam import GradCAM

    layers = [layer for layer in model.encoder.modules() if isinstance(layer, nn.Conv2d)]
    if not layers:
        raise ValueError("Select a convolutional backbone or implement a transformer reshape")
    with torch.enable_grad(), GradCAM(model=model, target_layers=[layers[-1]]) as cam:
        return cam(input_tensor=image.detach().requires_grad_(True), targets=None)[0]
