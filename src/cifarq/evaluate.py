"""Accuracy, one function for training, the champion and the model card. ``OBJECTIVE`` is the metric searches optimise
(on the validation split, ``DIRECTION``) and the champion is chosen by; ``KEY`` is its name in ``train``'s result."""
from __future__ import annotations

OBJECTIVE = 'acc'
DIRECTION = 'maximize'
KEY = f'val_{OBJECTIVE}'


def accuracy(net, x, y, norm, tta_flip: bool = False, batch: int = 1000) -> float:
    """The share of uint8 images ``x`` (NCHW, on the network's device) whose likeliest class is ``y``, normalised by
    ``norm``, under bf16 autocast; each image's logits averaged with its mirror's when ``tta_flip``. The network is
    left in training mode."""
    import torch
    net.eval()
    correct = 0
    with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16):
        for i in range(0, len(y), batch):
            xb = norm(x[i:i + batch].float() / 255)
            logits = net(xb)
            if tta_flip:                                     # the mirror's logits too
                logits = logits + net(xb.flip(3))
            correct += (logits.argmax(1) == y[i:i + batch]).sum().item()
    net.train()
    return correct / len(y)
