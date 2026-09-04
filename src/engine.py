"""Training and evaluation loops shared by the command-line entry points."""

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


def run_epoch(model, loader, device, criterion, optimizer=None) -> Dict[str, object]:
    training = optimizer is not None
    model.train(training)
    losses: List[float] = []
    labels: List[int] = []
    predictions: List[int] = []
    scores: List[float] = []
    patient_ids: List[str] = []

    for images, target, lengths, batch_ids in loader:
        images, target = images.to(device), target.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits, imaging_score = model(images, lengths)
            loss = criterion(logits, target)
            if training:
                loss.backward()
                optimizer.step()

        losses.extend([float(loss.item())] * target.size(0))
        labels.extend(target.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        scores.extend(imaging_score.detach().cpu().tolist())
        patient_ids.extend(batch_ids)

    metrics: Dict[str, object] = {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "patient_ids": patient_ids,
        "labels": labels,
        "predictions": predictions,
        "imaging_scores": scores,
    }
    metrics["auc"] = roc_auc_score(labels, scores) if len(set(labels)) == 2 else None
    return metrics

