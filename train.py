"""Train the cervical ultrasound ViT–RNN model."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import CervicalUltrasoundDataset, collate_patient_sequences
from src.engine import run_epoch
from src.model import CervicalUltrasoundViTRNN


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="De-identified CSV manifest")
    parser.add_argument("--data-root", required=True, help="Root containing ultrasound images")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="internal_test")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def make_loader(args, split, training):
    dataset = CervicalUltrasoundDataset(
        manifest_path=args.manifest,
        data_root=args.data_root,
        split=split,
        sequence_length=args.sequence_length,
        training=training,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.num_workers,
        collate_fn=collate_patient_sequences,
        pin_memory=torch.cuda.is_available(),
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CervicalUltrasoundViTRNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_loader = make_loader(args, args.train_split, training=True)
    validation_loader = make_loader(args, args.validation_split, training=False)

    best_balanced_accuracy = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer)
        validation_metrics = run_epoch(model, validation_loader, device, criterion)
        summary = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
            "validation_auc": validation_metrics["auc"],
        }
        history.append(summary)
        print(json.dumps(summary))

        current = float(validation_metrics["balanced_accuracy"])
        if current > best_balanced_accuracy:
            best_balanced_accuracy = current
            torch.save(
                {"model_state": model.state_dict(), "args": vars(args), "epoch": epoch},
                output_dir / "best_model.pth",
            )

    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

