"""Generate patient-level imaging scores from a trained checkpoint."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import CervicalUltrasoundDataset, collate_patient_sequences
from src.engine import run_epoch
from src.model import CervicalUltrasoundViTRNN


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="external_test")
    parser.add_argument("--output", default="outputs/imaging_scores.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CervicalUltrasoundDataset(
        args.manifest, args.data_root, args.split, args.sequence_length, training=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_patient_sequences,
    )
    model = CervicalUltrasoundViTRNN(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = run_epoch(model, loader, device, nn.CrossEntropyLoss())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "patient_id": metrics["patient_ids"],
            "label": metrics["labels"],
            "prediction": metrics["predictions"],
            "imaging_score": metrics["imaging_scores"],
        }
    ).to_csv(output_path, index=False)
    print(f"Saved patient-level results to {output_path}")
    print(
        f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}; "
        f"AUC: {metrics['auc'] if metrics['auc'] is not None else 'not available'}"
    )


if __name__ == "__main__":
    main()

