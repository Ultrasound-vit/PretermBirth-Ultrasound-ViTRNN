"""Patient-level cervical ultrasound dataset utilities.

The public interface uses a de-identified CSV manifest instead of the original
machine-specific paths and serialized research databases.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def build_transform(image_size: int = 384, training: bool = False):
    operations = [transforms.ToPILImage(), transforms.Resize((image_size, image_size))]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    return transforms.Compose(operations)


class CervicalUltrasoundDataset(Dataset):
    """Loads a variable-length ultrasound sequence for each patient."""

    REQUIRED_COLUMNS = {"patient_id", "image_path", "label", "split"}

    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        split: str,
        sequence_length: int = 100,
        image_size: int = 384,
        training: bool = False,
    ) -> None:
        manifest = pd.read_csv(manifest_path)
        missing = self.REQUIRED_COLUMNS.difference(manifest.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        manifest = manifest.loc[manifest["split"] == split].copy()
        if manifest.empty:
            raise ValueError(f"No rows found for split '{split}'")

        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.transform = build_transform(image_size=image_size, training=training)
        self.records: List[Dict[str, object]] = []

        for patient_id, rows in manifest.groupby("patient_id", sort=True):
            labels = rows["label"].astype(int).unique()
            if len(labels) != 1:
                raise ValueError(f"Patient {patient_id} has inconsistent labels")
            self.records.append(
                {
                    "patient_id": str(patient_id),
                    "label": int(labels[0]),
                    "images": rows["image_path"].astype(str).tolist(),
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_paths: Sequence[str] = record["images"]  # type: ignore[assignment]
        if len(image_paths) > self.sequence_length:
            image_paths = image_paths[: self.sequence_length]

        images = []
        for relative_path in image_paths:
            path = self.data_root / relative_path
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Unable to read ultrasound image: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(self.transform(image))

        if not images:
            raise ValueError(f"Patient {record['patient_id']} has no images")
        return torch.stack(images), int(record["label"]), str(record["patient_id"])


def collate_patient_sequences(batch):
    """Pads patient sequences and returns their original lengths."""

    sequences, labels, patient_ids = zip(*batch)
    lengths = torch.tensor([sequence.shape[0] for sequence in sequences], dtype=torch.long)
    max_length = int(lengths.max())
    channels, height, width = sequences[0].shape[1:]
    padded = torch.zeros(len(sequences), max_length, channels, height, width)
    for index, sequence in enumerate(sequences):
        padded[index, : sequence.shape[0]] = sequence
    return padded, torch.tensor(labels, dtype=torch.long), lengths, list(patient_ids)

