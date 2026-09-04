"""ViT–RNN model for patient-level preterm birth risk prediction."""

import timm
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class CervicalUltrasoundViTRNN(nn.Module):
    """Extracts image features with ViT and aggregates them with a BiLSTM."""

    def __init__(
        self,
        backbone: str = "vit_base_patch16_384",
        hidden_size: int = 128,
        rnn_layers: int = 2,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # The original notebooks used the 1,000-dimensional ImageNet head as
        # the per-image representation passed to the recurrent module.
        self.image_encoder = timm.create_model(backbone, pretrained=pretrained)
        feature_dim = self.image_encoder.num_classes
        self.sequence_encoder = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images: torch.Tensor, lengths: torch.Tensor):
        batch_size, steps, channels, height, width = images.shape
        flattened = images.reshape(batch_size * steps, channels, height, width)
        features = self.image_encoder(flattened)
        features = features.reshape(batch_size, steps, -1)

        packed = pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.sequence_encoder(packed)
        patient_features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        logits = self.classifier(patient_features)
        imaging_score = torch.softmax(logits, dim=1)[:, 1]
        return logits, imaging_score
