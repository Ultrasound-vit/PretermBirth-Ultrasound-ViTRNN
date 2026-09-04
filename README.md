# Ultrasound-Based Multimodal Deep Learning for Preterm Birth Prediction

Clean reference implementation for the manuscript **“Ultrasound-Based Deep Learning Multi-Modal Model for Predicting Preterm Birth in Women with a Short Cervix”**, submitted to *Scientific Reports*.

This directory is a consolidated, publication-oriented version of the original research notebooks. The originals are preserved unchanged in the parent directory. Machine-specific paths and legacy disease names have been removed from this version.

## What is included

- `src/dataset.py`: patient-level loading and ultrasound augmentation;
- `src/model.py`: ViT image encoder and bidirectional LSTM sequence encoder;
- `src/engine.py`: shared training and evaluation loop;
- `train.py`: reproducible model training and checkpoint selection;
- `predict.py`: patient-level prediction and imaging-score export;
- `examples/metadata_example.csv`: de-identified input format;
- `requirements.txt`: Python dependencies used by the cleaned implementation.

## Input format

Create a CSV with one row per ultrasound image:

| Column | Description |
|---|---|
| `patient_id` | De-identified patient identifier |
| `image_path` | Image path relative to `--data-root` |
| `label` | `0` for term birth and `1` for preterm birth |
| `split` | `train`, `internal_test` or `external_test` |

See `examples/metadata_example.csv`. Do not place patient names, hospital identifiers or other protected information in the manifest or filenames.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The manuscript experiments used Python 3.8 and PyTorch 1.10.2. A CUDA-capable GPU is recommended.

## Training

```bash
python train.py \
  --manifest /path/to/deidentified_metadata.csv \
  --data-root /path/to/deidentified_ultrasound_images \
  --train-split train \
  --validation-split internal_test \
  --output-dir outputs
```

All paths are supplied at runtime. No local workstation or hospital storage path is embedded in the source code.

## External validation and imaging-score export

```bash
python predict.py \
  --manifest /path/to/deidentified_metadata.csv \
  --data-root /path/to/deidentified_ultrasound_images \
  --checkpoint outputs/best_model.pth \
  --split external_test \
  --output outputs/external_imaging_scores.csv
```

The exported patient-level imaging score can be combined with the clinical predictors described in the manuscript for downstream multimodal logistic regression analysis.

## Implementation notes

- Cervical ultrasound images are resized to 384 × 384 pixels.
- The backbone is `vit_base_patch16_384` initialized from ImageNet weights.
- Patient image sequences are aggregated with a two-layer bidirectional LSTM.
- The recurrent hidden size is 128 and the default maximum sequence length is 100.
- The default learning rate is 0.0001 and the default training duration is 100 epochs.
- Variable-length sequences are padded only within a batch; original lengths are retained for the recurrent encoder.

## Data availability

The clinical dataset and original ultrasound images are not distributed because they contain sensitive medical information. Access to de-identified data is subject to institutional approval, applicable regulations and a data-use agreement.

## Disclaimer

This code is intended for academic research and method evaluation only. It is not a medical device and must not be used for clinical diagnosis or patient management.
