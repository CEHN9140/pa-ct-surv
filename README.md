# PA-CT-Surv

Pathology-CT survival prediction model.

## Pretrained Weights

Before training, download the CT pretrained ResNet weights and place them under `model/ct_pretrain/`:

```
model/ct_pretrain/resnet_10_23dataset.pth
model/ct_pretrain/resnet_18_23dataset.pth
```

Contact the authors for weight file access.

## Environment

```bash
conda env create -f environment.yml
# or
pip install -r requirements.txt
```

## Data

Place your dataset CSV files under `data/`. See `dataset.py` for expected format.

## Training

```bash
# Path-only model
python path_train.py

# CT-only model
python ct_train.py

# Distillation training
python distill_train.py

# PACT (Pathology + CT fusion)
python pact_train.py

# Final test
python final_test.py
```
