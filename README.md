# MLEM + PyTorch demo

Demo of usage MLEM + PyTorch


# Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

# Data

Data were taken from https://www.kaggle.com/c/dogs-vs-cats/data


# Run pipeline

## Train stage

```bash
PYTHONPATH=. python src/stages/train.py --config=params.yaml
```
