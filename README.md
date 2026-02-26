# adhdDetector

Final Project for CS4100, comparing different modeling approaches in detecting brain connectivity patterns to predict ADHD diagnoses and determine the superior model.

## Data Setup

The dataset is too large for git. Each team member must download it locally.

### 1. Phenotypic Labels

Download `adhd200_preprocessed_phenotypics.tsv` from the
[ADHD-200 Preprocessed Connectomes Project](https://www.nitrc.org/frs/shownotes.php?release_id=2048)
and save it to:

```
data/raw/phenotypics.tsv
```

### 2. Connectome Time Series

Download the CC200 time-course `.1D` files from the same NITRC project page and place them in:

```
data/raw/connectomes/
```

Your final folder structure should look like:

```
data/
└── raw/
    ├── phenotypics.tsv
    └── connectomes/
        ├── sfnwmrda0010001_session_1_rest_1_cc200_TCs.1D
        ├── snwmrda0010001_session_1_rest_1_cc200_TCs.1D
        └── ... (~2400 files)
```

### 3. Verify

```bash
pip install -r requirements.txt
python src/data_loader.py
```

Expected output:

```
Success! Loaded 770 subjects.
ROIs detected: 190
Feature vector length: 17955 (expected 17955)
Label distribution: {0: 489, 1: 160, 2: 11, 3: 110}
Train: (616, 17955)  Test: (154, 17955)
```

## Project Structure

```
src/
└── data_loader.py   # Loads connectomes, computes correlation matrices, splits data
```
