# adhdDetector

Final Project for CS 4100 (Artificial Intelligence), comparing different modeling approaches in detecting brain connectivity patterns to predict ADHD diagnoses.

## Team

- Ankita Sachdeva
- Haiden Busick-Warner
- Yidian Song
- Amir Cohen-Simayof

## Problem

ADHD diagnosis currently relies almost entirely on behavioral questionnaires and clinical interviews - a subjective process that can miss cases or lead to misdiagnosis. Brain connectivity data from fMRI scans offers a more objective alternative. This project asks: **can we predict ADHD diagnosis from brain connectivity information alone?**

## Dataset

We use the **ADHD-200 Preprocessed Connectomes** dataset (Athena pipeline), which contains resting-state fMRI data from **973 individuals** (362 ADHD, 585 typically developing controls, 26 unknown) across 8 international imaging sites.

We specifically use the **CC200 Atlas** (200 Regions of Interest), resulting in 19,900 unique connectivity features per patient.

**Classification Targets:**
1.  **Multi-class**: Control vs. ADHD-Combined vs. ADHD-Inattentive
2.  **Binary**: ADHD (Any type) vs. Neurotypical

## Repo Structure

TODO: Figure out repo structure

```text
adhdDetector/
├── data/
│   ├── adhd200_preprocessed_phenotypics.tsv
│   ├── ADHD200_CC200_TCs_filtfix/      # Extracted Training Data
│   └── ADHD200_CC200_TCs_TestRelease/  # Extracted Test Data
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

The data is too large for GitHub. Team members must download it manually.

### 1. Download the Data
1.  Create a free account on [NITRC.org](https://www.nitrc.org/) and Request Access to the **1000 Functional Connectomes Project**.
2.  Go to the **[Neuro Bureau ADHD-200 Downloads Page](https://www.nitrc.org/frs/?group_id=383)**.
3.  Download exactly these **three files**:

| Package | File | Description |
| :--- | :--- | :--- |
| **ADHD200 Phenotypics** | `adhd200_preprocessed_phenotypics.tsv` | The labels (Diagnosis, Age, Sex, etc.) |
| **ADHD200 Preproc Athena** | `ADHD200_CC200_TCs_filtfix.tar.gz` | Training Set (Brain Scans) |
| **ADHD200 Preproc Athena** | `ADHD200_CC200_TCs_TestRelease.tar` | Test Set (Brain Scans) |

### 2. Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/<your-username>/adhdDetector.git
    cd adhdDetector
    ```
2.  Create a `data/` folder in the root directory.
3.  Extract the downloaded files into `data/`. Your directory should look like this:
    ```text
    adhdDetector/
    ├── data/
    │   ├── adhd200_preprocessed_phenotypics.tsv
    │   ├── ADHD200_CC200_TCs_filtfix/      
    │   └── ADHD200_CC200_TCs_TestRelease/  
    ├── ...
    ```
4.  Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter torch
    ```

## References

- Bellec, P., et al. (2017). The Neuro Bureau ADHD-200 Preprocessed repository. *NeuroImage, 144*(B), 275-286.
- Brown, M.R., et al. (2012). ADHD-200 Global Competition: Diagnosing ADHD using personal characteristic data can outperform resting state fMRI measurements. *Frontiers in Systems Neuroscience*.
- Zhao, K., et al. (2022). A dynamic graph convolutional neural network framework reveals new insights into connectome dysfunctions in ADHD. *NeuroImage*.
