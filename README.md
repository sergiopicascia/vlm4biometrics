# Zero-Shot Evaluation of Vision-Language Models on Biometric Tasks

This repository contains the code and analysis for evaluating the zero-shot capabilities of **Gemma 3** vision-language models (4B, 12B, and 27B) on a diverse set of biometric datasets and tasks.

## Overview

Biometric recognition tasks are essential in various applications, including identity verification, forensics, and user authentication. This project explores whether general-purpose vision-language models can effectively perform such tasks in a zero-shot setting, i.e., without task-specific fine-tuning.

### Evaluated Models
- [**Gemma 3 - 4B**](https://huggingface.co/google/gemma-3-4b-it)
- [**Gemma 3 - 12B**](https://huggingface.co/google/gemma-3-12b-it)
- [**Gemma 3 - 27B**](https://huggingface.co/google/gemma-3-27b-it)

### Datasets and Tasks
- **LFW**: Face verification
- **AgeDB**: Age estimation
- **CASIA-Iris-Thousand**: Iris recognition
- **FVC**: Fingerprint verification
- **CelebA**: Attribute classification

---

## Repository Structure

```bash
.
├── experiments.sh             # Script to reproduce the experiments
├── results_analysis.ipynb     # Summary of performance metrics and visualizations
├── requirements.txt           # Required python packages 
├── scripts/                   # Code for the analysis of each dataset
└── README.md
```

## Getting Started
### Requirements
Install dependencies using:

```bash
pip install -r requirements.txt
```
Note: This project assumes access to the Gemma 3 models (4B, 12B, 27B). Ensure these are available locally or accessible through the appropriate model hub.

### Dataset Preparation
Datasets have been downloaded from the following sources:

Dataset	Link
- [LFW](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
- [AgeDB](https://www.kaggle.com/datasets/nitingandhi/agedb-database)
- [CASIA-Iris-Thousand](https://www.kaggle.com/datasets/sondosaabed/casia-iris-thousand)
- [FVC](http://bias.csr.unibo.it/fvc2004/default.asp)
- [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

After downloading, place the datasets in a `datasets/` folder at the project root, organized as required by the experimental scripts.

### Running Experiments
To reproduce the results from the paper, run:

```bash
bash experiments.sh
```
This script runs zero-shot inference tasks using the three Gemma 3 model variants across the five biometric datasets.

### Analyzing Results
Use the notebook:

```bash
results_analysis.ipynb
```

