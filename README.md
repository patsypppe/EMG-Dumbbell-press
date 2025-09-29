# EMG Signal Analytics + Genetic Algorithm Summarization

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)]()
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-informational.svg)]()

This repository contains two components:
- EMG feature analysis and labeling in Jupyter notebooks (EMG.ipynb, EMG_TEST.ipynb).
- A research prototype that applies a simple genetic algorithm to perturb selected weights of a DistilBART-CNN summarization model (GeneticAlgo.py).

Use the notebooks for EMG data ingestion, feature computation (RMS, skewness, kurtosis), area metrics, and Good/Bad labeling; use the Python script to evaluate and lightly optimize a pretrained summarizer on a validation subset of CNN/DailyMail.

## Table of contents
- Overview
- Project structure
- Environments
- Installation
- EMG workflow
- EMG_TEST workflow
- Summarization + GA workflow
- Configuration
- Expected outputs
- Troubleshooting
- License
- Citation

## Overview
- EMG.ipynb: Processes EMG time series, computes per-channel features for deltoid and pectoral signals, constructs area metrics, derives differences and a scaled score, and assigns Good/Bad labels across time.  
- EMG_TEST.ipynb: Reads a CSV (e.g., PRANAV2_256.csv), cleans columns, computes RMS and other statistics, and includes scaffolding to train/test ML baselines (RandomForest, XGBoost, simple neural nets) with optional scaling and grid search.  
- GeneticAlgo.py: Loads a pretrained DistilBART-CNN model and a small validation subset of CNN/DailyMail, computes ROUGE and METEOR, then runs a small genetic algorithm over targeted parameters to improve fitness signals while demonstrating inference and reporting example summaries.

## Project structure
- EMG.ipynb — EMG feature engineering (RMS, skewness, kurtosis), area metrics (del_area, pec_area), diff and scaled difference, Good/Bad labeling, and table previews.  
- EMG_TEST.ipynb — EMG CSV ingestion, data cleaning, time/value plots, RMS feature computation, train/test splits, StandardScaler, RandomForest/XGBoost baselines, and result inspection.  
- GeneticAlgo.py — DistilBART-CNN summarization evaluation, ROUGE/METEOR metrics, genetic algorithm (tournament selection, uniform crossover, Gaussian mutation), and example generation.  
- requirements.txt — Core dependencies for the summarization pipeline and general Python data/ML work.

## Environments
- Python 3.9+ is recommended.  
- Jupyter Notebook/Lab for running EMG.ipynb and EMG_TEST.ipynb.  
- Optional GPU (CUDA) for acceleration in the summarization script.

## Installation
1) Create and activate a virtual environment:
- python -m venv .venv
- source .venv/bin/activate    # macOS/Linux
- .venv\Scripts\activate       # Windows

2) Install dependencies:
- pip install -r requirements.txt

Notes:
- requirements.txt includes: torch, transformers, datasets, evaluate, rouge_score, numpy.
- For GPU acceleration, install a CUDA-compatible PyTorch build per the official instructions before the rest of the packages.

## EMG workflow
Use this when analyzing EM
