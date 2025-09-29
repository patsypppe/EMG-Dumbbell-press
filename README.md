# EMG Signal Feature Analysis

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)]()
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-informational.svg)]()

This repository provides two Jupyter notebooks for EMG analytics:
- EMG.ipynb: Feature engineering and labeling over time (RMS, skewness, kurtosis, area metrics, differences, scaled scores, Good/Bad labels).
- EMG_TEST.ipynb: CSV ingestion, cleaning, feature calculation, visualization, and baseline ML scaffolding (scaling, train/test splits, RandomForest/XGBoost/NN optional).

## Table of contents
- Overview
- Project structure
- Environment
- Installation
- Usage
- Configuration
- Expected outputs
- Troubleshooting
- License

## Overview
- EMG.ipynb constructs a time-indexed feature set for multiple channels (e.g., deltoid and pectoral), including RMS, skewness, kurtosis, and area metrics, then derives a difference and scaled difference to assign Good/Bad quality labels across the timeline.  
- EMG_TEST.ipynb demonstrates an end‑to‑end CSV workflow: read signal data (e.g., PRANAV2_256.csv), clean/rename columns, compute RMS and related statistics, visualize trends, and (optionally) run ML baselines with standard scaling and train/test splits.

## Project structure
- EMG.ipynb — Feature engineering (RMS, skewness, kurtosis), area metrics (del_area, pec_area), diff and scaled difference, Good/Bad labeling, and tabular previews.  
- EMG_TEST.ipynb — CSV ingestion, preprocessing, feature computation, exploratory plots, and optional ML baseline scaffolding (RandomForest, XGBoost, simple neural nets).  
- requirements.txt (optional) — If included, install to ensure consistent runtime.

## Environment
- Python 3.9+ recommended.  
- Jupyter Notebook or JupyterLab for interactive execution.  
- Common scientific stack: numpy, pandas, scipy, matplotlib, seaborn, scikit-learn.  
- Optional: xgboost and tensorflow if running the corresponding ML baseline cells in EMG_TEST.ipynb.

## Installation
1) Create and activate a virtual environment:
- python -m venv .venv
- source .venv/bin/activate    # macOS/Linux
- .venv\Scripts\activate       # Windows

2) Install dependencies (choose one):
- If requirements.txt is present:
  - pip install -r requirements.txt
- Minimal setup:
  - pip install jupyter numpy pandas scipy matplotlib seaborn scikit-learn
- Optional baselines:
  - pip install xgboost tensorflow

## Usage
- Launch Jupyter:
  - jupyter notebook
- Open and run EMG.ipynb to:
  - Load EMG time-series features.
  - Compute per‑channel RMS, skewness, kurtosis.
  - Compute area metrics (del_area, pec_area), diff, and scaled_difference.
  - Generate Good/Bad labels and preview the first/last rows.
- Open and run EMG_TEST.ipynb to:
  - Set the CSV path (e.g., PRANAV2_256.csv) in the designated cell.
  - Clean/rename columns as needed.
  - Compute RMS and other descriptive features.
  - Visualize trends (time series, distributions).
  - Optionally perform train/test splits, scaling, and fit baseline models.

## Configuration
- File paths: Update the CSV file path variable in EMG_TEST.ipynb.  
- Sampling/windowing: Adjust window sizes for RMS/skewness/kurtosis to match your sampling frequency and analysis needs.  
- Labeling thresholds: In EMG.ipynb, tune scaled_difference normalization and thresholds to calibrate Good/Bad sensitivity.  
- Channels: Extend beyond current channels by duplicating feature steps and updating downstream code to include new columns.

## Expected outputs
- From EMG.ipynb:
  - A time-indexed feature table with columns similar to:
    - time, del_area, pec_area, rms_del, rms_pec, skewness_del, skewness_pec, kurtosis_del, kurtosis_pec
  - A labeled table adding:
    - diff, scaled_difference, labels (e.g., Good/Bad)
- From EMG_TEST.ipynb:
  - A cleaned DataFrame from the CSV with computed features (e.g., RMS columns).
  - Visualizations for quick inspection of temporal trends and distributions.
  - Optional baseline metrics from classical models if those cells are executed.

## Troubleshooting
- NaNs or misaligned shapes:
  - Ensure all arrays align on the time index; drop or impute missing values before feature concatenation.  
- Label imbalance:
  - Revisit scaled_difference normalization and labeling thresholds; consider robust scaling or percentile‑based thresholds.  
- Plot errors or missing backends:
  - Upgrade matplotlib/seaborn and restart the kernel after installing new packages.  
- ML library issues:
  - Install xgboost or tensorflow only if running those cells; otherwise comment them out or skip.

## License
No license file is included. Add a LICENSE (e.g., MIT or Apache‑2.0) if distributing.
