# Reproducibility Guide — Lab 5

## Prerequisites

### Local Machine
- Python 3.9+
- Git
- Azure CLI with ML extension

### Install Azure CLI ML Extension
```bash
az extension add -n ml -y
az extension update -n ml
az login
```

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## Step-by-Step Reproduction

### 1. Clone the Repository
```bash
git clone https://github.com/Maymona-M/60306027-Cloud_Win-26.git
cd 60306027-Cloud_Win-26
git checkout Lab5
```

### 2. Download the Dataset
- Go to: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- Download item #6 — Turbofan Engine Degradation Simulation
- Extract and place files in `data/raw/`:
```bash
mkdir -p data/raw
cp ~/Downloads/CMAPSSData/*.txt data/raw/
```

### 3. Configure Azure ML
```bash
az configure --defaults group="rg-60306027" workspace="Amazon-Electronics-Lab-60306027"
az ml workspace show
```

### 4. Upload Data to Azure ML
```bash
az ml data create --name turbofan_fd001 --version 1 --path data/raw/ --type uri_folder --description "NASA Turbofan FD001 raw sensor data"
```

### 5. Create Compute Cluster
```bash
az ml compute create --name cpu-cluster --type amlcompute --min-instances 0 --max-instances 2 --size Standard_DS3_v2
```

### 6. Run the Full Pipeline
```bash
az ml job create --file pipelines/pipeline.yml --stream
```

### 7. Monitor the Pipeline
```
https://ml.azure.com → Jobs → turbofan_rul_pipeline
```

---

## Running Notebooks Locally

Run notebooks in this exact order:

### Notebook 01 — Exploration & Preprocessing
```bash
jupyter notebook notebooks/01_explore_and_preprocess.ipynb
```
- Input: `data/raw/train_FD001.txt`
- Output: `data/processed/train_FD001_preprocessed.csv`

### Notebook 02 — Feature Extraction
```bash
jupyter notebook notebooks/02_feature_extraction.ipynb
```
- Input: `data/processed/train_FD001_preprocessed.csv`
- Output: `data/processed/features_with_rul.csv`
- Runtime: ~202 seconds

### Notebook 03 — Feature Selection
```bash
jupyter notebook notebooks/03_feature_selection.ipynb
```
- Input: `data/processed/features_with_rul.csv`
- Output: `data/processed/features_final.csv`
- Runtime: ~94 seconds (GA)

### Notebook 04 — Model Training
```bash
jupyter notebook notebooks/04_model_and_evaluation.ipynb
```
- Input: `data/processed/features_final.csv`
- Output: `models/best_model.pkl`, `outputs/final_metrics.json`

---

## Expected Outputs

| File | Description |
|------|-------------|
| `data/processed/train_FD001_preprocessed.csv` | Clean sensor data with RUL |
| `data/processed/features_with_rul.csv` | 777 tsfresh features per engine |
| `data/processed/features_final.csv` | Final 8 selected features |
| `models/best_model.pkl` | Trained Gradient Boosting model |
| `outputs/final_metrics.json` | RMSE scores for all models |
| `outputs/selected_features.json` | Names of 8 selected features |
| `outputs/model_comparison.png` | Bar chart of model RMSE |
| `outputs/predicted_vs_actual.png` | Scatter plot of predictions |

---

## Expected Results

| Metric | Value |
|--------|-------|
| Features extracted | 777 |
| Features after GA | 8 |
| Best model | Gradient Boosting |
| CV RMSE | 6.35 cycles |
| Extraction runtime | 202.4s |
| GA runtime | 94.2s |

---

## Troubleshooting

**tsfresh import error:**
```bash
pip install tsfresh --upgrade
```

**LightGBM special character error:**
- Feature names are auto-cleaned in `train_evaluate.py`
- No action needed

**n_jobs=-1 error on Windows:**
- Use `n_jobs=0` instead in `feature_extraction.py`

**Azure compute not found:**
```bash
az ml compute list
```