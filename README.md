# DSAI3202
**Maymona Mustafa, 60306027**

---

# Assignment 2: Model Training & Automation with Azure ML

---

## Objective
Transition from feature engineering (Lab 4) into a full MLOps training and deployment workflow:
- Train a model on Azure ML compute clusters using engineered features
- Track experiments with MLflow
- Automate training via Azure DevOps CI pipeline
- Tune hyperparameters with Azure ML Sweep Jobs
- Deploy the model as a managed online endpoint
- Invoke the endpoint with the deployment dataset to simulate production

---

## Tools & Technologies
- Git & GitHub
- Python (pandas, scikit-learn, nltk, sentence-transformers)
- Azure Databricks
- Azure Data Lake Storage Gen2
- Azure Machine Learning (Components, Pipelines, Feature Store)
- Azure ML CLI
- Azure DevOps

---

## Project Structure
```bash
60306027-Cloud_Win-26/
├── src/
│   ├── train.py              # Training script (features, model, MLflow logging)
│   ├── score.py              # Scoring script for online endpoint
│   └── invoke_endpoint.py    # Script to invoke endpoint with deployment dataset
├── jobs/
│   ├── train_job.yml         # Azure ML command job definition
│   ├── sweep_job.yml         # Azure ML sweep job for hyperparameter tuning
│   └── deployment.yml        # Managed online deployment configuration
├── env/
│   ├── conda.yml             # Training environment dependencies
│   └── inference_conda.yml   # Inference environment dependencies
├── azure-pipelines.yml       # Azure DevOps CI pipeline
└── README.md
```

---

## Dataset Splits
| Split | Size | Purpose |
|-------|------|---------|
| Train | 60% | Model learning |
| Validation | 15% | Hyperparameter tuning |
| Test | 15% | Final offline evaluation |
| Deployment | 10% | Simulates production data (most recent reviews by `review_year`) |

All 4 splits registered as Azure ML Data Assets:
- `amazon_review_merged_features_train:1`
- `amazon_review_merged_features_val:1`
- `amazon_review_merged_features_test:1`
- `amazon_review_merged_features_deploy:1`

---

## Model Choice
**Logistic Regression** (`saga` solver, `n_jobs=-1`)

Chosen because:
- Fast to train on high-dimensional sparse+dense feature matrices
- Strong baseline for binary text classification
- Easy to debug and interpret
- Well-suited for CI pipelines where runtime matters

---

## Features Used
All features come directly from the Lab 4 pipeline outputs — no feature engineering is performed in training.

| Feature Group | Columns | Description |
|---------------|---------|-------------|
| SBERT embeddings | `bert_emb_0` to `bert_emb_383` | 384-dim dense semantic vectors |
| TF-IDF | `tfidf_*` | Sparse word frequency features (top 5000 terms) |
| Sentiment | `compound`, `sentiment_positive`, `sentiment_negative`, `sentiment_neutral` | VADER scores |
| Length | `review_length`, `word_count`, `avg_word_length`, `sentence_count` | Review statistics |

**Total feature matrix: ~5,392 columns**

---

## Label Definition
Binary classification:
- `1` (Positive) → `overall >= 4`
- `0` (Negative) → `overall < 4`

---

## MLflow Metrics Logged
For each split (train, val, test):
- `{split}_accuracy`
- `{split}_auc`
- `{split}_f1`
- `{split}_precision`
- `{split}_recall`
- `training_runtime_seconds`

---

## Manual Training Job Results
**Job:** `plum_station_896mwkcsbm`

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | 0.9033 | 0.9033 | 0.9033 |
| AUC | 0.9450 | 0.9450 | 0.9450 |
| F1 | 0.9421 | 0.9421 | 0.9421 |
| Precision | 0.9094 | 0.9094 | 0.9094 |
| Recall | 0.9772 | 0.9772 | 0.9772 |
| Runtime | - | - | 69.05s |

---

## Hyperparameter Tuning (Sweep Job)
**Job:** `yellow_energy_tw6x518766`

Search space:
- `C`: uniform [0.01, 10.0]
- `max_iter`: choice [500, 1000, 2000]

Sampling: random | Trials: 6 | Concurrent: 2
Objective: maximize `val_accuracy`

Best configuration selected from sweep child runs (see Azure ML Studio → Jobs → Sweep experiment).

---

## MLOps Workflow
```
code push → Azure DevOps pipeline → Azure ML training job → MLflow metrics → versioned model → deployed endpoint
```

### Azure DevOps CI Pipeline
- Trigger: push to `assignment2_model_training`
- Automatically submits Azure ML training job and streams logs
- Service connection: `SC-UDST-CCIT-DSAI3202-2`

### Registered Model
- Name: `amazon-review-sentiment-model`
- Version: 2
- Linked to job: `plum_station_896mwkcsbm`

---

## Deployment
- **Endpoint:** `amazon-review-ep-60306027`
- **Deployment:** `blue`
- **Instance:** `Standard_F2s_v2` (1 instance)
- **Auth:** key-based

Endpoint invoked using the deployment dataset (`amazon_review_merged_features_deploy:1`) to simulate production predictions and measure real-world performance.

---



---
### Step 1: GitHub Branch
```bash
git checkout assignment2_model_training
git pull origin assignment2_model_training
```

### Step 2: Configure Azure ML CLI
```bash
az extension add -n ml -y
az extension update -n ml
az login
az configure --defaults group="rg-60306027" workspace="Amazon-Electronics-Lab-60306027"
az ml workspace show
```

---

## Conclusion
This lab demonstrates...

---

