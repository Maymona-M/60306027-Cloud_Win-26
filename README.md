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
- **Git & GitHub** — version control and CI trigger
- **Python** — pandas, numpy, scikit-learn, joblib, mlflow
- **Azure Machine Learning** — compute clusters, command jobs, sweep jobs, managed endpoints, MLflow tracking, model registry, data assets
- **Azure DevOps** — CI pipeline for automated training job submission
- **Azure ML CLI v2** — job submission and resource management

---

## Project Structure
```bash
60306027-Cloud_Win-26/
├── src/
│   ├── train.py              # Training script: loads features, trains model, logs metrics
│   ├── score.py              # Scoring script for managed online endpoint
│   └── invoke_endpoint.py    # Invokes endpoint with deployment dataset
├── jobs/
│   ├── train_job.yml         # Azure ML command job definition
│   ├── sweep_job.yml         # Hyperparameter sweep job definition
│   └── deployment.yml        # Managed online deployment configuration
├── env/
│   ├── conda.yml             # Training environment (minimal, fast build)
│   └── inference_conda.yml   # Inference environment for endpoint container
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

**Total feature matrix: ~5,392 columns per sample**

---

## Label Definition
Binary classification:
- `1` (Positive) → `overall >= 4`
- `0` (Negative) → `overall < 4`

---

## Pipeline Components (from Lab 4)

| Component | Description |
|-----------|-------------|
| `split_dataset` | Splits into train/val/test/deploy using stratified sampling by `review_year` |
| `normalize_text` | Lowercases, removes URLs, numbers, punctuation, filters short reviews |
| `length_features` | Extracts `review_length`, `word_count`, `avg_word_length`, `sentence_count` |
| `sentiment_features` | Extracts VADER sentiment scores (pos, neg, neu, compound) |
| `tfidf_features` | Fits TF-IDF on train only, transforms all splits (prevents leakage) |
| `sbert_embeddings` | Encodes reviews with `all-MiniLM-L6-v2` into 384-dim vectors |
| `merge_features` | Inner joins all feature tables on `(asin, reviewerID)` |

---

## Procedure:

### 1. Clone and switch branch
```bash
git clone https://github.com/Maymona-M/60306027-Cloud_Win-26.git
cd 60306027-Cloud_Win-26
git checkout assignment2_model_training
```

### 2. Configure Azure ML CLI
```bash
az extension add -n ml -y
az configure --defaults group=rg-60306027 workspace=Amazon-Electronics-Lab-60306027
```

### 3. Register data assets (if not already registered)
```bash
az ml data create --name amazon_review_merged_features_train --version 1 \
  --type uri_folder --path <blob-path-to-train-out>
# repeat for val, test, deploy
```

### 4. Submit training job manually
```bash
az ml job create --file jobs/train_job.yml
```

### 5. Submit sweep job
```bash
az ml job create --file jobs/sweep_job.yml
```

### 6. Register the model
```bash
az ml model create --name amazon-review-sentiment-model \
  --path azureml://jobs/<JOB_NAME>/outputs/model_output --type custom_model
```

### 7. Create endpoint and deploy
```bash
az ml online-endpoint create --name amazon-review-ep-60306027 --auth-mode key
az ml online-deployment create --file jobs/deployment.yml --all-traffic
```

### 8. Invoke endpoint with deployment dataset
```bash
python src/invoke_endpoint.py --deploy_data <path-to-deploy-parquet>
```

### 9. Delete endpoint when done
```bash
az ml online-endpoint delete --name amazon-review-ep-60306027 --yes
```

---

## Project Screenshots

### 1. Metrics
![Metrics](images/1.%20Metrics.png)

### 2. Pipeline
![Pipeline](images/2.%20Pipeline.png)

### 3. Pipeline Run
![Pipeline Run](images/3.%20Pipeline%20Run.png)

### 4. Pipeline Jobs
![Pipeline Jobs](images/4.%20Pipeline%20Jobs%20.png)

### 5. Deployment Accuracy
![Deployment Accuracy](images/5.%20Deployment%20Acc.png)

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
**Job:** `plum_station_896mwkcsbm` | **Params:** C=1.0, max_iter=1000

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | 0.9033 | 0.9033 | 0.9033 |
| AUC | 0.9450 | 0.9450 | 0.9450 |
| F1 | 0.9421 | 0.9421 | 0.9421 |
| Precision | 0.9094 | 0.9094 | 0.9094 |
| Recall | 0.9772 | 0.9772 | 0.9772 |
| Runtime | — | — | 69.05s |

---

## Hyperparameter Tuning (Sweep Job)
**Job:** `yellow_energy_tw6x518766`

| Parameter | Search Space |
|-----------|-------------|
| `C` | uniform [0.01, 10.0] |
| `max_iter` | choice [500, 1000, 2000] |

- Sampling: random
- Max trials: 6 | Concurrent: 2
- Objective: maximize `val_accuracy`

Best configuration selected from sweep (see Azure ML Studio → Jobs → `amazon_review_training_experiment`).

---

## MLOps Workflow
```
code push → Azure DevOps pipeline → Azure ML training job → MLflow metrics → versioned model → deployed endpoint
```

**Azure DevOps CI:** triggers on push to `assignment2_model_training`, submits and streams training job automatically.
**Registered Model:** `amazon-review-sentiment-model` version 2
**Endpoint:** `amazon-review-ep-60306027` | Deployment: `blue` | Instance: `Standard_F2s_v2`

---

## Deployment
- **Endpoint:** `amazon-review-ep-60306027`
- **Deployment:** `blue`
- **Instance:** `Standard_F2s_v2` (1 instance)
- **Auth:** key-based

Endpoint invoked using the deployment dataset (`amazon_review_merged_features_deploy:1`) to simulate production predictions and measure real-world performance.

## Deployment Dataset Evaluation

| Metric | Deployment Dataset |
|--------|------------------|
| Accuracy | 0.888 |
| Samples Tested | 1000 |

---

## Important – Delete Endpoint

Idle endpoints continue consuming compute resources.  
After evaluating the deployment dataset and collecting metrics, delete the endpoint:

```bash
az ml online-endpoint delete --name amazon-review-ep-60306027 --yes
```

---

## Conclusion
This assignment demonstrates a complete end-to-end MLOps workflow built on Azure:
- Engineered features from Lab 4 were consumed directly as versioned data assets
- A Logistic Regression model achieved **90.3% accuracy and 94.5% AUC** on the test set
- MLflow tracked all metrics and parameters across every run
- Azure DevOps automated the training pipeline end-to-end with a single push
- Hyperparameter tuning via sweep jobs identified the optimal model configuration
- The final model was registered, deployed as a managed online endpoint, and invoked against the deployment dataset to simulate real production inference

---

## Bonus Question – 20%

**Question:** There is one thing we are doing “not correctly” in this assignment. What is it?

**Answer:**  
