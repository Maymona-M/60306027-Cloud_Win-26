# DSAI3202
**Maymona Mustafa, 60306027**

---

# Lab 4 — DSAI3202: Text Feature Engineering with Azure ML 

---

## Objective
Build an end-to-end text feature engineering pipeline on Azure ML:
- Explore and validate the Amazon Electronics Gold dataset in Databricks
- Create a drift-resistant sampled dataset
- Build and register Azure ML command components
- Orchestrate components into a full feature engineering pipeline
- Register engineered features in the Azure ML Feature Store

---

## Tools & Technologies
- Git & GitHub
- Python (pandas, scikit-learn, nltk, sentence-transformers)
- Azure Databricks
- Azure Data Lake Storage Gen2
- Azure Machine Learning (Components, Pipelines, Feature Store)
- Azure ML CLI

---

## Project Structure
```bash
60306027-Cloud_Win-26/
├── components/
│   ├── split_dataset/
│   │   ├── split.py
│   │   └── component.yml
│   ├── normalize_text/
│   │   ├── normalize.py
│   │   └── component.yml
│   ├── length_features/
│   │   ├── length_features.py
│   │   └── component.yml
│   ├── sentiment_features/
│   │   ├── sentiment.py
│   │   ├── component.yml
│   │   └── conda.yml
│   ├── tfidf_features/
│   │   ├── tfidf_features.py
│   │   └── component.yml
│   ├── sbert_embeddings/
│   │   ├── sbert_embeddings.py
│   │   ├── component.yml
│   │   └── conda.yml
│   └── merge_features/
│       ├── merge_features.py
│       └── component.yml
├── data/
│   └── features_v1_sampled.yml
├── feature_store/
│   ├── entity_amazon_review.yml
│   ├── FeatureSetSpec.yaml
│   └── feature_set_amazon_review_text_features.yml
├── notebooks/
│   └── 01_explore_validate_sample.ipynb
├── pipelines/
│   └── feature_pipeline.yml
└── README.md
```

---
## Part I — Databricks: Dataset Exploration and Sampling

### Step 1: Load and Inspect the Gold Dataset
- The Gold dataset (features_v1) was loaded from Azure Data Lake Storage into a Databricks notebook.
- The following checks were performed:
  - Row and column count verified
  - Schema inspected programmatically
  - reviewText confirmed as StringType
  - overall confirmed as numeric (DoubleType)
  - asin and reviewerID confirmed present as StringType
  - Missing and empty values checked in key columns

#### Visualization 1 — Rating Distribution
- A bar chart of 1–5 star ratings was plotted across the full dataset.
- The distribution shows class imbalance: 5-star reviews dominate the Amazon Electronics dataset.
- This means positive language will be over-represented in the TF-IDF vocabulary.
- It also informs downstream modeling decisions such as whether to apply class weighting or oversampling strategies.

#### Visualization 2 — Review Length Distribution
- A histogram of word counts per review was plotted, both full range and zoomed to 0–300 words.
- The distribution shows how many reviews fall below the 10-character filter threshold.
- It also indicates how much truncation SBERT will need to apply — reviews beyond 512 tokens are truncated by the transformer model. 
- The review_length_words feature itself is a meaningful signal since review length correlates with engagement and quality.


### Step 2: Create a Sampled Dataset
- Question: Language evolves over the years. This creates drift. How would you ensure your sampling is resistant to drift?
- Solution: Stratified sampling by review_year

```bash
sample_n    = 300_000
sample_seed = 42
total_rows  = df.count()
fraction    = sample_n / total_rows

year_fractions = {
    row["review_year"]: fraction
    for row in df.select("review_year").distinct().collect()
    if row["review_year"] is not None
}

df_sampled = df.sampleBy("review_year", fractions=year_fractions, seed=sample_seed)
print("Sampled rows:", df_sampled.count())
```


## Part 2 — Azure ML Feature Engineering Pipeline

### Pipeline Flow

```bash
features_v1_sampled (300k reviews)
        |
      split  (70% train / 15% val / 15% test)
        |
normalize_train   normalize_val   normalize_test   (parallel)
        |
  length_train   sentiment_train   tfidf   sbert_train
        |
     merge_all
        |
  Final merged feature dataset (parquet)
        |
  Azure ML Feature Store
```


### Step 1: GitHub Branch
```bash
git checkout -b lab4_feature_engineering
git push -u origin lab4_feature_engineering
```

### Step 2: Configure Azure ML CLI
```bash
az extension add -n ml -y
az extension update -n ml
az login
az configure --defaults group="rg-60306027" workspace="Amazon-Electronics-Lab-60306027"
az ml workspace show
```

### Step 3: Configure Azure ML Access to Data Lake
```bash
az ml datastore create -f datastores/curated_adls.yml
```

### Step 4: Register Sampled Gold Dataset as Data Asset
```bash
az ml data create -f data/features_v1_sampled.yml
az ml data show --name amazon_electronics_features_v1_sampled --version 1
```

### Step 5: Create Feature Store Entity
- Feature Store created in Azure ML Studio: amazon-electronics-fs-60306027

```bash
az ml feature-store-entity create --file feature_store/entity_amazon_review.yml --resource-group rg-60306027 --feature-store-name amazon-electronics-fs-60306027
```

### Step 6: Azure ML Command Components
1. split_dataset
- Splits the dataset into train (70%), validation (15%), and test (15%) using two sequential train_test_split calls. The split must run before any feature fitting to prevent data leakage.
- If TF-IDF is fitted on the full dataset before splitting, the vocabulary will contain information from validation and test reviews. This means the model indirectly sees test data during training, inflating performance metrics artificially.

2. normalize_text
- Cleans raw review text through a pipeline of regex transformations:
  - Lowercase: "Great" and "great" are the same word — without this TF-IDF treats them as separate tokens, inflating vocabulary size
  - Remove URLs: URLs like http://amazon.com carry no sentiment or topic signal
  - Replace numbers: Standalone numbers add noise without semantic meaning
  - Remove punctuation: Punctuation is not a meaningful token for TF-IDF or SBERT
  - Collapse whitespace: Previous steps leave multiple spaces; this normalizes them
  - Filter short reviews: Reviews under 10 characters after cleaning carry no useful signal

3. length_features
- Extracts two basic statistical features:
  - review_length_words: number of whitespace-separated tokens
  - review_length_words: number of whitespace-separated tokens

- Review length correlates with reviewer engagement and quality. Very short reviews are often uninformative. Very long reviews tend to contain detailed technical assessments. These are lightweight features that add signal with near-zero compute cost.

4. sentiment_features
- Uses NLTK VADER (Valence Aware Dictionary and sEntiment Reasoner) to extract four scores:
  - sentiment_pos: proportion of positive sentiment words (0–1)
  - sentiment_neg: proportion of negative sentiment words (0–1)
  - sentiment_neu: proportion of neutral words (0–1)
  - sentiment_compound: normalized overall polarity score (−1 to +1)
- VADER is designed specifically for informal text and product reviews. It handles slang, capitalization emphasis, punctuation emphasis, and negations (e.g., "not good") without requiring model training. 

5. tfidf_features
- Uses sklearn.feature_extraction.text.TfidfVectorizer with the following settings:
  - max_features=5000: limits vocabulary to the top 5000 most informative terms, keeping the feature matrix manageable
  - stop_words='english':removes common filler words (the, a, is) that appear everywhere and carry no discriminative signal
  - ngram_range=(1,2): captures both unigrams and bigrams, allowing phrases like "not good" or "battery life" to be represented as single features
  - sublinear_tf=True: applies log normalization to term frequency, reducing the dominance of very frequent terms
- Data leakage prevention: The vectorizer is fitted exclusively on the training split. The frozen vocabulary and IDF weights are then applied to validation and test splits using .transform() only — never .fit_transform(). The fitted vectorizer is saved as a .joblib file for auditability.

6. sbert_embeddings
-Uses all-MiniLM-L6-v2 from sentence-transformers to encode each review as a 384-dimensional dense vector (bert_emb_0 through bert_emb_383).
- TF-IDF treats each word independently and cannot capture semantic similarity. SBERT understands that "excellent battery life" and "great power duration" express the same concept. 
- Processing is done in chunks of 10,000 rows with gc.collect() between chunks to avoid out-of-memory errors on CPU compute nodes. 

7. merge_features
- Performs inner joins of all feature tables on composite key (asin, reviewerID).
- Each table is deduplicated before joining to prevent row count explosion from many-to-many joins. 
- Join order follows smallest-to-largest table to minimize peak memory usage.
- Pipeline execution time is logged at the end of this step as part of the optimization challenge.


### Step 7: Pipeline Definition and Submission
```bash
az ml job create --file pipelines/feature_pipeline.yml
```

### Step 8: Register Features in the Azure ML Feature Store
```bash
az ml feature-set create --file feature_store feature_set_amazon_review_text_features.yml --resource-group rg-60306027 --feature-store-name amazon-electronics-fs-60306027

az ml feature-set show --name amazon_review_text_features --version 1 --resource-group rg-60306027 --feature-store-name amazon-electronics-fs-60306027
``` 

### Step 9: Rommit and Push
```bash
git add components/ pipelines/ feature_store/ data/ .gitignore README.md
git commit -m "Lab 4: feature engineering pipeline, components, feature store assets, and README"
git push origin lab4_feature_engineering
```

---

## Conclusion
This lab demonstrates a full feature engineering lifecycle:
  - Dataset exploration and validation in Databricks
  - Drift-resistant temporal stratified sampling
  - Modular Azure ML components for text normalization, sentiment, TF-IDF, and SBERT
  - End-to-end pipeline orchestration on Azure ML compute
  - Feature registration in Azure ML Feature Store for reuse in downstream modeling labs

---

