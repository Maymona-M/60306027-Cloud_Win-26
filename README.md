# DSAI3202
**Maymona Mustafa, 60306027**

---

# Lab 1 — DSAI3202: End-to-End Machine Learning on Azure  

---

## Objective
Build a complete end-to-end machine learning pipeline:
- Version control with Git
- Train a machine learning model
- Save and version the trained model
- Deploy the model as an API on Azure
- Test the deployed service

---

## Tools & Technologies
- Git & GitHub  
- Python (scikit-learn)  
- Kaggle API  
- Azure Virtual Machines (Ubuntu)  
- Conda  
- FastAPI / Flask  
- Uvicorn  

---

## Project Structure
```bash
60306027-Cloud_Win-26/
├── data/
├── models/
├── outputs/
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── test.py
├── main.py
├── environment.yml
├── requirements.txt
└── README.md
```

---

### Step 1: Create & Clone GitHub Repository
```bash
git clone git@github.com:Maymona-M/60306027-Cloud_Win-26
cd 60306027-Cloud_Win-26
```

### Step 2: Configure Git
```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
git config --list
```

### Step 3: Create Azure Virtual Machine
  - OS: Ubuntu 22.04
  - Size: Standard_DS4_v3
  - Authentication: SSH

  Connect to VM:
  ```bash
  ssh MoonaVM@<VM_PUBLIC_IP>
  ```

#### Step 4: Install Miniconda on VM
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
conda --version
```

### Step 5: Clone Repository on VM
```bash
git clone git@github.com:<username>/60306027-Cloud_Win-26.git
cd 60306027-Cloud_Win-26
```

### Step 6: Configure Kaggle API
```bash
mkdir ~/.kaggle
nano ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Step 7: Download Titanic Dataset
```bash
kaggle competitions download -c titanic
unzip titanic.zip -d data/
ls data/
```

### Step 8: Train the Model
```bash
python main.py --mode train
```

### Step 9: Commit & Tag Model Version
```bash
git add .
git commit -m "Train Titanic model and save pipeline"
git tag v1.0
git push origin main --tags
```

### Step 10: Deploy Model as API
```bash
uvicorn main:app --reload
```

### Step 11: Test API
Open in browser:
```bash
http://127.0.0.1:8000/docs
```

Example response:
{
  "prediction": 1,
  "survival_status": "Survived"
}

---

## Conclusion
This lab demonstrates a full ML lifecycle:
  - Training
  - Versioning
  - Deployment

---

---

# Lab 2 — DSAI3202: Data Ingestion Pipeline on Azure  

---

## Objective
Design and implement a cloud-based data ingestion pipeline using Azure services.  
The pipeline ingests raw JSON data, transforms it into Parquet format, and stores it in a partitioned structure for efficient analytics.

---

## Tools & Technologies
- Azure Blob Storage (ADLS Gen2)  
- Azure Data Factory  
- Azure Machine Learning  
- Azure CLI  
- AzCopy  
- Python  

---

## Project Structure
```bash
60306027-Cloud_Win-26/
├── raw/
│   ├── reviews_Electronics_5.json
│   ├── meta_Electronics.json
│   └── meta_Electronics_fixed.json
├── processed/
│   └── reviews/
│       └── review_year=YYYY/
├── notebooks/
└── README.md
```

---

### Step 1: Create Azure Storage Account
- Enable **Hierarchical Namespace (ADLS Gen2)**
- Create the following containers:
```bash
raw
processed
curated
```

### Step 2: Upload Metadata File
Upload meta_Electronics.json to the raw container using the Azure Portal.

### Step 3: Create Azure ML Compute Instance
VM Size: Standard_DS3_v2
Open the terminal from Azure ML Studio.

### Step 4: Download Reviews Dataset
```bash
wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
ls -lh reviews_Electronics_5.json.gz
gunzip reviews_Electronics_5.json.gz
ls -lh reviews_Electronics_5.json
```

### Step 5: Upload Reviews Dataset Using AzCopy
```bash
azcopy copy \
"./reviews_Electronics_5.json" \
"https://<storage-account>.blob.core.windows.net/raw/reviews_Electronics_5.json?<SAS_TOKEN>" \
--overwrite=true
```

### Step 6: Fix Invalid Metadata JSON Format
```bash
python3 << 'EOF'
import ast, json

input_file = "meta_Electronics.json"
output_file = "meta_Electronics_fixed.json"

with open(input_file) as fin, open(output_file, "w") as fout:
    for line in fin:
        obj = ast.literal_eval(line)
        fout.write(json.dumps(obj) + "\n")

print("Metadata conversion complete.")
EOF
```

### Step 7: Upload Fixed Metadata File
```bash
azcopy copy \
"meta_Electronics_fixed.json" \
"https://<storage-account>.blob.core.windows.net/raw/meta_Electronics_fixed.json?<SAS_TOKEN>" \
--overwrite=true
```

### Step 8: Verify Uploaded Files
```bash
az storage blob list \
--account-name <storage-account> \
--container-name raw \
--sas-token <SAS_TOKEN> \
--output table
```

### Step 9: Configure Azure Data Factory

Create the following:
Linked Service: ADLS Gen2
Dataset: ds_reviews_raw_json
Dataset: ds_reviews_processed_parquet

### Step 10: Create Mapping Data Flow
Derived column:
year(toTimestamp(toLong(unixReviewTime) * 1000))

Sink configuration:
Partition column: review_year

### Step 11: Run the Data Pipeline
Pipeline name: pl_reviews_ingestion_parquet_partitioned

Output structure:
processed/reviews/review_year=YYYY/

### Step 12: Enable Pipeline Trigger
Trigger type: Schedule
Frequency: Daily
Status: Enabled

## Conclusion
This lab demonstrates a complete Azure data ingestion workflow:
- Raw JSON ingestion
- Data transformation using Azure Data Factory
- Parquet conversion
- Partitioned storage by year
- Automated pipeline execution
