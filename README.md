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

- Automated pipeline execution
