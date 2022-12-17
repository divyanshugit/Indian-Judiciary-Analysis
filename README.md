# Indian Judiciary Analysis

## Directory Sturcture:

```
divyanshugit/Indian-Judiciary Analysis
├── README.md                             <-- You are here 📌
├── src                               	  <-- Scripts to train/fine-tune model
└── notebooks                             <-- Jupyter Notebook with detailed walkthrough | Exploratory Data Analysis
```

## About Data:

For analysis of the Indian Judicial Data, we will utilize the dataset provided Development Data Lab(DDL) Judicial Data.
You can find the dataset and details [here](https://www.devdatalab.org/judicial-data). The open-source data contains data
from 2010 to 2018. We will take data collected after 2016 into consideration for the majority of our analysis. DDL also
released a [manuscript](https://shrug-assets-ddl.s3.amazonaws.com/static/main/assets/other/India_Courts_In_Group_Bias.pdf)
where they examined the group bias in the Indian Judiciary System. A few sections of the analysis of data are inspired by the paper.

After downloading the datasets:

```bash
tar -xvf judges_clean.tar.gz -C ~/divyanshu/PreCog/data/judge/
tar -xvf cases/cases.tar.gz -C ~/divyanshu/PreCog/data/case/
tar -xvf keys/keys.tar.gz -C ~/divyanshu/PreCog/data/
```

- We will go through `cases_2010.csv`, `cases_2017.csv`, and `cases_2018.csv` for analysis.
- We will be utilizing the following datasets `judge_clean.csv`, `cases_state_key.csv` and `cases_district_key.csv` for classification.



## Findings:

### Year 2016:


## Classification for Judge assignment on a case:

For classification of Judge on a case:

```
Inputs: Judge Position, Location(State, District)
Labels: Female, Otherwise, Unclear
```

We will address the multi-label classification problem. To solve this we're gonna approach it in different ways:

- Classical ML Approach(XGBoost)
- Transformer based approach(DistillBERT)
- Will decide later on(Only if needed)

### Training XGBoost
To train the XGBoost classifier,
Key things:
- In the `judge_clean.csv`,  we will encode the `judge_position`.

```python
$ python src/train_xgboost.py
```

Metric(f1-score):
- On test data: 56%

- [x] Optimize the Hyperparameter of XGBoost.

```python
$ python src/optimize_xgb.py
```

### Fine-tuning DistillBERT
To train DistillBERT,
Key things:
- We will enhance, the data of `judge_clean.csv` by utilizing `cases_state_key.csv` and `cases_district_key.csv` to map the actual name of States and Districts.

```python
$ python src/preprocessor.py # To generate data
$ python src/train.py
```
Metric(f1-score):
- On test data: 45%

To do:
- [ ] Optimize the Hyperparameter of DistillBERT.


