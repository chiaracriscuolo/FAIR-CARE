# FAIR-CARE
A Comparative Evaluation of Unfairness Mitigation Approaches for Healthcare Datasets

This Github repository contains the source code and the plots of the experiments of the paper.

# Repository structure

```
├── FairAlgorithm
│   ├── data
│   │   ├── mitigated
│   │   ├── predictions_and_tests
│   │   ├── preprocessed
│   │   ├── raw
│   │   └── reports
│   ├── Flow_FairAlgorithm.drawio
│   ├── measurements
│   │   ├── metrics-aids-age_cat-aif360-ad.p
│   │   ...
│   ├── plots
│   │   ├── aids-age_cat
│   │   ├── aids-homo_cat
│   │   ├── aids-race_cat
│   │   ├── alzheimer-disease-Ethnicity_cat
│   │   ├── alzheimer-disease-Gender_cat
│   │   ├── diabetes-prediction-race_category
│   │   ├── diabetes-women-AgeCategory
│   │   ├── myocardial-infarction-SEX
│   │   ├── sepsis-Age_cat
│   │   ├── sepsis-Gender_cat
│   │   └── stroke-prediction-residence_category
│   └── source
│       ├── all-plots.ipynb
│       ├── measurement-original.ipynb
│       ├── measurement-post-mitigation.ipynb
│       ├── mitigation
│       ├── mitigation.ipynb
│       ├── plots
│       ├── preprocessing
│       ├── requirements.txt
│       ├── tuning
│       └── utils
└── README.md
```
# Info for developers
It is possible to visualize the content of this repository also in [this Google Colab project](https://drive.google.com/drive/folders/182YKE0bNOltAezFfcEVEy7-FwXemlWX8?usp=sharing) otherwise the following specifications allow to execute this code locally.

# Requirement
It is possible to execute the code, both on Google Colaboratory or locally using Jupyter Notebook with Python kernel.
