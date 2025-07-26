# 🔬 FAIR-CARE
A Comparative Evaluation of Unfairness Mitigation Approaches for Healthcare Datasets. FAIR-CARE is a Python-based framework designed to evaluate and mitigate algorithmic bias in healthcare machine learning pipelines. It supports multiple fairness metrics, mitigation strategies, and  machine learning models commonly used in healthcare prediction tasks.

This Github repository contains the source code and the plots of the experiments of the paper.

# 💬 Features

- Supports a wide range of fairness metrics (e.g., demographic parity, equal opportunity) both in division and subtraction comparison
- Includes pre-processing, in-processing, and post-processing bias mitigation techniques
- Works with both interpretable (e.g., logistic regression) and complex (e.g., random forest, boosting) machine learning models
- Modular design for easy customization
- Reproducible experiments with standardized reporting
- Preconfigured examples with real healthcare datasets

# 📁 Repository Structure

```
FairAlgorithm/
├── data/                    # Organized datasets
│   ├── raw/                # Original unprocessed datasets
│   ├── preprocessed/       # Cleaned and transformed datasets
│   ├── mitigated/          # Fairness-mitigated datasets
│   ├── predictions_and_tests/  # Model predictions and test results
│   └── reports/            # Summarized results and logs
├── measurements/           # Serialized evaluation metrics (e.g., fairness, accuracy)
├── fidings/                # Summary findings from experiments
│   ├── fairness/
│   ├── performance/
│   └── trade-off/
├── plots/                  # Fairness-performance trade-off plots for each dataset
├── source/                 # Source code and main notebooks
│   ├── preprocessing/      # Data cleaning and feature preparation
│   ├── mitigation/         # Fairness mitigation techniques
│   ├── plots/              # Plotting utilities
│   ├── tuning/             # Hyperparameter tuning scripts
│   ├── utils/              # Shared utility functions
│   ├── all-plots.ipynb
│   ├── mitigation.ipynb
│   ├── measurement-original.ipynb
│   ├── measurement-post-mitigation.ipynb
│   └── requirements.txt
├── Flow_FairAlgorithm.drawio # Flowchart of the framework
└── README.md                # Project documentation
# Repository structure
```

# 💬 Info for developers
It is possible to visualize the content of this repository also in [this Google Colab project](https://drive.google.com/drive/folders/182YKE0bNOltAezFfcEVEy7-FwXemlWX8?usp=sharing) otherwise the following specifications allow to execute this code locally using Jupyter Notebook with Python kernel.

## 📊 Run the Pipeline on an Existing Dataset
You can reproduce fairness and performance evaluations by running:

- source/measurement-original.ipynb: Runs baseline evaluation without mitigation.
- source/mitigation.ipynb: Applies selected fairness mitigation techniques.
- source/measurement-post-mitigation.ipynb: Re-evaluates model after mitigation.
- source/all-plots.ipynb: Generates comparative plots for fairness vs performance.

## 🆕 Using FAIR-CARE with a New Dataset
To apply the framework to a new dataset:

1. Place your dataset in the data/raw/ directory.

2. Follow the instructions in Colab notebooks (source/preprocessing/) to:

- Handle missing values
- Remove outliers
- Binarize protected attributes and target variable
- Perform any required transformations

3. Update the configuration file in source/utils/ :
In your notebooks or script, modify the relevant parameters (e.g., dataset name, target column, protected attributes). You can also adapt existing functions and notebook cells for your specific use case.

🛠 Note: Preprocessing is not automated by default to allow flexibility for healthcare-specific cleaning logic. However, we provide modular code and clear examples to support adaptation.

# 📁 Datasets Included
This repository contains experiments with multiple real-world healthcare datasets (under data/ and plots/):

- AIDS
- Alzheimer's Disease 
- Diabetes Prediction
- Myocardial Infarction 
- Sepsis
- Stroke Risk

Each dataset has been evaluated for both fairness and predictive performance, with results available in the fidings/ and plots/ folders.

# 📈 Fairness & Performance Trade-Off
FAIR-CARE allows comparison across:
- Fairness metrics (e.g., demographic parity, equal opportunity)
- ML models (e.g., logistic regression, decision tree, random forest, boosting)
- Mitigation techniques (pre-, in-, post-processing)
- Performance outcomes (accuracy, F1-score, AUC)

# 📓 Visualization
The Flow_FairAlgorithm.drawio file provides a visual overview of the pipeline, from preprocessing to mitigation and evaluation.

# ⚙️ Configuration Guide (config file)
This framework is configured through a Python configuration file that defines:

## 🔍 Models
Supported ML models include:
- Logistic Regression
- Decision Tree (DT)
- Bagging (DT base)
- Random Forest
- Extremely Randomized Trees
- AdaBoost (DT base)

Set via the models dictionary using Scikit-learn implementations. All models use a fixed random_seed for reproducibility.

## 📊 Metrics
- Performance: accuracy, precision, recall, f1_score
- Fairness: Includes metrics like EqualOpportunity, GroupFairness, PredictiveParity, and more, with 2 possible comparisons division and subtraction

## 🧪 Datasets
Each dataset in datasets_config includes:
- target_variable: The prediction label
- sensible_attributes: Protected features
- ignore_cols: (Optional) Features to exclude
- default_mappings: Maps numeric values to readable labels
- n_splits: Cross-validation folds

Update this section when adding a new dataset.

## 🛠️ Mitigation Techniques
Mitigation codes refer to:
- fl- for Fairlearn
- aif360- for IBM's AI Fairness 360
  
Techniques are grouped by type:

- Pre-processing: fl-cr, aif360-di, etc.
- In-processing: aif360-ad, aif360-pr
- Post-processing: fl-to, aif360-roc

Listed under all_techniques, with subsets for filtering.

## ⚖️ Fairness Config (AIF360)
The aif_config block defines:
- privileged_groups / unprivileged_groups
- Model for in-processing (e.g., lmod)
- Mitigation params: k, max_iter_lfr, etc.

Configure these per attribute for fairness evaluation with IBM techniques




