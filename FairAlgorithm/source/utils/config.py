
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# config of ML models
n_estimators = 30
random_seed = 1234

models = {'Logistic Regression':LogisticRegression(max_iter=500),
          'Decision Tree':DecisionTreeClassifier(max_depth=None),
          'Bagging':BaggingClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=n_estimators),
          'Random Forest':RandomForestClassifier(n_estimators=n_estimators),
          'Extremely Randomized Trees':ExtraTreesClassifier(n_estimators=n_estimators),
          'Ada Boost':AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=n_estimators)}

family = ['division', 'subtraction']
fairness_catalogue = ['GroupFairness', 'PredictiveParity', 'PredictiveEquality', 'EqualOpportunity', 'EqualizedOdds', 'ConditionalUseAccuracyEquality', 'OverallAccuracyEquality', 'TreatmentEquality', 'FORParity', 'FN', 'FP']
performance_metrics = ['accuracy', 'precision', 'recall', 'f1_score']

all_techniques = ['original', 'fl-cr', 'fl-to', 'aif360-rw', 'aif360-di', 'aif360-lfr', 'aif360-op', 'aif360-ad', 'aif360-pr', 'aif360-er', 'aif360-ce', 'aif360-eo', 'aif360-roc']
all_mitigations = ['fl-cr', 'fl-to', 'aif360-rw', 'aif360-di', 'aif360-lfr', 'aif360-op', 'aif360-ad', 'aif360-pr', 'aif360-er', 'aif360-ce', 'aif360-eo', 'aif360-roc']
preprocessing_mitigation_list = ['fl-cr', 'aif360-rw', 'aif360-di', 'aif360-lfr', 'aif360-op']
inprocessing_mitigation_list = ["aif360-ad", "aif360-er",'aif360-pr'] 
postprocessing_mitigation_list = ['aif360-roc', 'aif360-ce', 'fl-to', 'aif360-eo']
without_model_mitigations = ['aif360-ad', 'aif360-pr', 'aif360-er']
new_dataset_mitigations = ["fl-cr", "aif360-di", "aif360-op" , "aif360-lfr"]

#config dataset
datasets_config = {
  "diabetes-women":{
    'ignore_cols': ['Age'],
    'target_variable': 'Outcome',
    'target_variable_labels': [1,0],
    'sensible_attributes': ['AgeCategory'],
    'default_mappings': {
      'label_maps': [{1.0: 'Diabetic', 0.0: 'NonDiabetic'}],
      'protected_attribute_maps': [{1.0: 'Adult', 0.0: 'Young'}]
    },
    'n_splits': 5
  },
  "sepsis": {
    'ignore_cols': [],
    'target_variable': 'Mortality',
    'target_variable_labels': [1,0],
    'sensible_attributes': ['Gender_cat', 'Age_cat'],
    'default_mappings': {},
    'n_splits': 5
  },
  'aids': {
    'ignore_cols': [],
    'target_variable': 'cid',
    'target_variable_labels': [1, 0],
    'sensible_attributes':  ['homo_cat', 'race_cat', 'age_cat'],
    'default_mappings': {},
    'n_splits': 10
},
  "myocardial-infarction":{
    'ignore_cols': [],
    'target_variable': 'LET_IS_cat',
    'target_variable_labels': [1,0],
    'sensible_attributes': ['SEX'],
    'default_mappings': {
        'label_maps': [{1: 'Complication', 0: 'No Alzheimer'}],
        'protected_attribute_maps': [{0: 'Female', 1: 'Male'}]
    },
    'n_splits': 10
  },
  'alzheimer-disease':{
    'ignore_cols': [],
    'target_variable': 'Diagnosis',
    'target_variable_labels': [1, 0],
    'sensible_attributes': ['Ethnicity_cat', 'Gender_cat'],
    'default_mappings': {
        'label_maps': [{1: 'Alzheimer', 0: 'No Alzheimer'}],
        'protected_attribute_maps': [{"Ethnicity_cat":{1: "Caucasian", 0: "Non-Caucasian"}, "Gender_cat":{1: "Male", 0: "Female"}}]
    },
    'n_splits': 10
  },
  "diabetes-prediction":{
    'ignore_cols': [],
    'target_variable': 'diabetes',
    'target_variable_labels': [1, 0],
    'sensible_attributes': ['race_category'],
    'default_mappings': {'label_maps': [{1.0: 'Diabetic', 0.0: 'NonDiabetic'}],
                        'protected_attribute_maps': [{1.0: 'Caucasian', 0.0: 'Non-Caucasian'}] 
    },
    'n_splits': 10
  },
  "stroke-prediction":{
    'ignore_cols': [],
    'target_variable': 'stroke_prediction',
    'target_variable_labels': [1, 0],
    'sensible_attributes': ['residence_category'],
    'default_mappings': {
        'label_maps': [{1.0: 'Stroke', 0.0: 'No Stroke'}],
        'protected_attribute_maps': [{1.0: 'Urban', 0.0: 'Rural'}]
    },
    'n_splits': 10
  }
}
