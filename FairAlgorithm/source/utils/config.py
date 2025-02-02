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
      'AgeCategory':{
        'label_maps': [{1.0: 'Diabetic', 0.0: 'NonDiabetic'}],
        'protected_attribute_maps': [{1.0: 'Adult', 0.0: 'Young'}]
      }
    },
    'n_splits': 5
  },
  "sepsis": {
    'ignore_cols': [],
    'target_variable': 'Mortality',
    'target_variable_labels': [1,0],
    'sensible_attributes': ['Gender_cat', 'Age_cat'],
    'default_mappings': {
        'Gender_cat':{
          'label_maps': [{1.0: 'Death', 0.0: 'Censored'}],
          'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'}]
      },
        'Age_cat': {
          'label_maps': [{1.0: 'Death', 0.0: 'Censored'}],
          'protected_attribute_maps': [{1.0: 'Adult', 0.0: 'Young'}]
      }
    },
    'n_splits': 5
  },
  'aids': {
    'ignore_cols': [],
    'target_variable': 'cid',
    'target_variable_labels': [1, 0],
    'sensible_attributes':  ['homo_cat', 'race_cat', 'age_cat'],
    'default_mappings': {
        'homo_cat':{
            'label_maps': [{1.0: 'Death', 0.0: 'Censored'}],
            'protected_attribute_maps': [{1.0: 'Non-Homo', 0.0: 'Homo'}]
        },
        'race_cat':{
            'label_maps': [{1.0: 'Death', 0.0: 'Censored'}],
            'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-white'}]
        },
        'age_cat':{
            'label_maps': [{1.0: 'Death', 0.0: 'Censored'}],
            'protected_attribute_maps': [{1.0: 'Adult', 0.0: 'Young'}]
        }
        
    },
    'n_splits': 10
},
  "myocardial-infarction":{
    'ignore_cols': [],
    'target_variable': 'LET_IS_cat',
    'target_variable_labels': [1,0],
    'sensible_attributes': ['SEX'],
    'default_mappings':{
      'SEX': {
          'label_maps': [{1: 'Complication', 0: 'No Alzheimer'}],
          'protected_attribute_maps': [{0: 'Female', 1: 'Male'}]
        }
    },
    'n_splits': 5
  },
  'alzheimer-disease':{
    'ignore_cols': [],
    'target_variable': 'Diagnosis',
    'target_variable_labels': [1, 0],
    'sensible_attributes': ['Ethnicity_cat', 'Gender_cat'],
    'default_mappings': {
      "Ethnicity_cat": {
        'label_maps': [{1: 'Alzheimer', 0: 'No Alzheimer'}],
        'protected_attribute_maps': [{1: "Caucasian", 0: "Non-Caucasian"}]
    },
      "Gender_cat": {
        'label_maps': [{1: 'Alzheimer', 0: 'No Alzheimer'}],
        'protected_attribute_maps': [{1: "Male", 0: "Female"}]
    },
    },
    'n_splits': 10
  },
  "diabetes-prediction":{
    'ignore_cols': [],
    'target_variable': 'diabetes',
    'target_variable_labels': [1, 0],
    'sensible_attributes': ['race_category'],
    'default_mappings': {
      'race_category': {
        'label_maps': [{1.0: 'Diabetic', 0.0: 'NonDiabetic'}],
        'protected_attribute_maps': [{1.0: 'Caucasian', 0.0: 'Non-Caucasian'}] 
      }
    },
    'n_splits': 10
  },
  "stroke-prediction":{
    'ignore_cols': [],
    'target_variable': 'stroke_prediction',
    'target_variable_labels': [1, 0],
    'sensible_attributes': ['residence_category'],
    'default_mappings': {
        'residence_category': {
          'label_maps': [{1.0: 'Stroke', 0.0: 'No Stroke'}],
          'protected_attribute_maps': [{1.0: 'Urban', 0.0: 'Rural'}]
      }
    },
    'n_splits': 5
  }
}

aif_config = {
  "diabetes-women":{
      'AgeCategory': {
        'privileged_groups': [{'AgeCategory': 1}],
        'unprivileged_groups': [{'AgeCategory': 0}],
        'reduced_df_techniques': None,
        'params': {
          'k': 87,
          'max_iter_lfr': 500,
          'lmod' : LogisticRegression(solver='lbfgs', max_iter=1000),
          'metric_name_roc': "Statistical parity difference",
          'n_splits_eo':5 
      } 
    }
  },
  "sepsis": {
    'Gender_cat': {
      'privileged_groups': [{'Gender_cat': 1}], 
      'unprivileged_groups': [{'Gender_cat': 0}],
      'reduced_df_techniques': None,
      'params': {
          'k':52,
          'max_iter_lfr': 5000,
          'lmod': LogisticRegression(solver='lbfgs', max_iter=1000),
          'metric_name_roc': "Statistical parity difference",
          'n_splits_eo':4
      }
    }, 
    'Age_cat':{
      'privileged_groups': [{'Age_cat': 1}], 
      'unprivileged_groups': [{'Age_cat': 0}],
      'reduced_df_techniques': None,
      'params': {
          'k':43,
          'max_iter_lfr': 5000,
          'lmod': LogisticRegression(solver='lbfgs', max_iter=1000),
          'metric_name_roc': "Statistical parity difference",
          'n_splits_eo':5 
      }
    }
  },
  'aids':{
    'homo_cat':{
      'privileged_groups': [{'homo_cat': 1}], 
      'unprivileged_groups': [{'homo_cat': 0}],
      'reduced_df_techniques': None,
      'params': {
          'k':147,
          'max_iter_lfr': 5000,
          'lmod': LogisticRegression(solver='lbfgs', max_iter=1000),
          'metric_name_roc': "Statistical parity difference",
          'n_splits_eo':10
      }
    }, 
    'race_cat': {
      'privileged_groups': [{'race_cat': 1}], 
      'unprivileged_groups': [{'race_cat': 0}],
      'reduced_df_techniques': None,
      'params': {
          'k':51,
          'max_iter_lfr': 5000,
          'lmod': LogisticRegression(solver='lbfgs', max_iter=1000),
          'metric_name_roc': "Statistical parity difference",
          'n_splits_eo':10
      }
    },
    'age_cat': {
      'privileged_groups': [{'age_cat': 1}], 
      'unprivileged_groups': [{'age_cat': 0}],
      'reduced_df_techniques': None,
      'params': {
          'k':51,
          'max_iter_lfr': 5000,
          'lmod': LogisticRegression(solver='lbfgs', max_iter=1000),
          'metric_name_roc': "Statistical parity difference",
          'n_splits_eo':10
      }
    }
  },
  "myocardial-infarction":{
    'SEX': {
      'privileged_groups': [{'SEX': 1}], 
      'unprivileged_groups': [{'SEX': 0}],
      'reduced_df_techniques': None,
      'params': {
          'k':32,
          'max_iter_lfr': 5000,
          'lmod': LogisticRegression(solver='lbfgs', max_iter=1000),
          'metric_name_roc': "Statistical parity difference",
          'n_splits_eo':5
      }
    }
  },
  'alzheimer-disease':{
    'Ethnicity_cat': {
      'privileged_groups': [{'Ethnicity_cat': 1}], 
      'unprivileged_groups': [{'Ethnicity_cat': 0}],
      'reduced_df_techniques': None,
      'params': {
          'k':80,
          'max_iter_lfr': 5000,
          'lmod': LogisticRegression(solver='lbfgs', max_iter=1000),
          'metric_name_roc': "Statistical parity difference",
          'n_splits_eo':10
      }
    },
    'Gender_cat': {
      'privileged_groups': [{'Gender_cat': 1}], 
      'unprivileged_groups': [{'Gender_cat': 0}],
      'reduced_df_techniques': None,
      'params': {
          'k':80,
          'max_iter_lfr': 5000,
          'lmod': LogisticRegression(solver='lbfgs', max_iter=1000),
          'metric_name_roc': "Statistical parity difference",
          'n_splits_eo':10
      }
    }
  },
  "diabetes-prediction":{
    'race_category':{
      'privileged_groups': [{'race_category': 1}], 
      'unprivileged_groups': [{'race_category': 0}],
      'reduced_df_techniques': ['aif360-lfr', 'aif360-op', 'aif360-roc'],
      'params': {
          'k':4,
          'max_iter_lfr': 500,
          'lmod': DecisionTreeClassifier(max_depth=50),
          'metric_name_roc': "Equal opportunity difference",
          'n_splits_eo':10 
      }
    }
  },
  "stroke-prediction":{
    'residence_category':{
        'privileged_groups': [{'residence_category': 1}],  
        'unprivileged_groups': [{'residence_category': 0}],
        'reduced_df_techniques': ['aif360-lfr', 'aif360-op', 'aif360-roc'],
        'params': {
          'k': 92,
          'max_iter_lfr': 500,
          'lmod' : DecisionTreeClassifier(max_depth=50),
          'metric_name_roc': "Statistical parity difference",
          'n_splits_eo':5 
      } 
    }    
  }   
}