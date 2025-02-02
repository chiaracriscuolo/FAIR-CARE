#import libraries
import numpy as np
import pandas as pd
from rich import print
import statistics
from source.utils.print_util import *
from source.utils.data_preprocessing import *
import matplotlib.pyplot as plt
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.postprocessing import ThresholdOptimizer
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover, Reweighing, LFR, OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.inprocessing import PrejudiceRemover, AdversarialDebiasing, ExponentiatedGradientReduction
from aif360.algorithms.postprocessing import RejectOptionClassification, CalibratedEqOddsPostprocessing, EqOddsPostprocessing
from aif360.algorithms import Transformer
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tqdm.notebook import tqdm
import pickle
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, recall_score, accuracy_score, precision_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import preprocessing
from source.utils.config import *
from source.utils.metrics_utils import *


#UTILS TRAIN_TEST SPLIT
def df_X_Y_split_2D(df_train, df_test, target_variable, sensible_attribute):
  Y_train = df_train[target_variable].values
  S_train = df_train[sensible_attribute].values
  X_train = df_train.drop(target_variable, axis=1)
  X_train = X_train.drop(sensible_attribute, axis=1)

  Y_test = df_test[target_variable]
  X_test = df_test.drop(target_variable, axis=1)
  S_test = df_test[sensible_attribute].values

  X_train = normalize(X_train)
  X_test = normalize(X_test)

  pca = PCA(n_components=2)
  X_train2D = pca.fit(X_train).transform(X_train)
  X_test2D = pca.fit(X_test).transform(X_test)
  return X_train2D, Y_train, X_test2D, Y_test, S_train, S_test

def df_X_Y_split_S(df_train, df_test, target_variable, sensible_attribute):
  Y_train = df_train[target_variable].values
  S_train = df_train[sensible_attribute].values
  X_train = df_train.drop(target_variable, axis=1)

  Y_test = df_test[target_variable]
  X_test = df_test.drop(target_variable, axis=1)
  S_test = df_test[sensible_attribute].values

  return X_train, Y_train, X_test, Y_test, S_train, S_test

def fl_am_compute_predictions_and_tests(df, target_variable, sensible_attribute, n_splits, classifier):
  predicted_and_real_values = {}

  df_splitting = train_test_splitting(df, n_splits)
  pred_and_y = {}
  for i in range(0,n_splits):

    df_split = df_splitting[i]
    df_train = df_split['train']
    df_test = df_split['test']
    ##Input data should be in 2-dimensions
    X_train2D, Y_train, X_test2D, Y_test, S_train, S_test = df_X_Y_split_2D(df_train, df_test, target_variable, sensible_attribute)
    #print(len(X_train2D), len(Y_train),len(S_train))

    #classifier is given in in-processing mitigation
    predictor = classifier.fit(X_train2D, Y_train, sensitive_features=S_train)
    y_pred = predictor.predict(X_test2D)
    pred_and_y[i] = {'y_test': Y_test.to_numpy(), 'y_pred': y_pred, 's_test':  S_test }

  predicted_and_real_values = pred_and_y

  return predicted_and_real_values

def fl_to_compute_predictions_and_tests(df, target_variable, sensible_attribute, n_splits, classifier):
  predicted_and_real_values = {}

  df_splitting = train_test_splitting(df, n_splits)
  pred_and_y = {}
  for i in range(0,n_splits):

    df_split = df_splitting[i]
    df_train = df_split['train']
    df_test = df_split['test']
    ##Input data should be in 2-dimensions
    X_train, Y_train, X_test, Y_test, S_train, S_test = df_X_Y_split_S(df_train, df_test, target_variable, sensible_attribute)
    #print(len(X_train2D), len(Y_train),len(S_train))

    #classifier is given in in-processing mitigation
    predictor = classifier.fit(X_train, Y_train, sensitive_features=S_train)
    y_pred = predictor.predict(X_test, sensitive_features=S_test)
    pred_and_y[i] = {'y_test': Y_test.to_numpy(), 'y_pred': y_pred, 's_test':  S_test}

  predicted_and_real_values = pred_and_y

  return predicted_and_real_values

# SAVING UTILS
def unpack_config(config):
  return config['df'], config['dataset_name'], config['target_variable'], config['sensible_attribute'], config['path_to_project'], config['n_splits'], config['models'],config['random_seed'], config['privileged_groups'], config['unprivileged_groups'], config['default_mappings'], config['reduced_df_techniques'], config['params']

def save_predictions_and_tests(predicted_and_test, dataset_name, sensible_attribute, mitigation,path_to_project):
  save_path = path_to_project + '/data/predictions_and_tests/pred_test-{}-{}-{}.p'.format(dataset_name, sensible_attribute, mitigation)
  with open(save_path, 'wb') as fp:
    pickle.dump(predicted_and_test, fp, protocol=pickle.HIGHEST_PROTOCOL)

def save_mitigated_dataset(mitigated_dataset, path_to_project,dataset_name, sensible_attribute, mitigation):
  mitigated_dataset.to_csv("{}/data/mitigated/mitigated-{}-{}-{}.csv".format(path_to_project,dataset_name, sensible_attribute, mitigation), sep=',', index=False, encoding='utf-8')

#REDUCE DATASET UTILS
def reduce_df_diabetes_prediction(df, sensible_attribute, target_variable):
    # Build a list with format (feature, correlation with target)
    features_corr = [(column, correlation) for column, correlation in zip(df.columns, df.corr()[target_variable])] # Modify diabetes with the target_variable

    # Sort the features by correlation
    sorted_features = sorted(features_corr, key=lambda x: x[1], reverse=True)

    # Clean and take top 4
    main_features = [feature[0] for feature in sorted_features if feature[1] > 0][:5]

    # Add sensitive attribute
    main_features = main_features + [sensible_attribute]

    # Create a new reduced df
    df_reduced = df[main_features]
    df_reduced = df_reduced[:5000]
    return df_reduced

def reduced_df_stroke(df, sensible_attribute, target_variable):
  # Build a list with format (feature, correlation with target)
    features_corr = [(column, correlation) for column, correlation in zip(df.columns, df.corr()[target_variable])] # Modify stroke with the target_variable

    # Sort the features by correlation
    sorted_features = sorted(features_corr, key=lambda x: x[1], reverse=True)

    # Clean and take top 4
    main_features = [feature[0] for feature in sorted_features if feature[1] > 0][:5]

    # Add sensitive attribute
    main_features = main_features + [sensible_attribute]

    # Create a new reduced df
    df_reduced = df[main_features]
    return df_reduced

def extract_reduction(dataset_name, sensible_attribute):
  if dataset_name=='diabetes-prediction':
    reduce_df = reduce_df_diabetes_prediction
  elif dataset_name=='stroke-prediction':
    reduce_df = reduced_df_stroke
  return reduce_df

#MITIGATION UTILS
def fl_cr(config):
  mitigation='fl-cr'
  df, dataset_name, target_variable,sensible_attribute, path_to_project, n_splits, models, random_seed, privileged_groups, unprivileged_groups, default_mappings, reduced_df_techniques, params = unpack_config(config)
  if reduced_df_techniques is not None and mitigation in reduced_df_techniques:
    reduce_df = extract_reduction(dataset_name, sensible_attribute)
    df = reduce_df(df, sensible_attribute, target_variable)

  X_raw = df.drop(target_variable, axis=1)
  X_raw = pd.get_dummies(X_raw)
  y = df[target_variable].values

  cr = CorrelationRemover(sensitive_feature_ids=[sensible_attribute])
  X_cr = cr.fit_transform(X_raw)

  X_cr = pd.DataFrame(
      X_cr, columns = X_raw.drop(sensible_attribute, axis=1).columns
  )

  X_cr[sensible_attribute] = X_raw[sensible_attribute]
  mit_fl_cr = X_cr.copy(deep=True)
  mit_fl_cr[target_variable] = y

  save_mitigated_dataset(mit_fl_cr,path_to_project,dataset_name, sensible_attribute, mitigation)
  predictions_and_tests = compute_predictions_and_tests(mit_fl_cr, sensible_attribute, target_variable, n_splits, models)
  save_predictions_and_tests(predictions_and_tests, dataset_name, sensible_attribute, mitigation,path_to_project)
  return mit_fl_cr, predictions_and_tests

def fl_to(config):
  mitigation='fl-to'
  df, dataset_name, target_variable,sensible_attribute, path_to_project, n_splits, models, random_seed, privileged_groups, unprivileged_groups, default_mappings, reduced_df_techniques, params = unpack_config(config)
  if reduced_df_techniques is not None and mitigation in reduced_df_techniques:
    reduce_df = extract_reduction(dataset_name, sensible_attribute)
    df = reduce_df(df, sensible_attribute, target_variable)

  predictions_and_tests= {}

  for model_name in tqdm(models):
    classifier =  models[model_name]
    TO = ThresholdOptimizer(estimator=classifier)
    pred_and_y = fl_to_compute_predictions_and_tests(df, target_variable, sensible_attribute, n_splits, TO)
    predictions_and_tests[model_name] = pred_and_y

  save_predictions_and_tests(predictions_and_tests, dataset_name, sensible_attribute, mitigation, path_to_project)
  return predictions_and_tests

def aif360_rw(config):
  mitigation = 'aif360-rw'
  df, dataset_name, target_variable,sensible_attribute, path_to_project, n_splits, models, random_seed, privileged_groups, unprivileged_groups, default_mappings, reduced_df_techniques, params = unpack_config(config)
  if reduced_df_techniques is not None and mitigation in reduced_df_techniques:
    reduce_df = extract_reduction(dataset_name, sensible_attribute)
    df = reduce_df(df, sensible_attribute, target_variable)

  predictions_and_tests = {}
  for model_name in models:
    clf = models[model_name]
    RW = Reweighing(unprivileged_groups = unprivileged_groups,
                privileged_groups = privileged_groups)

    df_splitting = train_test_splitting(df, n_splits)
    pred_and_y = {}
    for i in range(0,n_splits):
      df_split = df_splitting[i]
      df_train = df_split['train']
      df_test = df_split['test']

      X_train, Y_train, X_test, Y_test = df_X_Y_split(df_train, df_test, target_variable)
      S_test = df_test[sensible_attribute].values

      data_orig_aif = BinaryLabelDataset(favorable_label = 1, unfavorable_label = 0, df = df_train.copy(), label_names = [target_variable], protected_attribute_names = [sensible_attribute])
      rw_dataset = RW.fit_transform(data_orig_aif)
      rw_df = rw_dataset.convert_to_dataframe()
      sample_weights= rw_df[1]["instance_weights"]


      clf.fit(X_train,Y_train,sample_weight=sample_weights)
      y_pred = clf.predict(X_test)
      pred_and_y[i] = {'y_test': Y_test.to_numpy().astype(int), 'y_pred': y_pred.astype(int), 's_test':  S_test.astype(int)}

      predictions_and_tests[model_name] = pred_and_y

  save_predictions_and_tests(predictions_and_tests, dataset_name, sensible_attribute, mitigation, path_to_project)

  return predictions_and_tests

def aif360_di(config):
  mitigation = 'aif360-di'
  df, dataset_name, target_variable,sensible_attribute, path_to_project, n_splits, models, random_seed, privileged_groups, unprivileged_groups, default_mappings, reduced_df_techniques, params = unpack_config(config)
  if reduced_df_techniques is not None and mitigation in reduced_df_techniques:
    reduce_df = extract_reduction(dataset_name, sensible_attribute)
    df = reduce_df(df, sensible_attribute, target_variable)

  predictions_and_tests = {}

  #levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  DIR = DisparateImpactRemover(repair_level=0.5)
  data_orig_aif = BinaryLabelDataset(favorable_label = 1, unfavorable_label = 0, df = df.copy(), label_names = [target_variable], protected_attribute_names = [sensible_attribute])

  train_repd = DIR.fit_transform(data_orig_aif)
  mit_aif360_di = train_repd.convert_to_dataframe()[0]

  save_mitigated_dataset(mit_aif360_di,path_to_project,dataset_name, sensible_attribute, mitigation)

  predictions_and_tests = compute_predictions_and_tests(mit_aif360_di, sensible_attribute, target_variable, n_splits, models)

  save_predictions_and_tests(predictions_and_tests, dataset_name, sensible_attribute, mitigation, path_to_project)

  return predictions_and_tests

def aif360_lfr(config):
  mitigation = 'aif360-lfr'
  df, dataset_name, target_variable,sensible_attribute, path_to_project, n_splits, models, random_seed, privileged_groups, unprivileged_groups, default_mappings, reduced_df_techniques, params = unpack_config(config)
  if reduced_df_techniques is not None and mitigation in reduced_df_techniques:
    reduce_df = extract_reduction(dataset_name, sensible_attribute)
    df = reduce_df(df, sensible_attribute, target_variable)

  predictions_and_tests = {}

  #TR = LFR(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, seed= random_seed, k=10, Ax=0.01, Ay=1.0, Az=50.0, verbose=1)
  TR = LFR(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, seed= random_seed, k=params['k'], verbose=1)
  data_orig_aif = BinaryLabelDataset(favorable_label = 1, unfavorable_label = 0, df = df.copy(), label_names = [target_variable], protected_attribute_names = [sensible_attribute])

  #TR = TR.fit(data_orig_aif, maxiter=5000, maxfun=5000)
  TR = TR.fit(data_orig_aif, maxiter=params['max_iter_lfr'], maxfun=params['max_iter_lfr']) 
  transf_dataset = TR.transform(data_orig_aif)
  mit_aif360_lfr = transf_dataset.convert_to_dataframe()[0]

  save_mitigated_dataset(mit_aif360_lfr,path_to_project,dataset_name, sensible_attribute, mitigation)

  predictions_and_tests = compute_predictions_and_tests(mit_aif360_lfr, sensible_attribute, target_variable, n_splits, models)

  save_predictions_and_tests(predictions_and_tests, dataset_name, sensible_attribute, mitigation, path_to_project)

  return predictions_and_tests


def distortion_function_diabetes_women(df, sensible_attribute):
  df2=df.drop(columns=['Pregnancies','DiabetesPedigreeFunction'])
  #df2 = df.copy()
  df2['BMI'] = np.where(df2['BMI'].between(18,25), 1, 0) # 1 valore giusto, 0 anomalo
  df2['Glucose'] = np.where(df2['Glucose'] < 120, 1, 0)
  df2["BloodPressure"]=np.where(df2["BloodPressure"].between(60,80),1,0)
  df2["SkinThickness"]=np.where(df2["SkinThickness"].between(22,24),1,0)
  df2["Insulin"]=np.where(df2["Insulin"].between(5,25),1,0)
  return df2

def get_distortion_diabetes_women(vold,vnew):
        bad_val=3
        OutNew=vnew["Outcome"]
        OutOld=vold["Outcome"]
        # GluNew=vnew["Glucose"]
        # GluOld=vold["Glucose"]
        InsOld=vold["Insulin"]
        InsNew=vnew["Insulin"]

        if ((OutNew>OutOld)& (InsNew<InsOld)): #| ((OutNew>OutOld) & (GluNew<GluOld)):
            return bad_val
        else:
            return 0
        
def distortion_function_diabetes_prediction(df, sensible_attribute):
  df2 = df.copy()
  df2['time_in_hospital'] = np.where(df2['time_in_hospital'].between(1,6), 1, 0) # 1 valore giusto, 0 anomalo
  df2["number_inpatient"]=np.where(df2["number_inpatient"] < 1, 1, 0)
  df2["number_diagnoses"]=np.where(df2["number_diagnoses"].between(1,10), 1, 0)

#   if not apply_dimensionality_reduction:
#     df2 = df2.drop(columns=['glipizide_category', 'glyburide_category']) # drop non-binary columns that are full of zero's
#     df2["num_procedures"]=np.where(df2["num_procedures"] < 5, 1, 0)
#     df2["num_medications"]=np.where(df2["num_medications"] < 20, 1, 0)
#     df2['num_lab_procedures'] = np.where(df2['num_lab_procedures'] < 50, 1, 0) # based on distribution

  return df2

def get_distortion_diabetes_prediction(vold,vnew): # NEED TO CHANGE THIS
        bad_val=3
        OutNew=vnew["diabetes"]
        OutOld=vold["diabetes"]

        # if not apply_dimensionality_reduction:
        #   InsOld=vold["num_lab_procedures"]
        #   InsNew=vnew["num_lab_procedures"]
        # else:
        InsOld = 0
        InsNew = 0

        if ((OutNew>OutOld)& (InsNew<InsOld)): 
            return bad_val
        else:
            return 0

def get_distortion_stroke(vold,vnew):
        bad_val=3
        OutNew=vnew["stroke_prediction"]
        OutOld=vold["stroke_prediction"]

        # if not apply_dimensionality_reduction:
        #   InsOld=vold["avg_glucose_level"]
        #   InsNew=vnew["avg_glucose_level"]
        # else:
        InsOld = 0
        InsNew = 0

        if ((OutNew>OutOld) & (InsNew<InsOld)):
            return bad_val
        else:
            return 0
        
def distortion_function_stroke(df, sensible_attribute):
  df2 = df.copy() # drop non-binary columns that are full of zero's
#   if not apply_dimensionality_reduction:
#     df2['avg_glucose_level'] = np.where(df2['avg_glucose_level'] < 105, 1, 0) # 1 valore giusto, 0 anomalo
#     df2['bmi'] = np.where(df2['bmi'].between(15, 30), 1, 0) # based on distribution
  return df2

def get_distortion_sepsis(vold,vnew):   #calculates distance between two instances of 'target'
        bad_val=3
        OutNew=vnew["Mortality"]
        OutOld=vold["Mortality"]

        if (OutNew>OutOld): #| ((OutNew>OutOld) & (GluNew<GluOld)):
            return bad_val
        else:
            return 0

def distortion_function_sepsis(df, sensible_attribute):
  df2=df.drop(columns=['WBCC','NeuC','LymC','NLCR','PLTC','MPV'])
  # keeping attributes 'APACHE II','SOFA', 'Group', 'LOS-ICU', and 'gender_cat' or 'age_cat', the one that is not used as a sensible_attribute
  # the values used for the binarization are taken from the preprocessing: they correspond to the median of the values of the attributes
  #for each non-binary attribute, make it binary
  df2['APACHE II'] = np.where(df2['APACHE II'] < 11, 1, 0)
  df2['SOFA'] = np.where(df2['SOFA'] < 1, 1, 0)
  #df2['Group'] is already binary
  df2['LOS-ICU'] = np.where(df2['LOS-ICU'] < 1, 1, 0)
  return df2

def get_distortion_aids(vold,vnew):   #calculates distance between two instances of target
        bad_val=3
        OutNew=vnew["cid"]
        OutOld=vold["cid"]

        if (OutNew>OutOld):
            return bad_val
        else:
            return 0

def distortion_function_aids(df, sensible_attribute):
  list_sensible = ['race_cat', 'homo_cat', 'age_cat']
  if sensible_attribute in list_sensible:
    list_sensible.remove(sensible_attribute)

  df2=df.drop(columns=['pidnum', 'trt', 'wtkg', 'hemo', 'drugs', 'karnof','z30', 'oprior', 'preanti', 'gender', 'str2', 'treat', 'offtrt', 'cd80', 'cd820'])
  df2 = df2.drop(columns = list_sensible)
  # keeping attributes 'time', 'strat', 'symptom', 'cd40', 'cd420'    #works with 6 attributes too (z30)
  # the values used for the binarization are taken from the preprocessing: they correspond to the median of the values of the attributes
  #for each non-binary attribute, make it binary
  df2['time'] = np.where(df2['time'] < 997, 1, 0)
  df2['strat'] = np.where(df2['strat'].between(1,2), 1, 0)
  df2['cd40'] = np.where(df2['cd40'] < 340, 1, 0)
  df2['cd420'] = np.where(df2['cd420'] < 353, 1, 0)
  return df2

def distortion_function_alzheimer(df, sensible_attribute):
  # Keep the 5 most correlated columns together with the sensitive attribute.
  cols_to_keep = ['Diagnosis', 'MemoryComplaints', 'BehavioralProblems', 'CholesterolHDL',
                  'Hypertension', sensible_attribute]
  df2 = df[cols_to_keep].copy()

  # Binarization of attributes based on typical values or literals
  df2['MemoryComplaints'] = np.where(df2['MemoryComplaints'] == 1, 1, 0)
  df2['BehavioralProblems'] = np.where(df2['BehavioralProblems'] == 1, 1, 0)
  df2['CholesterolHDL'] = np.where(df2['CholesterolHDL'] < 60, 1, 0)  # Healthy HDL > 60
  df2['Hypertension'] = np.where(df2['Hypertension'] == 1, 1, 0)  # 1=High pressure

  return df2 
   
def get_distortion_alzheimer(vold,vnew):
  bad_val = 3
  OutNew = vnew["Diagnosis"]
  OutOld = vold["Diagnosis"]
  #MemNew = vnew["MemoryComplaints"]
  #MemOld = vold["MemoryComplaints"]
  #BehNew = vnew["BehavioralProblems"]
  #BehOld = vold["BehavioralProblems"]

  if (OutNew > OutOld): # & (MemNew < MemOld)) | ((OutNew > OutOld) & (BehNew < BehOld)):
      return bad_val
  else:
      return 0
   
def distortion_function_myocardial(df, sensible_attribute):
    # Keep the 5 most correlated columns together with the sensitive attribute.
    cols_to_keep = ['LET_IS_cat', 'RAZRIV', 'REC_IM', 'ZSN_A', 'AGE', 'SEX']
    df2 = df[cols_to_keep].copy()

    # Binarization of attributes based on typical values or literals
    df2['RAZRIV'] = np.where(df2['RAZRIV'] == 1, 1, 0)
    df2['REC_IM'] = np.where(df2['REC_IM'] == 1, 1, 0)
    df2['ZSN_A'] = np.where(df2['ZSN_A'] > 1, 1, 0)  # Presence (1) or absence (0) of heart failure
    df2['AGE'] = np.where(df2['AGE'] >= 60, 1, 0)  # Age over or under 60 years old
    df2[sensible_attribute] = np.where(df2[sensible_attribute] == 1, 1, 0)  # Sex: 1=Male, 0=Female

    return df2

def get_distortion_myocardial(vold,vnew):
    bad_val = 3
    OutNew = vnew["LET_IS_cat"]
    OutOld = vold["LET_IS_cat"]
    #RazNew = vnew["RAZRIV"]
    #RazOld = vold["RAZRIV"]
    #RecNew = vnew["REC_IM"]
    #RecOld = vold["REC_IM"]

    if (OutNew>OutOld): #& (RazNew < RazOld)) #| ((OutNew > OutOld) & (RecNew < RecOld)):
        return bad_val
    else:
        return 0

def extract_op_functions(dataset_name, sensible_attribute):
    if dataset_name=='diabetes-women':
        distortion_function = distortion_function_diabetes_women
        get_distortion_function = get_distortion_diabetes_women
    elif dataset_name=='diabetes-prediction':
        distortion_function = distortion_function_diabetes_prediction
        get_distortion_function = get_distortion_diabetes_prediction
    elif dataset_name=='stroke-prediction':
        distortion_function = distortion_function_stroke
        get_distortion_function = get_distortion_stroke
    elif dataset_name=='sepsis':
        distortion_function = distortion_function_sepsis
        get_distortion_function = get_distortion_sepsis
    elif dataset_name=='aids':
        distortion_function = distortion_function_aids
        get_distortion_function = get_distortion_aids
    elif dataset_name=='myocardial-infarction':
        distortion_function = distortion_function_myocardial
        get_distortion_function = get_distortion_myocardial
    elif dataset_name=='alzheimer-disease':
        distortion_function = distortion_function_alzheimer
        get_distortion_function = get_distortion_alzheimer
       
    return distortion_function, get_distortion_function
  
def aif360_op(config):
  mitigation = 'aif360-op'
  df, dataset_name, target_variable,sensible_attribute, path_to_project, n_splits, models, random_seed, privileged_groups, unprivileged_groups, default_mappings, reduced_df_techniques, params = unpack_config(config)
  if reduced_df_techniques is not None and mitigation in reduced_df_techniques:
    reduce_df = extract_reduction(dataset_name, sensible_attribute)
    df = reduce_df(df, sensible_attribute, target_variable)

  distortion_function, get_distortion_function = extract_op_functions(dataset_name, sensible_attribute)

  df_op = distortion_function(df, sensible_attribute)

  data_orig_aif_op = BinaryLabelDataset(
    favorable_label = 1,
    unfavorable_label = 0,
    df = df_op.copy(),
    label_names = [target_variable],
    protected_attribute_names = [sensible_attribute],
    metadata = default_mappings)

  optim_options = {
    "distortion_fun": get_distortion_function,
    "epsilon": 0.05,
    "clist": [0.99, 1.99, 2.99],
    "dlist": [.1, 0.05, 0]
  }

  predictions_and_tests = {}
  OP = OptimPreproc(OptTools, optim_options)
  OP_fitted= OP.fit(data_orig_aif_op)
  transf_dataset = OP_fitted.transform(data_orig_aif_op, transform_Y=True)
  mit_aif360_op = transf_dataset.convert_to_dataframe()[0]

  save_mitigated_dataset(mit_aif360_op,path_to_project,dataset_name, sensible_attribute, mitigation)

  predictions_and_tests = compute_predictions_and_tests(mit_aif360_op, sensible_attribute, target_variable, n_splits, models)

  save_predictions_and_tests(predictions_and_tests, dataset_name, sensible_attribute, mitigation, path_to_project)

  return predictions_and_tests

def aif360_pr(config):
  mitigation = 'aif360-pr'
  df, dataset_name, target_variable,sensible_attribute, path_to_project, n_splits, models, random_seed, privileged_groups, unprivileged_groups, default_mappings, reduced_df_techniques, params = unpack_config(config)
  if reduced_df_techniques is not None and mitigation in reduced_df_techniques:
    reduce_df = extract_reduction(dataset_name, sensible_attribute)
    df = reduce_df(df, sensible_attribute, target_variable)

  predicted_and_real_tests = {}

  df_splitting = train_test_splitting(df, n_splits)
  pred_and_y = {}
  for i in range(0,n_splits):

    PR = PrejudiceRemover(eta=1.0, sensitive_attr=sensible_attribute, class_attr=target_variable)

    df_split = df_splitting[i]
    df_train = df_split['train']
    df_test = df_split['test']

    X_train, Y_train, X_test, Y_test, S_train, S_test = df_X_Y_split_S(df_train, df_test, target_variable, sensible_attribute)

    data_orig_aif_train = BinaryLabelDataset(favorable_label = 1,
      unfavorable_label = 0,
      df = df_train.copy(),
      label_names = [target_variable],
      protected_attribute_names = [sensible_attribute])

    data_orig_aif_test = BinaryLabelDataset(favorable_label = 1,
      unfavorable_label = 0,
      df = df_test.copy(),
      label_names = [target_variable],
      protected_attribute_names = [sensible_attribute])

    PR.fit(data_orig_aif_train)
    predictions = PR.predict(data_orig_aif_test)
    mit_aif360_pr = predictions.convert_to_dataframe()[0]

    y_pred = mit_aif360_pr[target_variable]
    pred_and_y[i] = {'y_test': Y_test.to_numpy(), 'y_pred': y_pred.to_numpy().astype(int), 's_test':  S_test}

    predicted_and_real_tests = pred_and_y

  save_predictions_and_tests(predicted_and_real_tests, dataset_name, sensible_attribute, mitigation, path_to_project)
  return predicted_and_real_tests

def aif360_er(config):
  mitigation = 'aif360-er'
  df, dataset_name, target_variable,sensible_attribute, path_to_project, n_splits, models, random_seed, privileged_groups, unprivileged_groups, default_mappings, reduced_df_techniques, params = unpack_config(config)
  if reduced_df_techniques is not None and mitigation in reduced_df_techniques:
    reduce_df = extract_reduction(dataset_name, sensible_attribute)
    df = reduce_df(df, sensible_attribute, target_variable)

  df_splitting = train_test_splitting(df, n_splits)
  pred_and_y = {}
  predictions_and_tests = {}
  for i in range(0,n_splits):

    df_split = df_splitting[i]
    df_train = df_split['train']
    df_test = df_split['test']

    X_train, Y_train, X_test, Y_test, S_train, S_test = df_X_Y_split_S(df_train, df_test, target_variable, sensible_attribute)

    data_orig_aif_train = BinaryLabelDataset(favorable_label = 1,
      unfavorable_label = 0,
      df = df_train.copy(),
      label_names = [target_variable],
      protected_attribute_names = [sensible_attribute])

    data_orig_aif_test = BinaryLabelDataset(favorable_label = 1,
      unfavorable_label = 0,
      df = df_test.copy(),
      label_names = [target_variable],
      protected_attribute_names = [sensible_attribute])

    #lmod = DecisionTreeClassifier(max_depth=None)
    #lmod = LogisticRegression(solver='lbfgs', max_iter=1000)

    exp_grad_red = ExponentiatedGradientReduction(estimator=params['lmod'], constraints='DemographicParity', drop_prot_attr=False)
    exp_grad_red.fit(data_orig_aif_train)
    predictions = exp_grad_red.predict(data_orig_aif_test)

    mit_aif360_er = predictions.convert_to_dataframe()[0]
    y_pred = mit_aif360_er[target_variable]
    pred_and_y[i] = {'y_test': Y_test.to_numpy(), 'y_pred': y_pred.to_numpy().astype(int), 's_test':  S_test}

  predictions_and_tests = pred_and_y

  save_predictions_and_tests(predictions_and_tests, dataset_name, sensible_attribute, mitigation, path_to_project)
  return predictions_and_tests

def aif360_ad(config):
  mitigation = 'aif360-ad'
  df, dataset_name, target_variable,sensible_attribute, path_to_project, n_splits, models, random_seed, privileged_groups, unprivileged_groups, default_mappings, reduced_df_techniques, params = unpack_config(config)
  if reduced_df_techniques is not None and mitigation in reduced_df_techniques:
    reduce_df = extract_reduction(dataset_name, sensible_attribute)
    df = reduce_df(df, sensible_attribute, target_variable)

  predicted_and_real_tests = {}

  df_splitting = train_test_splitting(df, n_splits)
  pred_and_y = {}
  for i in range(0,n_splits):
    # Create a new session for each fold
    sess = tf.Session()
    # Added a unique scope name using the fold index
    AD = AdversarialDebiasing(privileged_groups = privileged_groups,
                        unprivileged_groups = unprivileged_groups,
                        scope_name=f'plain_classifier_{i}', # Changed scope name
                        debias=False,
                        sess=sess)

    df_split = df_splitting[i]
    df_train = df_split['train']
    df_test = df_split['test']

    X_train, Y_train, X_test, Y_test, S_train, S_test = df_X_Y_split_S(df_train, df_test, target_variable, sensible_attribute)

    data_orig_aif_train = BinaryLabelDataset(favorable_label = 1,
      unfavorable_label = 0,
      df = df_train.copy(),
      label_names = [target_variable],
      protected_attribute_names = [sensible_attribute])

    data_orig_aif_test = BinaryLabelDataset(favorable_label = 1,
      unfavorable_label = 0,
      df = df_test.copy(),
      label_names = [target_variable],
      protected_attribute_names = [sensible_attribute])

    AD.fit(data_orig_aif_train)

    predictions = AD.predict(data_orig_aif_test)

    mit_aif360_ad = predictions.convert_to_dataframe()[0]
    y_pred = mit_aif360_ad[target_variable]
    pred_and_y[i] = {'y_test': Y_test.to_numpy(), 'y_pred': y_pred.to_numpy().astype(int), 's_test':  S_test}

    predicted_and_real_tests = pred_and_y

    # Close the session after each fold
    sess.close()
  save_predictions_and_tests(predicted_and_real_tests, dataset_name, sensible_attribute, mitigation, path_to_project)
  return predicted_and_real_tests

def compute_predictions(x_train, y_train, x_test, model):
  model.fit(x_train, y_train)
  return model.predict(x_test)

def aif360_ce(config):
  mitigation = 'aif360-ce'
  df, dataset_name, target_variable,sensible_attribute, path_to_project, n_splits, models, random_seed, privileged_groups, unprivileged_groups, default_mappings, reduced_df_techniques, params = unpack_config(config)
  if reduced_df_techniques is not None and mitigation in reduced_df_techniques:
    reduce_df = extract_reduction(dataset_name, sensible_attribute)
    df = reduce_df(df, sensible_attribute, target_variable)

  predictions_and_tests = {}
  for model_name in models:
    clf = models[model_name]
    CEPP = CalibratedEqOddsPostprocessing(unprivileged_groups= unprivileged_groups, privileged_groups = privileged_groups, cost_constraint='weighted', seed=random_seed)

    df_splitting = train_test_splitting(df, n_splits)
    pred_and_y = {}
    for i in range(0,n_splits):
      df_split = df_splitting[i]
      df_train = df_split['train']
      df_test = df_split['test']

      X_train, Y_train, X_test, Y_test = df_X_Y_split(df_train, df_test, target_variable)
      S_test = df_test[sensible_attribute].values

      # Compute the prediction for this model
      Y_pred = compute_predictions(X_train, Y_train, X_test, clf)

      # Define a new data frame with the predicted values as labels
      df_pred = df.loc[Y_test.index]
      df_pred[target_variable] = Y_pred


      # Create the aif datasets with y_test and y_pred data frames
      data_orig_aif_test = BinaryLabelDataset(favorable_label = 1,
                                              unfavorable_label = 0,
                                              df = df.iloc[Y_test.index],
                                              label_names = [target_variable],
                                              protected_attribute_names =
                                               [sensible_attribute])

      data_orig_aif_pred = BinaryLabelDataset(favorable_label = 1,
                                              unfavorable_label = 0,
                                              df = df_pred.copy(),
                                              label_names = [target_variable],
                                              protected_attribute_names =
                                               [sensible_attribute])


      # Fit the post-processing technique with these two datasets i.e. test, pred
      predictions = CEPP.fit_predict(data_orig_aif_test, data_orig_aif_pred)
      mit_aif360_cpp = predictions.convert_to_dataframe()[0]
      y_pred = mit_aif360_cpp[target_variable]
      pred_and_y[i] = {'y_test': Y_test.to_numpy(), 'y_pred': y_pred.to_numpy().astype(int), 's_test':  S_test}

      predictions_and_tests[model_name] = pred_and_y

  save_predictions_and_tests(predictions_and_tests, dataset_name, sensible_attribute, mitigation, path_to_project)

  return predictions_and_tests

def aif360_eo(config):
  mitigation = 'aif360-eo'
  df, dataset_name, target_variable,sensible_attribute, path_to_project, n_splits, models, random_seed, privileged_groups, unprivileged_groups, default_mappings, reduced_df_techniques, params = unpack_config(config)
  if reduced_df_techniques is not None and mitigation in reduced_df_techniques:
    reduce_df = extract_reduction(dataset_name, sensible_attribute)
    df = reduce_df(df, sensible_attribute, target_variable)

  predictions_and_tests = {}
  for model_name in models:
    clf = models[model_name]
    EOPP = EqOddsPostprocessing(unprivileged_groups= unprivileged_groups, privileged_groups = privileged_groups, seed=random_seed)

    df_splitting = train_test_splitting(df, params['n_splits_eo'])
    pred_and_y = {}
    for i in range(0,params['n_splits_eo']):
      df_split = df_splitting[i]
      df_train = df_split['train']
      df_test = df_split['test']

      X_train, Y_train, X_test, Y_test = df_X_Y_split(df_train, df_test, target_variable)
      S_test = df_test[sensible_attribute].values

      Y_pred = compute_predictions(X_train, Y_train, X_test, clf)

      # Define a new data frame with the predicted values as labels
      df_pred = df.loc[Y_test.index]
      df_pred[target_variable] = Y_pred


      # Create the aif datasets with y_test and y_pred data frames
      data_orig_aif_test = BinaryLabelDataset(favorable_label = 1,
                                              unfavorable_label = 0,
                                              df = df.iloc[Y_test.index],
                                              label_names = [target_variable],
                                              protected_attribute_names =
                                               [sensible_attribute])

      data_orig_aif_pred = BinaryLabelDataset(favorable_label = 1,
                                              unfavorable_label = 0,
                                              df = df_pred.copy(),
                                              label_names = [target_variable],
                                              protected_attribute_names =
                                               [sensible_attribute])


      # Fit the post-processing technique with these two datasets i.e. test, pred
      predictions = EOPP.fit_predict(data_orig_aif_test, data_orig_aif_pred)

      mit_aif360_eopp = predictions.convert_to_dataframe()[0]
      y_pred = mit_aif360_eopp[target_variable]
      pred_and_y[i] = {'y_test': Y_test.to_numpy(), 'y_pred': y_pred.to_numpy().astype(int), 's_test':  S_test}

      predictions_and_tests[model_name] = pred_and_y

  save_predictions_and_tests(predictions_and_tests, dataset_name, sensible_attribute, mitigation, path_to_project)

  return predictions_and_tests

def aif360_roc(config):
  mitigation = 'aif360-roc'
  df, dataset_name, target_variable,sensible_attribute, path_to_project, n_splits, models, random_seed, privileged_groups, unprivileged_groups, default_mappings, reduced_df_techniques, params = unpack_config(config)
  if reduced_df_techniques is not None and mitigation in reduced_df_techniques:
    reduce_df = extract_reduction(dataset_name, sensible_attribute)
    df = reduce_df(df, sensible_attribute, target_variable)

  predictions_and_tests = {}
  for model_name in models:
    clf = models[model_name]
    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                 privileged_groups=privileged_groups,
                                 low_class_thresh=0.01, high_class_thresh=0.99,
                                  num_class_thresh=100, num_ROC_margin=50,
                                  metric_name=params['metric_name_roc'], metric_ub=0.05, metric_lb=-0.05)

    df_splitting = train_test_splitting(df, n_splits)
    pred_and_y = {}
    for i in range(0,n_splits):
      df_split = df_splitting[i]
      df_train = df_split['train']
      df_test = df_split['test']

      X_train, Y_train, X_test, Y_test = df_X_Y_split(df_train, df_test, target_variable)
      S_test = df_test[sensible_attribute].values.astype(int)

      # Compute the prediction for this model
      Y_pred = compute_predictions(X_train, Y_train, X_test, clf)

      # Define a new data frame with the predicted values as labels
      df_pred = df.loc[Y_test.index]
      df_pred[target_variable] = Y_pred

      # Create the aif datasets with y_test and y_pred data frames
      data_orig_aif_test = BinaryLabelDataset(favorable_label = 1,
                                              unfavorable_label = 0,
                                              df = df.iloc[Y_test.index],
                                              label_names = [target_variable],
                                              protected_attribute_names =
                                               [sensible_attribute])

      data_orig_aif_pred = BinaryLabelDataset(favorable_label = 1,
                                              unfavorable_label = 0,
                                              df = df_pred.copy(),
                                              label_names = [target_variable],
                                              protected_attribute_names =
                                               [sensible_attribute])

      # Fit the post-processing technique with these two datasets i.e. test, pred
      predictions = ROC.fit_predict(data_orig_aif_test, data_orig_aif_pred)
      mit_aif360_roc = predictions.convert_to_dataframe()[0]
      y_pred = mit_aif360_roc[target_variable]
      pred_and_y[i] = {'y_test': Y_test.to_numpy(), 'y_pred': y_pred.to_numpy().astype(int), 's_test':  S_test}

      predictions_and_tests[model_name] = pred_and_y

  save_predictions_and_tests(predictions_and_tests, dataset_name, sensible_attribute, mitigation, path_to_project)

  return predictions_and_tests