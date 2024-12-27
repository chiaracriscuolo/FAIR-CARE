#import libraries
import numpy as np
import pandas as pd
#from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
from source.utils.config import *

#UTILS TRAIN_TEST SPLIT
def train_test_splitting(df, n_splits):
  df_splitting = {}

  w = int(len(df)/n_splits)
  window = w
  start_point = 0
  for i in range(0,n_splits):
      df_train = {}
      df_test = {}
      df_train_1 = {}
      df_train_2 = {}
      df_test = df[start_point:window]
      if i != 0:
        df_train_1 = df[0: start_point]

      if i != n_splits-1:
        df_train_2 = df[window: len(df)]

      if (i != 0 and  i != n_splits-1):
        concat_df = [df_train_1, df_train_2]
        df_train = pd.concat(concat_df)
      elif i != 0:
        df_train = df_train_1
      else:
        df_train = df_train_2

      start_point= window
      window = window + w

      df_splitting[i] = {'train': df_train, 'test': df_test}
  return df_splitting

def df_X_Y_split(df_train, df_test, target_variable):
  Y_train = df_train[target_variable]
  X_train = df_train.drop(target_variable, axis=1)
  Y_test = df_test[target_variable]
  X_test = df_test.drop(target_variable, axis=1)
  return X_train, Y_train, X_test, Y_test

def compute_predictions_and_tests(df, sensible_attribute, target_variable, n_splits, models, n_estimators, random_seed):
  predicted_and_real_values = {}
  for model_name in tqdm(models):
    clf = models[model_name]
    df_splitting = train_test_splitting(df, n_splits)
    pred_and_y = {}
    for i in range(0,n_splits):
      df_split = df_splitting[i]
      df_train = df_split['train']
      df_test = df_split['test']

      X_train, Y_train, X_test, Y_test = df_X_Y_split(df_train, df_test, target_variable)
      clf.fit(X_train,Y_train)
      y_pred = clf.predict(X_test)

      S_test = X_test[sensible_attribute].values
      pred_and_y[i] = {'y_test': Y_test.to_numpy().astype(int), 'y_pred': y_pred.astype(int), 's_test':  S_test.astype(int)}

    predicted_and_real_values[model_name] = pred_and_y

  return predicted_and_real_values


#UTIL PERFORMANCE
def compute_mean_std_dev(metric_list, models):
  metric_dict = {}
  if models is not None:
    for model_name in (models):
      metric = np.array(metric_list[model_name])
      mean_metric = metric.mean()
      std_metric = metric.std()
      metric_dict[model_name] = [mean_metric, std_metric]
  else:
    metric = np.array(metric_list)
    mean_metric = metric.mean()
    std_metric = metric.std()
    metric_dict = [mean_metric, std_metric]
  return metric_dict


def compute_performance_metrics(predictions_and_tests, models, n_splits, mitigation):
  accuracy, precision, recall, f1_score = compute_scores(predictions_and_tests, models, n_splits, mitigation)

  if mitigation not in without_model_mitigations:
    #for each model compute mean and standard deviation
    acc = compute_mean_std_dev(accuracy, models)
    prec = compute_mean_std_dev(precision, models)
    rec = compute_mean_std_dev(recall, models)
    f1 = compute_mean_std_dev(f1_score, models)
  else:
    acc = compute_mean_std_dev(accuracy, None)
    prec = compute_mean_std_dev(precision, None)
    rec = compute_mean_std_dev(recall, None)
    f1 = compute_mean_std_dev(f1_score, None)

  performance_metrics = {}
  performance_metrics['accuracy'] = acc
  performance_metrics['precision'] = prec
  performance_metrics['recall'] = rec
  performance_metrics['f1_score'] = f1

  return performance_metrics

# def compute_performance_metrics(accuracy, precision, recall, f1_score):
#   performance_metrics = {}
#   performance_metrics['accuracy'] = compute_mean_std_dev(accuracy,models)
#   performance_metrics['precision'] = compute_mean_std_dev(precision, models)
#   performance_metrics['recall'] = compute_mean_std_dev(recall, models)
#   performance_metrics['f1_score'] = compute_mean_std_dev(f1_score, models)
#   return performance_metrics

def compute_scores(predictions_and_tests, models, n_splits, mitigation):
  precision = {}
  recall = {}
  accuracy = {}
  f1_score = {}

  if mitigation not in without_model_mitigations:
    for model_name in (models):

      precisions = []
      recalls = []
      accuracys = []
      f1_scores = []
      for i in range(0,n_splits):
        y_test = predictions_and_tests[model_name][i]['y_test']
        y_pred = predictions_and_tests[model_name][i]['y_pred']
        precisions.append(metrics.precision_score(y_test, y_pred))
        recalls.append(metrics.recall_score(y_test, y_pred))
        accuracys.append(metrics.accuracy_score(y_test, y_pred))
        f1_scores.append(metrics.f1_score(y_test, y_pred))
      precision[model_name] = precisions
      recall[model_name] = recalls
      accuracy[model_name] = accuracys
      f1_score[model_name] = f1_scores
  else:
    precisions = []
    recalls = []
    accuracys = []
    f1_scores = []
    for i in range(0,n_splits):
        y_test = predictions_and_tests[i]['y_test']
        y_pred = predictions_and_tests[i]['y_pred']
        precisions.append(metrics.precision_score(y_test, y_pred))
        recalls.append(metrics.recall_score(y_test, y_pred))
        accuracys.append(metrics.accuracy_score(y_test, y_pred))
        f1_scores.append(metrics.f1_score(y_test, y_pred))
    precision = precisions
    recall = recalls
    accuracy = accuracys
    f1_score = f1_scores
  return accuracy, precision, recall, f1_score

#UTILS FAIRNESS METRICS
# We need to save the indexes of both groups `privileged` and `discriminated` in two lists.
# `y_privileged` is the part the dataset where `sensible_value` = 1 (for example `AgeCategory` = 1), and `y_discriminated` is the part of dataset where `sensible_value` = 0.
# Build the confusion matrices (one for the privileged group, one for the discriminated group) for each model. 

def compute_confusion_matrices(predictions_and_tests, target_variable_labels, models, n_splits, mitigation):
  confusion_matrices = {}
  if mitigation not in without_model_mitigations:
    for model_name in (models):
      cm_splits = {}
      for i in range(0,n_splits):
        temp_dict = {}
        cm_privileged = {}
        cm_discriminated = {}
        y_test = predictions_and_tests[model_name][i]['y_test']
        y_pred = predictions_and_tests[model_name][i]['y_pred']
        s_test = predictions_and_tests[model_name][i]['s_test']

        df_metrics = pd.DataFrame({'s_test': s_test, 'y_test':y_test, 'y_pred':y_pred})

        df_discrim = df_metrics[df_metrics['s_test'] == 0]
        #len_dicr = len(df_discrim)
        df_priv = df_metrics[df_metrics['s_test'] == 1]
        #len_priv = len(df_priv)

        cm_discriminated = confusion_matrix(df_discrim['y_test'], df_discrim['y_pred'], labels=target_variable_labels)
        cm_privileged = confusion_matrix(df_priv['y_test'], df_priv['y_pred'], labels=target_variable_labels)
        temp_dict['discriminated'] = cm_discriminated
        temp_dict['privileged'] = cm_privileged
        cm_splits[i] = temp_dict
      confusion_matrices[model_name] = cm_splits
  else:
    cm_splits = {}
    for i in range(0,n_splits):
      temp_dict = {}
      cm_privileged = {}
      cm_discriminated = {}
      y_test = predictions_and_tests[i]['y_test']
      y_pred = predictions_and_tests[i]['y_pred']
      s_test = predictions_and_tests[i]['s_test']

      df_metrics = pd.DataFrame({'s_test': s_test, 'y_test':y_test, 'y_pred':y_pred})

      df_discrim = df_metrics[df_metrics['s_test'] == 0]
      #len_dicr = len(df_discrim)
      df_priv = df_metrics[df_metrics['s_test'] == 1]
      #len_priv = len(df_priv)

      cm_discriminated = confusion_matrix(df_discrim['y_test'], df_discrim['y_pred'], labels=target_variable_labels)
      cm_privileged = confusion_matrix(df_priv['y_test'], df_priv['y_pred'], labels=target_variable_labels)
      temp_dict['discriminated'] = cm_discriminated
      temp_dict['privileged'] = cm_privileged
      cm_splits[i] = temp_dict
      confusion_matrices = cm_splits
  return confusion_matrices

# Retrieve TP, TN, FP, FN values from a confusion matrix
def retrieve_values(cm):
  TN = cm[0][0]
  FP = cm[0][1]
  FN = cm[1][0]
  TP = cm[1][1]
  total = TN+FP+FN+TP
  return TP, TN, FP, FN, total

def rescale(metric):
  rescaled_metric = (metric - 1)/(metric + 1)
  return rescaled_metric

def standardization(metric):
  if metric > 1:
    metric = 1
  elif metric < -1:
    metric = -1
  return metric

def valid(metric, th):
  if metric > 1-th and metric < 1+th:
    return True
  return False

def sub_valid(metric, th):
  if metric > -th and metric < th:
    return True
  return False

def sub_and_function2(m1, m2):
  return m1 if abs(m1) > abs(m2) else m2

def and_function(m1,m2, th):
  m1_new = rescale(m1)
  m2_new = rescale(m2)
  m = sub_and_function(m1_new, m2_new, th)
  return m+1

def sub_and_function(m1, m2, th):
  max_value = max(abs(m1), abs(m2))
  if max_value == abs(m1):
    return m1
  else:
    return m2
  
# Fairness metrics computed using division operator
def fairness_metrics_division(confusion_matrix, threshold = 0.15):

  TP_priv, TN_priv, FP_priv, FN_priv, len_priv = retrieve_values(confusion_matrix['privileged'])
  TP_discr, TN_discr, FP_discr, FN_discr, len_discr = retrieve_values(confusion_matrix['discriminated'])

  GroupFairness_discr = (TP_discr+FP_discr)/len_discr
  GroupFairness_priv = (TP_priv+FP_priv)/len_priv
  if GroupFairness_priv == 0:
    GroupFairness = 2  #max value
  else:
     GroupFairness = GroupFairness_discr/GroupFairness_priv

  if TP_discr+FP_discr == 0:
    PredictiveParity_discr = 0
    PredictiveParity = 0  #min value
  else:
    PredictiveParity_discr = (TP_discr)/(TP_discr+FP_discr)
  if TP_priv+FP_priv == 0:
    PredictiveParity_priv = 0
    PredictiveParity = 2  #max value
  else:
    PredictiveParity_priv = (TP_priv)/(TP_priv+FP_priv)
  if PredictiveParity_discr != 0 and PredictiveParity_priv != 0:
    PredictiveParity = PredictiveParity_discr/PredictiveParity_priv
  elif PredictiveParity_priv == 0:
    PredictiveParity = 2  #max value
  else:
    PredictiveParity = 0  #min value

  if TN_discr+FP_discr == 0:
    PredictiveEquality_discr = 0
    PredictiveEquality = 0  #min value
  else:
    PredictiveEquality_discr = (FP_discr)/(TN_discr+FP_discr)
  if TN_priv+FP_priv == 0:
    PredictiveEquality_priv = 0
    PredictiveEquality = 2  #max value
  else:
    PredictiveEquality_priv = (FP_priv)/(TN_priv+FP_priv)
  if PredictiveEquality_discr != 0 and PredictiveEquality_priv != 0:
    PredictiveEquality = PredictiveEquality_discr/PredictiveEquality_priv
  elif PredictiveEquality_priv == 0:
    PredictiveEquality = 2  #max value
  else:
    PredictiveEquality = 0  #min value

  if FN_priv+TP_priv == 0:
    EqualOpportunity_priv = 0
    EqualOpportunity = 2  #max value
  else:
    EqualOpportunity_priv = (FN_priv)/(TP_priv+FN_priv)
  if FN_discr+TP_discr == 0:
    EqualOpportunity_discr = 0
    EqualOpportunity = 0  #min value
  else:
    EqualOpportunity_discr = (FN_discr)/(TP_discr+FN_discr)
  if EqualOpportunity_priv != 0 and EqualOpportunity_discr != 0:
    EqualOpportunity = EqualOpportunity_discr/EqualOpportunity_priv
  elif EqualOpportunity_priv == 0:
    EqualOpportunity = 2  #max value
  else:
    EqualOpportunity = 0  #min value

  if FN_discr+TP_discr == 0:
    EqualizedOdds1 = 0
    EqualizedOdds = 0 #min value
  elif FN_priv+TP_priv == 0:
    EqualizedOdds1 = 0
    EqualizedOdds = 2 #max value
  elif (TP_priv/(TP_priv+FN_priv)) == 0:
    EqualizedOdds1 = 2 #max value
  else:
    EqualizedOdds1 = ((TP_discr/(TP_discr+FN_discr)) / (TP_priv/(TP_priv+FN_priv))) # (1-equalOpportunity_discr)/(1-equalOpportunity_priv)
  if TN_priv+FP_priv == 0:
    EqualizedOdds2 = 0
    EqualizedOdds = 2 #max value
  elif TN_discr+FP_discr == 0:
    EqualizedOdds2 = 0
    EqualizedOdds = 0 #min value
  elif (FP_priv/(TN_priv+FP_priv)) == 0:
    EqualizedOdds2 = 2 #max value
  else:
    EqualizedOdds2 = ((FP_discr/(TN_discr+FP_discr)) / (FP_priv/(TN_priv+FP_priv))) # = PredictiveEquality
  if EqualizedOdds1 != 0 and EqualizedOdds2 != 0:
    EqualizedOdds = and_function(EqualizedOdds1, EqualizedOdds2, threshold)
  else:
    EqualizedOdds = 2 #max value

  if TP_discr+FP_discr == 0 or TN_discr+FP_discr == 0:
    ConditionalUseAccuracyEquality1 = 0
    ConditionalUseAccuracyEquality= 0 #min value
  elif (TP_priv/(TP_priv+FP_priv)) == 0:
    ConditionalUseAccuracyEquality1 = 2 #max value
  else:
    ConditionalUseAccuracyEquality1 = ((TP_discr/(TP_discr+FP_discr)) / (TP_priv/(TP_priv+FP_priv)))
  if TN_discr+FN_discr == 0 or TN_priv+FN_priv == 0:
    ConditionalUseAccuracyEquality2 = 0
    ConditionalUseAccuracyEquality = 2 #max value
  elif (TN_priv/(TN_priv+FN_priv)) == 0:
    ConditionalUseAccuracyEquality2 = 2 #max value
  else:
    ConditionalUseAccuracyEquality2 = ((TN_discr/(TN_discr+FN_discr)) / (TN_priv/(TN_priv+FN_priv)))
  if ConditionalUseAccuracyEquality1 != 0 and ConditionalUseAccuracyEquality2 != 0:
    ConditionalUseAccuracyEquality = and_function(ConditionalUseAccuracyEquality1, ConditionalUseAccuracyEquality2, threshold)
  else:
    ConditionalUseAccuracyEquality = 2 #max value

  OAE_discr = (TP_discr+TN_discr)/len_discr
  OAE_priv = (TP_priv+TN_priv)/len_priv
  if OAE_priv != 0 and OAE_discr != 0:
    OverallAccuracyEquality = OAE_discr/OAE_priv
  elif OAE_priv == 0:
    OverallAccuracyEquality = 2  #max value
  else:
    OverallAccuracyEquality = 0 #min value

  if FP_priv == 0:
    TreatmentEquality_priv = 0
    TreatmentEquality = 2  #max value
  else:
    TreatmentEquality_priv = (FN_priv/FP_priv)
  if FP_discr == 0:
    TreatmentEquality_discr = 0
    TreatmentEquality = 0 #min value
  elif (FN_discr/FP_discr) == 0:
    TreatmentEquality_discr = 0 #max value
    TreatmentEquality = 0 #min value
  else:
    TreatmentEquality_discr = (FN_discr/FP_discr)
  if TreatmentEquality_priv != 0 and TreatmentEquality_discr != 0:
    TreatmentEquality = TreatmentEquality_discr/TreatmentEquality_priv
  elif TreatmentEquality_priv == 0:
    TreatmentEquality = 2 #max value
  else:
    TreatmentEquality = 0 #min value

  if TN_priv+FN_priv == 0:
    FORParity_priv = 0
    FORParity = 2 #max value
  else:
    FORParity_priv = (FN_priv)/(TN_priv+FN_priv)
  if TN_discr+FN_discr == 0:
    FORParity_discr = 0
    FORParity = 0  #min value
  elif (FN_discr)/(TN_discr+FN_discr) == 0:
    FORParity_discr = 0
    FORParity = 0 #min value
  else:
    FORParity_discr = (FN_discr)/(TN_discr+FN_discr)
  if FORParity_priv != 0 and FORParity_discr != 0:
    FORParity = FORParity_discr/FORParity_priv
  elif FORParity_priv == 0:
    FORParity = 2 #max value
  else:
    FORParity = 0 #min value


  FN_P_discr = (FN_discr)/len_discr
  FN_P_priv = (FN_priv)/len_priv
  if FN_P_priv == 0:
    FN_metric = 2  #max value
  else:
    FN_metric = FN_P_discr/FN_P_priv


  FP_P_discr = (FP_discr)/len_discr
  FP_P_priv = (FP_priv)/len_priv
  if FP_P_priv == 0:
    FP_metric = 2  #max value
  else:
    FP_metric = FP_P_discr/FP_P_priv


  #RecallParity = (TP_discr/(TP_discr+FN_discr))/(TP_priv/(TP_priv+FN_priv))

  metrics = {}
  metrics['GroupFairness'] = [GroupFairness, GroupFairness_discr, GroupFairness_priv]
  metrics['PredictiveParity'] = [PredictiveParity, PredictiveParity_discr, PredictiveParity_priv]
  metrics['PredictiveEquality'] = [PredictiveEquality, PredictiveEquality_discr, PredictiveEquality_priv]
  metrics['EqualOpportunity'] = [EqualOpportunity, EqualOpportunity_discr, EqualOpportunity_priv]
  metrics['EqualizedOdds'] = [EqualizedOdds, EqualizedOdds1, EqualizedOdds2]
  metrics['ConditionalUseAccuracyEquality'] = [ConditionalUseAccuracyEquality, ConditionalUseAccuracyEquality1 , ConditionalUseAccuracyEquality2]
  metrics['OverallAccuracyEquality'] = [OverallAccuracyEquality, OAE_discr, OAE_priv]
  metrics['TreatmentEquality'] = [TreatmentEquality, TreatmentEquality_discr, TreatmentEquality_priv]
  metrics['FORParity'] = [FORParity, FORParity_discr, FORParity_priv]
  metrics['FN'] = [FN_metric, FN_P_discr, FN_P_priv]
  metrics['FP'] = [FP_metric, FP_P_discr, FP_P_priv]

  for k in metrics.keys():
    value = standardization(rescale(metrics[k][0]))
    discr = metrics[k][1]
    priv = metrics[k][2]
    metrics[k] = {'Value': value, 'Discr_group': discr, 'Priv_group': priv}

  return metrics


# Fairness metrics computed using subtraction operator
def fairness_metrics_subtraction(confusion_matrix, threshold = 0.15):

  TP_priv, TN_priv, FP_priv, FN_priv, len_priv = retrieve_values(confusion_matrix['privileged'])
  TP_discr, TN_discr, FP_discr, FN_discr, len_discr = retrieve_values(confusion_matrix['discriminated'])

  GroupFairness_discr = (TP_discr+FP_discr)/len_discr
  GroupFairness_priv = (TP_priv+FP_priv)/len_priv
  GroupFairness = GroupFairness_discr-GroupFairness_priv

  if (TP_discr+FP_discr) == 0:
    PredictiveParity_discr = 0
  else:
    PredictiveParity_discr = (TP_discr)/(TP_discr+FP_discr)
  if (TP_priv+FP_priv) == 0:
    PredictiveParity_priv = 0
  else:
    PredictiveParity_priv = (TP_priv)/(TP_priv+FP_priv)
  PredictiveParity = PredictiveParity_discr-PredictiveParity_priv

  if TN_discr+FP_discr == 0:
    PredictiveEquality_discr = 0
  else:
    PredictiveEquality_discr = (FP_discr)/(TN_discr+FP_discr)
  if TN_priv+FP_priv == 0:
    PredictiveEquality_priv = 0
  else:
    PredictiveEquality_priv = (FP_priv)/(TN_priv+FP_priv)
  PredictiveEquality = PredictiveEquality_priv-PredictiveEquality_discr

  if TP_discr+FN_discr == 0:
    EqualOpportunity_discr = 0
  else:
    EqualOpportunity_discr = (FN_discr)/(TP_discr+FN_discr)
  if TP_priv+FN_priv == 0:
    EqualOpportunity_priv = 0
  else:
    EqualOpportunity_priv = (FN_priv)/(TP_priv+FN_priv)
  EqualOpportunity = EqualOpportunity_discr-EqualOpportunity_priv

  if FN_discr+TP_discr == 0:
    EqualizedOdds1 = 0
  elif FN_priv+TP_priv == 0:
    EqualizedOdds1 = 0
  else:
    EqualizedOdds1 = (TP_priv/(TP_priv+FN_priv))-(TP_discr/(TP_discr+FN_discr)) # (1-equalOpportunity_discr)/(1-equalOpportunity_priv)
  if FP_priv+TN_priv == 0:
    EqualizedOdds2 = 0
  elif FP_discr+TN_discr == 0:
    EqualizedOdds2 = 0
  else:
    EqualizedOdds2 = (FP_priv/(TN_priv+FP_priv))-(FP_discr/(TN_discr+FP_discr)) # = PredictiveEquality
  EqualizedOdds = sub_and_function(EqualizedOdds1, EqualizedOdds2, threshold)

  if TP_discr+FP_discr == 0:
    ConditionalUseAccuracyEquality1 = 0
  elif TP_priv+FP_priv == 0:
    ConditionalUseAccuracyEquality1 = 0
  else:
    ConditionalUseAccuracyEquality1 = (TP_priv/(TP_priv+FP_priv)) - (TP_discr/(TP_discr+FP_discr))
  if TN_discr+FN_discr == 0:
    ConditionalUseAccuracyEquality2 = 0
  elif TN_priv+FN_priv == 0:
    ConditionalUseAccuracyEquality2 = 0
  else:
    ConditionalUseAccuracyEquality2 = (TN_priv/(TN_priv+FN_priv)) - (TN_discr/(TN_discr+FN_discr))
  ConditionalUseAccuracyEquality = sub_and_function(ConditionalUseAccuracyEquality1, ConditionalUseAccuracyEquality2, threshold)

  OAE_discr = (TP_discr+TN_discr)/len_discr
  OAE_priv = (TP_priv+TN_priv)/len_priv
  OverallAccuracyEquality = OAE_discr-OAE_priv

  if FP_discr == 0:
    TreatmentEquality_discr = 0
  else:
    TreatmentEquality_discr = (FN_discr/FP_discr)
  if FP_priv == 0:
    TreatmentEquality_priv = 0
  else:
    TreatmentEquality_priv = (FN_priv/FP_priv)
  TreatmentEquality = TreatmentEquality_discr-TreatmentEquality_priv

  if TN_discr+FN_discr == 0:
    FORParity_discr = 0
  else:
    FORParity_discr = (FN_discr)/(TN_discr+FN_discr)
  if TN_priv+FN_priv == 0:
    FORParity_priv = 0
  else:
    FORParity_priv = (FN_priv)/(TN_priv+FN_priv)
  FORParity = FORParity_discr-FORParity_priv


  FN_P_discr =  (FN_discr)/len_discr
  FN_P_priv =  (FN_priv)/len_priv

  FP_P_discr = (FP_discr)/len_discr
  FP_P_priv =  (FP_priv)/len_priv

  #RecallParity = (TP_discr/(TP_discr+FN_discr))/(TP_priv/(TP_priv+FN_priv))

  metrics = {}
  metrics['GroupFairness'] = [GroupFairness, GroupFairness_discr, GroupFairness_priv]
  metrics['PredictiveParity'] = [PredictiveParity, PredictiveParity_discr, PredictiveParity_priv]
  metrics['PredictiveEquality'] = [PredictiveEquality, PredictiveEquality_discr, PredictiveEquality_priv]
  metrics['EqualOpportunity'] = [EqualOpportunity, EqualOpportunity_discr, EqualOpportunity_priv]
  metrics['EqualizedOdds'] = [EqualizedOdds, EqualizedOdds1, EqualizedOdds2]
  metrics['ConditionalUseAccuracyEquality'] = [ConditionalUseAccuracyEquality, ConditionalUseAccuracyEquality1 , ConditionalUseAccuracyEquality2]
  metrics['OverallAccuracyEquality'] = [OverallAccuracyEquality, OAE_discr, OAE_priv]
  metrics['TreatmentEquality'] = [TreatmentEquality, TreatmentEquality_discr, TreatmentEquality_priv]
  metrics['FORParity'] = [FORParity, FORParity_discr, FORParity_priv]
  metrics['FN'] = [FN_P_discr-FN_P_priv, FN_P_discr, FN_P_priv]
  metrics['FP'] = [FP_P_discr-FP_P_priv, FP_P_discr, FP_P_priv]

  for k in metrics.keys():
    value = standardization(metrics[k][0])
    discr = metrics[k][1]
    priv = metrics[k][2]
    metrics[k] = {'Value': value, 'Discr_group': discr, 'Priv_group': priv}

  return metrics

def compute_fairness_metrics(predictions_and_tests, target_variable_labels, models, n_splits, mitigation):
  confusion_matrices = compute_confusion_matrices(predictions_and_tests, target_variable_labels, models, n_splits, mitigation)
  fairness_metrics = {}
  sub_fairness_metrics = {}
  div_fairness_metrics = {}
  sub_dict = {}
  div_dict = {}
  #mitigation technique allow multiple models
  if mitigation not in without_model_mitigations:
    for model_name in (models):
      sub_dict = {}
      div_dict = {}
      for i in range(0,n_splits):
        model_split_conf_matrix = fairness_metrics_division(confusion_matrices[model_name][i])
        sub_dict[i] = fairness_metrics_subtraction(confusion_matrices[model_name][i])
        div_dict[i] = fairness_metrics_division(confusion_matrices[model_name][i])

      div_fairness_metrics[model_name] = div_dict
      sub_fairness_metrics[model_name] = sub_dict
  else:
    sub_dict = {}
    div_dict = {}
    for i in range(0,n_splits):
        sub_dict[i] = fairness_metrics_subtraction(confusion_matrices[i])
        div_dict[i] = fairness_metrics_division(confusion_matrices[i])

    div_fairness_metrics = div_dict
    sub_fairness_metrics = sub_dict

  fairness_metrics['division'] = div_fairness_metrics
  fairness_metrics['subtraction'] = sub_fairness_metrics

  return fairness_metrics

def compute_mean_std_dev_fairness_metrics(fairness_metrics, models, n_splits, mitigation):
  family_metrics = {}
  for f in family:
    model_metrics = {}
    #mitigation technique allow multiple models
    if mitigation not in without_model_mitigations:
      for m in models:
        metric_dict = {}
        for fair_m in fairness_catalogue:
          vec_metrics = []
          for i in range(0,n_splits):
            vec_metrics.append(fairness_metrics[f][m][i][fair_m]['Value'])
          #print(vec_metrics)
          #print(np.mean(vec_metrics), np.std(vec_metrics))
          metric_dict[fair_m] = [np.mean(vec_metrics), np.std(vec_metrics)]
        print(m, metric_dict)
        model_metrics[m] = metric_dict
    #without multiple models
    else:
      metric_dict = {}
      for fair_m in fairness_catalogue:
        vec_metrics = []
        for i in range(0,n_splits):
          vec_metrics.append(fairness_metrics[f][i][fair_m]['Value'])
        metric_dict[fair_m] = [np.mean(vec_metrics), np.std(vec_metrics)]
      model_metrics = metric_dict

    family_metrics[f]=model_metrics

  return family_metrics

def compute_final_fairness_metrics(mitigation, predictions_and_tests, target_variable_labels, models, n_splits):
  if mitigation not in without_model_mitigations:
    fairness_metrics = compute_fairness_metrics(predictions_and_tests, target_variable_labels, models, n_splits, mitigation)
    final_metrics = compute_mean_std_dev_fairness_metrics(fairness_metrics, models, n_splits, mitigation)
  else:
    fairness_metrics = compute_fairness_metrics(predictions_and_tests, target_variable_labels, None, n_splits, mitigation)
    final_metrics = compute_mean_std_dev_fairness_metrics(fairness_metrics, None, n_splits, mitigation)

  return final_metrics

# Terminology:

# - d is the predicted value,
# - Y is the actual value in the dataset
# - G the protected attribute, priv= privileged group, discr=discriminated group
# - L is the legittimate attribute (only for Conditional Statistical Parity)

# Fairness Metrics List:

# 1. Group Fairness: (d=1|G=priv) = (d=1|G=discr)
# 2. Predictive Parity: (Y=1|d=1,G=priv) = (Y=1|d=1,G=discr)
# 3. Predictive Equality: (d=1|Y=0,G=priv) = (d=1|Y=0,G=discr)
# 4. Equal Opportunity:  (d=0|Y=1,G=priv) = (d=0|Y=1,G=discr)
# 5. Equalized Odds: (d=1|Y=i,G=priv) = (d=1|Y=i,G=discr), i ∈ 0,1
# 6. ConditionalUseAccuracyEquality: (Y=1|d=1, G=priv) = (Y=1|d=1,G=discr) and (Y=0|d=0,G=priv) = (Y=0|d=0,G=discr)
# 7. Overall Accuracy Equality: (d=Y, G=priv) = (d=Y, G=priv)
# 8. Treatment Equality: (Y=1, d=0, G=priv)/(Y=0, d=1, G=priv) = (Y=1, d=0, G=discr)/(Y=0, d=1, G=discr)
# 9. FOR Parity: (Y=1|d=0, G=priv) = (Y=1|d=0,G=discr)

# How to evaluate the results?

# Looking at the value for each corresponding metric:

# - If the value is between 0 and 1-t the discriminated group suffers from unfairness
# - If the value is greater than 1+t the privileged group suffers from unfairness
# - If the value is between 1-t and 1+t both privileged and discriminated group have a fair treatment

# t is a threshold that should be choose by the user according to the context and the goal of the task.