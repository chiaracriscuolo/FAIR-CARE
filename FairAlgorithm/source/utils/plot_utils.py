#import libraries
import numpy as np
import os
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import euclidean
from source.utils.config import *

path_to_project = '/content/drive/MyDrive/FairAlgorithm'
metrics = fairness_catalogue
perf_metrics = performance_metrics

def map_mitigation(mitigation):
  mit_dict = {
      'original': 'OR',
      'aif360-ad': 'AD',
      'aif360-di': 'DI',
      'aif360-lfr' : 'LF',
      'aif360-op': 'OP',
      'fl-cr': 'CR',
      'fl-to': 'TO',
      'aif360-rw': 'RW',
      'aif360-pr': 'PR',
      'aif360-er': 'ER',
      'aif360-ce': 'CE',
      'aif360-eo': 'EO',
      'aif360-roc': 'RO',
      'orig-Logistic Regression': 'oLORE',
      'orig-Bagging': 'oBAGG',
      'orig-Random Forest': 'oRAFO',
      'orig-Extremely Randomized Trees': 'oEXRT',
      'orig-Decision Tree': 'oDETR',
      'orig-Ada Boost': 'oADBO'

  }
  return mit_dict[mitigation]

def map_metric(metric):
  met_dict = {
      'GroupFairness': 'GFA',
      'EqualizedOdds': 'EOD',
      'PredictiveParity': 'PPA',
      'PredictiveEquality': 'PEQ',
      'EqualOpportunity': 'EOP',
      'ConditionalUseAccuracyEquality': 'CUA',
      'OverallAccuracyEquality': 'OAE',
      'TreatmentEquality': 'TEQ',
      'FORParity': 'FOR',
      'FN': 'FNP',
      'FP': 'FPP'
  }
  return met_dict[metric]

def perf_grouped_bar(metrics_dict, mitigation_list, model, mitigation_category, dataset_name, sensible_attribute):
  bars_per_group = len(perf_metrics)
  num_groups = len(mitigation_list)

  data = []
  for mitigation in mitigation_list:
    l = []
    for metric in perf_metrics:
      l.append(metrics_dict[mitigation][metric][model][0])
    data.append(l)

  data = np.asarray(data)
  data = data.reshape((num_groups, bars_per_group))

  # Set up bar positions
  bar_width = 0.1
  bar_positions = np.arange(bars_per_group)

  plt.figure(figsize=(12, 6))

  # Plot the vertical bar plot
  for i in range(num_groups):
      plt.bar(bar_positions + i*bar_width,
              data[i],
              width=bar_width,
              label=mitigation_list[i])

  # Add labels and title
  plt.xlabel('Perfomance Metrics')
  plt.ylabel('Perfomance Values')
  plt.title(dataset_name+' '+sensible_attribute+' '+model+' '+mitigation_category)

  plt.xticks(bar_positions + bar_width, perf_metrics, rotation=45, ha='right')
  plt.legend()

  model_no_spaces = model.replace(" ", "")
  configuration = f"{dataset_name}-{sensible_attribute}/performance"
  file_name = f"{mitigation_category}-{model_no_spaces}.png"

  path = os.path.join(path_to_project,'plots', configuration, file_name )
  plt.savefig(path, bbox_inches="tight")

  # Show the plot
  plt.show()

def perf_grouped_bar_no_model(metrics_dict, mitigation_list, mitigation_category, dataset_name, sensible_attribute):
  bars_per_group = len(perf_metrics)
  num_groups = len(mitigation_list)

  data = []
  for mitigation in mitigation_list:
    l = []
    for metric in perf_metrics:
      l.append(metrics_dict[mitigation][metric][0])
    data.append(l)

  data = np.asarray(data)
  data = data.reshape((num_groups, bars_per_group))

  # Set up bar positions
  bar_width = 0.1
  bar_positions = np.arange(bars_per_group)

  plt.figure(figsize=(12, 6))

  # Plot the vertical bar plot
  for i in range(num_groups):
      plt.bar(bar_positions + i*bar_width,
              data[i],
              width=bar_width,
              label=mitigation_list[i])
  # Add red threshold lines
  threshold_high = 0.15
  threshold_low = -0.15
  plt.axhline(y=threshold_high, color='red', linestyle='--')
  plt.axhline(y=threshold_low, color='red', linestyle='--')

  # Add labels and title
  plt.xlabel('Performance Metrics')
  plt.ylabel('Performance Metric Values')
  plt.title(dataset_name+' '+mitigation_category)

  plt.xticks(bar_positions + bar_width, perf_metrics, rotation=45, ha='right')
  plt.legend()

  configuration = f"{dataset_name}-{sensible_attribute}/performance"
  file_name = f"{mitigation_category}.png"
  path = os.path.join(path_to_project,'plots',configuration, file_name )
  plt.savefig(path,  bbox_inches="tight")

  # Show the plot
  plt.show()

def grouped_bar(metrics_dict, mitigation_list, comparison, model, mitigation_category, dataset_name, sensible_attribute):
  bars_per_group = len(metrics)
  num_groups = len(mitigation_list)

  data = []
  for mitigation in mitigation_list:
    l = []
    for metric in metrics:
      l.append(metrics_dict[mitigation][comparison][model][metric][0])
    data.append(l)

  #print(data)
  data = np.asarray(data)
  data = data.reshape((num_groups, bars_per_group))

  # Set up bar positions
  bar_width = 0.1
  bar_positions = np.arange(bars_per_group)

  plt.figure(figsize=(12, 6))

  # Plot the vertical bar plot
  for i in range(num_groups):
      plt.bar(bar_positions + i*bar_width,
              data[i],
              width=bar_width,
              label=map_mitigation(mitigation_list[i]))

  # Add red threshold lines
  threshold_high = 0.15
  threshold_low = -0.15

  # Add labels and title with increased font size and bold text
  plt.xlabel('Fairness Metrics', fontsize=14)
  plt.ylabel('Fairness Metric Values', fontsize=14)
  plt.title(f"{dataset_name} {sensible_attribute} {model} {comparison} {mitigation_category}", fontsize=16, fontweight='bold')

  # Map metrics and set x-ticks
  mapped_metrics = list(map(map_metric, metrics))
  plt.xticks(bar_positions + bar_width, mapped_metrics, rotation=45, ha='right', fontsize=14)

  # Adjust legend font size and weight
  plt.legend(fontsize=12)
  plt.axhline(y=threshold_high, color='red', linestyle='--') #, label='Threshold (+0.15)')
  plt.axhline(y=threshold_low, color='red', linestyle='--') #, label='Threshold (-0.15)')

  #mapped_metrics = map(map_metric, metrics)
  #plt.xticks(bar_positions + bar_width, mapped_metrics, rotation=45, ha='right')
  #plt.legend()

  model_no_spaces = model.replace(" ", "")
  configuration = f"{dataset_name}-{sensible_attribute}"

  #png_file_name = f"{mitigation_category}-{model_no_spaces}-{comparison}.png"
  pdf_file_name = f"{mitigation_category}-{model_no_spaces}-{comparison}.pdf"
  #png_path = os.path.join(path_to_project, 'plots', configuration, png_file_name)
  pdf_path = os.path.join(path_to_project, 'plots', configuration, pdf_file_name)

  #plt.savefig(png_path, bbox_inches="tight")
  plt.savefig(pdf_path, bbox_inches="tight")

  # Show the plot
  plt.show()

def grouped_bar_no_model(metrics_dict, mitigation_list, comparison, mitigation_category, dataset_name, sensible_attribute ):
  bars_per_group = len(metrics)
  num_groups = len(mitigation_list)

  data = []
  for mitigation in mitigation_list:
    l = []
    for metric in metrics:
      l.append(metrics_dict[mitigation][comparison][metric][0])
    data.append(l)

  data = np.asarray(data)
  data = data.reshape((num_groups, bars_per_group))

  # Set up bar positions
  bar_width = 0.1
  bar_positions = np.arange(bars_per_group)

  plt.figure(figsize=(12, 6))

  # Plot the vertical bar plot
  for i in range(num_groups):
      plt.bar(bar_positions + i*bar_width,
              data[i],
              width=bar_width,
              label=map_mitigation(mitigation_list[i]))
  # Add red threshold lines
  plt.legend(fontsize=12, ncol=2)
  threshold_high = 0.15
  threshold_low = -0.15
  plt.axhline(y=threshold_high, color='red', linestyle='--')
  plt.axhline(y=threshold_low, color='red', linestyle='--')

  # Add labels and title
  plt.xlabel('Fairness Metrics', fontsize=14)
  plt.ylabel('Fairness Metric Values', fontsize=14)
  plt.title(dataset_name+' '+sensible_attribute+' '+comparison+' '+mitigation_category, fontsize=16, fontweight='bold')

  mapped_metrics = map(map_metric, metrics)
  plt.xticks(bar_positions + bar_width, mapped_metrics, rotation=45, ha='right',  fontsize=14)
  #plt.xticks(bar_positions + bar_width, metrics, rotation=45, ha='right')
  plt.legend(fontsize=12, loc='lower left', bbox_to_anchor=(0, 0))

  configuration = f"{dataset_name}-{sensible_attribute}"
  #file_name = f"{mitigation_category}-{comparison}.png"
  png_file_name = f"{mitigation_category}-{comparison}.png"
  pdf_file_name = f"{mitigation_category}-{comparison}.pdf"
  png_path = os.path.join(path_to_project, 'plots', configuration, png_file_name)
  pdf_path = os.path.join(path_to_project, 'plots', configuration, pdf_file_name)
  plt.savefig(png_path, bbox_inches="tight")
  plt.savefig(pdf_path, bbox_inches="tight")
  #path = os.path.join(path_to_project,'plots',configuration, file_name )
  #plt.savefig(path,  bbox_inches="tight")

  # Show the plot
  plt.show()

def grouped_bar_std_dev(metrics_dict, mitigation_list, comparison, model, mitigation_category, dataset_name, sensible_attribute):
  bars_per_group = len(metrics)
  num_groups = len(mitigation_list)

  data = []
  errors = []
  for mitigation in mitigation_list:
    l = []
    e = []
    for metric in metrics:
      l.append(metrics_dict[mitigation][comparison][model][metric][0])
      e.append(metrics_dict[mitigation][comparison][model][metric][1])  # Standard deviation
    data.append(l)
    errors.append(e)

  #print(data)
  data = np.asarray(data)
  errors = np.asarray(errors)
  data = data.reshape((num_groups, bars_per_group))
  errors = errors.reshape((num_groups, bars_per_group))

  # Set up bar positions
  bar_width = 0.1
  bar_positions = np.arange(bars_per_group)

  plt.figure(figsize=(12, 6))

  # Plot the vertical bar plot
  for i in range(num_groups):
      plt.bar(bar_positions + i*bar_width,
              data[i],
              yerr=errors[i],  # Adding standard deviation as error bars
              capsize=5,  # Error bar caps
              width=bar_width,
              label=map_mitigation(mitigation_list[i]))

  # Add red threshold lines
  threshold_high = 0.15
  threshold_low = -0.15

  # Add labels and title with increased font size and bold text
  plt.xlabel('Fairness Metrics', fontsize=14)
  plt.ylabel('Fairness Metric Values', fontsize=14)
  plt.title(f"{dataset_name} {sensible_attribute} {model} {comparison} {mitigation_category}", fontsize=16, fontweight='bold')

  # Map metrics and set x-ticks
  mapped_metrics = list(map(map_metric, metrics))
  plt.xticks(bar_positions + bar_width, mapped_metrics, rotation=45, ha='right', fontsize=14)

  # Adjust legend font size and weight
  plt.legend(fontsize=12)
  plt.axhline(y=threshold_high, color='red', linestyle='--') #, label='Threshold (+0.15)')
  plt.axhline(y=threshold_low, color='red', linestyle='--') #, label='Threshold (-0.15)')

  #mapped_metrics = map(map_metric, metrics)
  #plt.xticks(bar_positions + bar_width, mapped_metrics, rotation=45, ha='right')
  #plt.legend()

  model_no_spaces = model.replace(" ", "")
  configuration = f"{dataset_name}-{sensible_attribute}/std-dev"

  #png_file_name = f"{mitigation_category}-{model_no_spaces}-{comparison}-std-dev.png"
  pdf_file_name = f"{mitigation_category}-{model_no_spaces}-{comparison}-std-dev.pdf"
  #png_path = os.path.join(path_to_project, 'plots', configuration, png_file_name)
  pdf_path = os.path.join(path_to_project, 'plots', configuration, pdf_file_name)

  #plt.savefig(png_path, bbox_inches="tight")
  plt.savefig(pdf_path, bbox_inches="tight")

  # Show the plot
  plt.show()

def grouped_bar_no_model_std_dev(metrics_dict, mitigation_list, comparison, mitigation_category, dataset_name, sensible_attribute ):
  bars_per_group = len(metrics)
  num_groups = len(mitigation_list)

  data = []
  errors= []
  for mitigation in mitigation_list:
    l = []
    e = []
    for metric in metrics:
      l.append(metrics_dict[mitigation][comparison][metric][0])
      e.append(metrics_dict[mitigation][comparison][metric][1])
    data.append(l)
    errors.append(e)

  data = np.asarray(data)
  errors = np.asarray(errors)
  data = data.reshape((num_groups, bars_per_group))
  errors = errors.reshape((num_groups, bars_per_group))

  # Set up bar positions
  bar_width = 0.1
  bar_positions = np.arange(bars_per_group)

  plt.figure(figsize=(12, 6))

  # Plot the vertical bar plot
  for i in range(num_groups):
      plt.bar(bar_positions + i*bar_width,
              data[i],
              yerr=errors[i],  # Adding standard deviation as error bars
              capsize=5,  # Error bar caps
              width=bar_width,
              label=map_mitigation(mitigation_list[i]))
  # Add red threshold lines
  plt.legend(fontsize=12, ncol=2)
  threshold_high = 0.15
  threshold_low = -0.15
  plt.axhline(y=threshold_high, color='red', linestyle='--')
  plt.axhline(y=threshold_low, color='red', linestyle='--')

  # Add labels and title
  plt.xlabel('Fairness Metrics', fontsize=14)
  plt.ylabel('Fairness Metric Values', fontsize=14)
  plt.title(dataset_name+' '+sensible_attribute+' '+comparison+' '+mitigation_category, fontsize=16, fontweight='bold')

  #mapped_metrics = map(map_metric, metrics)
  #plt.xticks(bar_positions + bar_width, mapped_metrics, rotation=45, ha='right',  fontsize=14)
  #plt.legend(fontsize=12, loc='lower left', bbox_to_anchor=(0, 0))

  configuration = f"{dataset_name}-{sensible_attribute}/std-dev"
  #file_name = f"{mitigation_category}-{comparison}.png"
  #png_file_name = f"{mitigation_category}-{comparison}-std-dev.png"
  pdf_file_name = f"{mitigation_category}-{comparison}-std-dev.pdf"
  #png_path = os.path.join(path_to_project, 'plots', configuration, png_file_name)
  pdf_path = os.path.join(path_to_project, 'plots', configuration, pdf_file_name)
  #plt.savefig(png_path, bbox_inches="tight")
  plt.savefig(pdf_path, bbox_inches="tight")
  #path = os.path.join(path_to_project,'plots',configuration, file_name )
  #plt.savefig(path,  bbox_inches="tight")

  # Show the plot
  plt.show()

def color_cells_mean(value):
    numeric_value = float(value.split('+/-')[0])  # Extract numeric part
    threshold_high = 0.15
    threshold_low = -0.15
    if threshold_low <= numeric_value <= threshold_high:
    # Pastel green for values within the range
        return 'background-color: #b2d8b2; color: black;'
    else:
        # Pastel orange for values outside the range
        return 'background-color: #f9ccac; color: black;'

# Define a function for conditional formatting
def color_cells(value):
    # Extract the numeric part and the uncertainty
    main_value, uncertainty = map(float, value.split('+/-'))
    # Compute the complete value
    complete_value_max = main_value + uncertainty
    complete_value_min = main_value - uncertainty
    # Apply thresholds
    if (-0.15 <= complete_value_max <= 0.15) and (-0.15 <= complete_value_min <= 0.15):
      # Pastel green for values within the range
      return 'background-color: #b2d8b2; color: black;'
    else:
      # Pastel orange for values outside the range
      return 'background-color: #f9ccac; color: black;'

def data_framing(dictionary, dataset, sensible_attribute, comparison, model, mitigation_list):
  # Define the columns for your DataFrame
  columns = ['Mitigation'] + metrics

  # Create an empty DataFrame with the specified columns
  data = pd.DataFrame(columns=columns)

  for mitigation in mitigation_list:
    l = []
    l.append(mitigation)
    for metric in metrics:
      if model is not None:
        l.append(f"{dictionary[dataset][sensible_attribute][mitigation][comparison][model][metric][0]:.3f}+/-{dictionary[dataset][sensible_attribute][mitigation][comparison][model][metric][1]:.3f}")
      else:
        l.append(f"{dictionary[dataset][sensible_attribute][mitigation][comparison][metric][0]:.3f}+/-{dictionary[dataset][sensible_attribute][mitigation][comparison][metric][1]:.3f}")
    row_df = pd.DataFrame([l], columns=columns)
    data = pd.concat([data, row_df], ignore_index=True)

  data.set_index('Mitigation', inplace=True)
  if model is not None:
    title = f"{dataset} | {sensible_attribute} | {comparison} | Model: {model}"
  else:
    title = f"{dataset} | {sensible_attribute} | {comparison}"

  # Apply the conditional formatting
  styled_data = data.style.set_caption(title).applymap(color_cells, subset=metrics)
  return styled_data
  
def perf_data_framing(dictionary, dataset, sensible_attribute, model, mitigation_list):
  # Define the columns for your DataFrame
  columns = ['Mitigation'] + perf_metrics

  # Create an empty DataFrame with the specified columns
  data = pd.DataFrame(columns=columns)

  for mitigation in mitigation_list:
    l = []
    l.append(mitigation)
    for metric in perf_metrics:
      if model is not None:
        l.append(f"{dictionary[dataset][sensible_attribute][mitigation][metric][model][0]:.3f}+/-{dictionary[dataset][sensible_attribute][mitigation][metric][model][1]:.3f}")
      else:
        l.append(f"{dictionary[dataset][sensible_attribute][mitigation][metric][0]:.3f}+/-{dictionary[dataset][sensible_attribute][mitigation][metric][1]:.3f}")
    row_df = pd.DataFrame([l], columns=columns)
    data = pd.concat([data, row_df], ignore_index=True)

  data.set_index('Mitigation', inplace=True)
  return data

# Load the metrics results
def load_metrics(dataset_name, sensible_attribute, mitigation):
  load_path = path_to_project + '/measurements/metrics-{}-{}-{}.p'.format(dataset_name, sensible_attribute, mitigation)
  with open(load_path, 'rb') as fp:
      mitigation_metrics = pickle.load(fp)
  return mitigation_metrics

def load_performance_metrics(dataset_name, sensible_attribute, mitigation):
  load_path = path_to_project + '/measurements/performance_metrics-{}-{}-{}.p'.format(dataset_name, sensible_attribute, mitigation)
  with open(load_path, 'rb') as fp:
      performance_metrics = pickle.load(fp)
  return performance_metrics

def load_original_performance_metrics(dataset_name, sensible_attribute):
  load_path = path_to_project + '/measurements/performance_metrics-{}-{}-original.p'.format(dataset_name, sensible_attribute)
  with open(load_path, 'rb') as fp:
      performance_metrics = pickle.load(fp)
  return performance_metrics

def load_original_metrics(dataset, sensible_attribute):
  load_path = path_to_project + '/measurements/metrics-{}-{}-original.p'.format(dataset, sensible_attribute)
  with open(load_path, 'rb') as fp:
      original_metrics = pickle.load(fp)
  return original_metrics

def load_array_metrics(dataset_list, all_mitigations, preprocessing_mitigation_list, inprocessing_mitigation_list, postprocessing_mitigation_list):
  overall_metrics = {}
  preprocessing_metrics = {}
  inprocessing_metrics = {}
  postprocessing_metrics = {}
  performance_metrics= {}

  # Load metrics for all datasets and mitigations
  for dataset in dataset_list:
    overall_metrics[dataset] = {}
    preprocessing_metrics[dataset]= {}
    inprocessing_metrics[dataset] = {}
    postprocessing_metrics[dataset] = {}
    performance_metrics[dataset] = {}
     #print(datasets_config[dataset]['sensible_attributes'])
    for sensible_attribute in datasets_config[dataset]['sensible_attributes']:
      ds_metrics = {}
      pre_metrics = {}
      in_metrics = {}
      post_metrics = {}
      perf_metrics = {}
      print(dataset)
      for mitigation in all_mitigations:
        ds_metrics[mitigation] = load_metrics(dataset, sensible_attribute, mitigation)
        perf_metrics[mitigation] = load_performance_metrics(dataset, sensible_attribute, mitigation)
      #print('pre')
      for mitigation in preprocessing_mitigation_list:
        pre_metrics[mitigation] = load_metrics(dataset, sensible_attribute, mitigation)
      #print('in')
      for mitigation in inprocessing_mitigation_list:
        in_metrics[mitigation] = load_metrics(dataset, sensible_attribute, mitigation)
      #print('post')
      for mitigation in postprocessing_mitigation_list:
        post_metrics[mitigation] = load_metrics(dataset, sensible_attribute, mitigation)

      overall_metrics[dataset][sensible_attribute] = ds_metrics
      preprocessing_metrics[dataset][sensible_attribute] = pre_metrics
      inprocessing_metrics[dataset][sensible_attribute] = in_metrics
      postprocessing_metrics[dataset][sensible_attribute] = post_metrics
      performance_metrics[dataset][sensible_attribute] = perf_metrics
      print('metrics mitigation loaded')
      original_metrics = load_original_metrics(dataset, sensible_attribute)

      overall_metrics[dataset][sensible_attribute]['original'] = original_metrics
      preprocessing_metrics[dataset][sensible_attribute]['original'] = original_metrics
      postprocessing_metrics[dataset][sensible_attribute]['original'] = original_metrics
      performance_metrics[dataset][sensible_attribute]['original'] = load_original_performance_metrics(dataset, sensible_attribute)
      for m in models:
        #print(m)
        orig_metrics = {}
        orig_metrics['division'] = original_metrics['division'][m]
        orig_metrics['subtraction'] = original_metrics['subtraction'][m]
        inprocessing_metrics[dataset][sensible_attribute]['orig-'+str(m)] = orig_metrics
        #inprocessing_metrics[dataset][m]['division'] = original_metrics['division'][m]
        #inprocessing_metrics[dataset][m]['subtraction'] = original_metrics['subtraction'][m]
      print('original metrics  loaded')

  return overall_metrics, preprocessing_metrics, inprocessing_metrics, postprocessing_metrics, performance_metrics




