import numpy as np
import pandas as pd
import math
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from scipy.stats import norm


def fair_metrics(dataset, pred):
  dataset_pred = dataset.copy()
  dataset_pred.labels = pred
  
  cols = ['statistical_parity_difference', 'equal_opportunity_difference', 'average_abs_odds_difference',  'disparate_impact', 'theil_index']
  obj_fairness = [[0,0,0,1,0]]
  
  fair_metrics = pd.DataFrame(data=obj_fairness, index=['objective'], columns=cols)
  
  for attr in dataset_pred.protected_attribute_names:
      idx = dataset_pred.protected_attribute_names.index(attr)
      privileged_groups =  [{attr:dataset_pred.privileged_protected_attributes[idx][0]}] 
      unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}] 
      
      classified_metric = ClassificationMetric(dataset, 
                                                    dataset_pred,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups)

      metric_pred = BinaryLabelDatasetMetric(dataset_pred,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups)

      row = pd.DataFrame([[ 
                            metric_pred.mean_difference(),
                            classified_metric.equal_opportunity_difference(),
                            classified_metric.average_abs_odds_difference(),
                            metric_pred.disparate_impact(),
                            classified_metric.theil_index()]],
                          columns  = cols,
                          index = [attr]
                        )
      fair_metrics = pd.concat([fair_metrics, row])
  
  fair_metrics = fair_metrics.replace([-np.inf, np.inf], 2)
      
  return fair_metrics
  