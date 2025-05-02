import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from IPython.display import display

def map_acronym(acronym):
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
      'orig-Ada Boost': 'oADBO',
      'Logistic Regression': 'LORE',
      'Bagging': 'BAGG',
      'Random Forest': 'RAFO',
      'Extremely Randomized Trees': 'EXRT',
      'Decision Tree': 'DETR',
      'Ada Boost': 'ADBO',
  }
  return mit_dict[acronym]

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
      'FP': 'FPP',
      'accuracy': 'Acc',
      'precision': 'Pre',
      'recall': 'Rec',
      'f1_score': 'F1'
  }
  return met_dict[metric]

# Perfomance utils
def dict_to_dataframe(results_dict, dataset_name, mit_list):
    """
    Converts the dictionary into a DataFrame in long format for ANOVA analysis, 
    including only selected mitigations from mit_list.
    
    Parameters:
        results_dict (dict): The nested dictionary with results.
        dataset_name (str): The dataset name.
        mit_list (list): List of selected mitigation techniques to include.

    Returns:
        pd.DataFrame: A formatted DataFrame ready for analysis.
    """
    records = []
    
    for mitigation, metrics in results_dict.items():  # Loop over mitigation strategies
        if mitigation in mit_list:  # Only process if in the selected list
            for metric, model_results in metrics.items():  # Loop over performance metrics
                if isinstance(model_results, dict): 
                    for model in model_results.keys():  # Loop over models
                        values = model_results[model]
                        
                        # Ensure values is a list with at least two elements (mean, std dev)
                        if isinstance(values, list) and len(values) >= 2:
                            mean_score, std_dev = values[:2]  
                        else:
                            mean_score, std_dev = values, None  # Fallback for unexpected cases
                        # Append extracted information as a new record
                        records.append({
                            "Dataset": dataset_name,
                            "Mitigation": mitigation,
                            "Metric": metric,
                            "Model": model,
                            "Score": float(mean_score),  # Convert to float for numerical analysis
                            "StdDev": float(std_dev) if std_dev is not None else None
                        })
                else:
                  mean_score, std_dev = model_results[:2]
                  # Append extracted information as a new record
                  records.append({
                      "Dataset": dataset_name,
                      "Mitigation": mitigation,
                      "Metric": metric,
                      "Model": mitigation,
                      "Score": float(mean_score),  # Convert to float for numerical analysis
                      "StdDev": float(std_dev) if std_dev is not None else None
                })        

    # Convert list of records into a DataFrame
    return pd.DataFrame(records)

def create_df_list_full(values_dict, dataset_category, mit_list=None):
  df_list= []
  for d in dataset_category.keys():
    dataset_df = pd.DataFrame()
    for s in dataset_category[d]:
      dataset_df = dict_to_dataframe(values_dict[d][s], str(d+' '+s), mit_list)
      df_list.append(dataset_df)
  full_df = pd.concat(df_list, ignore_index=True)
  return full_df

#Fairness utils
def dict_to_dataframe_fairness(results_dict, dataset_name, modality, mit_list):
    """
    Converts the dictionary into a DataFrame in long format for ANOVA analysis, 
    including only selected mitigations from mit_list.
    
    Parameters:
        results_dict (dict): The nested dictionary with results.
        dataset_name (str): The dataset name.
        mit_list (list): List of selected mitigation techniques to include.

    Returns:
        pd.DataFrame: A formatted DataFrame ready for analysis.
    """
    records = []
    
    for mitigation, models in results_dict.items():  # Loop over mitigation strategies
        if mitigation in mit_list:  # Only process if in the selected list
            for model, metric_results in models[modality].items():  # Loop over performance metrics
                if isinstance(metric_results, dict): 
                    for metric in metric_results.keys():  # Loop over models
                        values = metric_results[metric]
                        
                        # Ensure values is a list with at least two elements (mean, std dev)
                        if isinstance(values, list) and len(values) >= 2:
                            mean_score, std_dev = values[:2]  
                        else:
                            mean_score, std_dev = values, None  # Fallback for unexpected cases
                        # Append extracted information as a new record
                        records.append({
                            "Dataset": dataset_name,
                            "Mitigation": mitigation,
                            "Metric": metric,
                            "Model": model,
                            "Score": float(mean_score),  # Convert to float for numerical analysis
                            "StdDev": float(std_dev) if std_dev is not None else None
                        })
                else:
                  mean_score, std_dev = metric_results[:2]
                  # Append extracted information as a new record
                  records.append({
                      "Dataset": dataset_name,
                      "Mitigation": mitigation,
                      "Metric": model,
                      "Model": mitigation,
                      "Score": float(mean_score),  # Convert to float for numerical analysis
                      "StdDev": float(std_dev) if std_dev is not None else None
                })        

    # Convert list of records into a DataFrame
    return pd.DataFrame(records)

def create_df_fairness_list_full(values_dict, dataset_category, modality, mit_list=None):
  df_list= []
  for d in dataset_category.keys():
    dataset_df = pd.DataFrame()
    for s in dataset_category[d]:
      dataset_df = dict_to_dataframe_fairness(values_dict[d][s], str(d+' '+s), modality, mit_list)
      df_list.append(dataset_df)
  full_df = pd.concat(df_list, ignore_index=True)
  return full_df

def run_anova(df):
    """
    Performs ANOVA for each metric to check for statistical differences.
    """
    anova_results = {}

    for metric in df["Metric"].unique():
        metric_df = df[df["Metric"] == metric]
        
        # Group by mitigation strategy and extract values
        groups = [metric_df[metric_df["Mitigation"] == mit]["Score"].astype(float) 
                  for mit in metric_df["Mitigation"].unique()]
        
        # Run ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        anova_results[metric] = {"F-statistic": f_stat, "p-value": p_value}
        
        print(f"ANOVA for {metric}: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")

    return anova_results

def run_tukey(df):
    """
    Runs Tukey's HSD test for each metric where ANOVA found significance.
    """
    tukey_results = {}

    for metric in df["Metric"].unique():
        metric_df = df[df["Metric"] == metric]

        # Run Tukey's HSD
        tukey = pairwise_tukeyhsd(endog=metric_df["Score"].astype(float),
                                  groups=metric_df["Mitigation"],
                                  alpha=0.05)
        
        print(f"\nTukey HSD results for {metric}:\n")
        print(tukey)
        
        tukey_results[metric] = tukey
    
    return tukey_results

def aggregate_results_ranking(df):
    """
    Aggregates results across datasets to find the best performing mitigation per metric.
    """
    summary = df.groupby(["Mitigation", "Metric"])["Score"].agg(["mean", "std"]).reset_index()
    
    # Rank mitigations per metric (higher is better)
    summary["Rank"] = summary.groupby("Metric")["mean"].rank(ascending=False)
    
    return summary.sort_values(["Metric", "Rank"])

def aggregate_results_percentage(df):
    """
    Aggregates results across datasets to find the best performing mitigation per metric,
    including the percentage of times each mitigation was the best.
    """
    # Compute mean and std of scores
    summary = df.groupby(["Mitigation", "Metric"])["Score"].agg(["mean", "std"]).reset_index()
    
    # Rank mitigations per metric (higher is better)
    summary["Rank"] = summary.groupby("Metric")["mean"].rank(ascending=False)
    
    # Count how many times each mitigation was the best per dataset
    best_counts = df.loc[df.groupby(["Dataset", "Metric"])["Score"].idxmax()]
    best_counts = best_counts.groupby(["Mitigation", "Metric"]).size().reset_index(name="Best_Count")

    # Compute total number of datasets
    total_datasets = df["Dataset"].nunique()
    
    # Merge best count data with summary
    summary = summary.merge(best_counts, on=["Mitigation", "Metric"], how="left").fillna(0)
    
    # Compute percentage of times a mitigation was best
    summary["Best_Percentage"] = (summary["Best_Count"] / total_datasets) * 100
    
    return summary.sort_values(["Metric", "Rank"])

def aggregate_results_with_significance(df):
    """
    Aggregates results across datasets and determines if a mitigation method 
    is statistically significant compared to others.
    
    Parameters:
        df (pd.DataFrame): The input dataframe with columns:
                           ['Dataset', 'Mitigation', 'Metric', 'Score']
                           
    Returns:
        pd.DataFrame: Aggregated results with mean, std, rank, best %, and significance.
    """
    summary = df.groupby(["Mitigation", "Metric"])["Score"].agg(["mean", "std"]).reset_index()
    summary["Rank"] = summary.groupby("Metric")["mean"].rank(ascending=False)

    # Compute best percentage
    best_counts = (
        df.loc[df.groupby(["Dataset", "Metric"])["Score"].idxmax()]
        .groupby(["Mitigation", "Metric"])
        .size()
        .reset_index(name="Best_Count")
    )
    total_datasets = df["Dataset"].nunique()
    best_counts["Best_Percentage"] = (best_counts["Best_Count"] / total_datasets) * 100
    summary = summary.merge(best_counts, on=["Mitigation", "Metric"], how="left").fillna(0)

    # ANOVA + Tukey's HSD for statistical significance
    significance_results = []

    for metric in df["Metric"].unique():
        metric_data = df[df["Metric"] == metric]
        
        # Perform one-way ANOVA
        groups = [metric_data[metric_data["Mitigation"] == m]["Score"].values for m in metric_data["Mitigation"].unique()]
        anova_p_value = stats.f_oneway(*groups).pvalue

        # If ANOVA shows a significant difference, perform Tukey's HSD
        if anova_p_value < 0.05:
            tukey = pairwise_tukeyhsd(metric_data["Score"], metric_data["Mitigation"], alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])  # Convert to DataFrame

            # Identify statistically significant mitigations
            significant_methods = set(tukey_df.loc[tukey_df["reject"] == True, "group1"]) | set(
                tukey_df.loc[tukey_df["reject"] == True, "group2"]
            )

        else:
            significant_methods = set()  # No significant differences
        
        for mitigation in metric_data["Mitigation"].unique():
            significance_results.append({
                "Mitigation": mitigation,
                "Metric": metric,
                "Stat_Significance": "Significant" if mitigation in significant_methods else "Not Significant"
            })

    # Merge with summary table
    significance_df = pd.DataFrame(significance_results)
    summary = summary.merge(significance_df, on=["Mitigation", "Metric"], how="left")
    return summary.sort_values(["Metric", "Rank"])


# def plot_box_plots(df):

#   plt.figure(figsize=(12, 6))
#   sns.boxplot(x="Metric", y="Score", hue="Mitigation", data=df)
#   plt.title("Performance of Mitigation Strategies Across Metrics")
#   plt.xticks(rotation=45)
#   plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#   plt.show()

def plot_box_plots(df):
    # Apply mappings to create new columns
    df = df.copy()  # To avoid modifying the original DataFrame
    df['MitigationMapped'] = df['Mitigation'].apply(map_acronym)
    df['MetricMapped'] = df['Metric'].apply(map_metric)

    # Plot using the mapped labels
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="MetricMapped", y="Score", hue="MitigationMapped", data=df)
    plt.title("Performance of Mitigation Strategies Across Metrics")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.show()



def statistical_tests_results(full_df, title):
  anova_results = run_anova(full_df)

  # Run Tukey’s HSD only if ANOVA found significance
  if any(res["p-value"] < 0.05 for res in anova_results.values()):
      tukey_results = run_tukey(full_df)

  agg_results = aggregate_results_ranking(full_df)
  print('Overall ranking')
  display(agg_results)
  print('----------')
  
  agg_results = aggregate_results_percentage(full_df)
  print('Adding percentage')
  display(agg_results)  
  print('----------')

  agg_results = aggregate_results_with_significance(full_df)
  print('Adding signinficance test result')
  display(agg_results)
  print('----------')

  #plot_box_plots(full_df)
  print('Color-blind version')
  plot_box_plots_color_blind(full_df, title)

# By ML model analysis
def aggregate_results_by_model(df):
    """
    Aggregates results across datasets for each ML model separately,
    ranking mitigation methods per model and metric.
    """
    # Group by Model, Mitigation, and Metric, and compute mean and std of scores
    summary = (
        df.groupby(["Model", "Mitigation", "Metric"])["Score"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    
    # Rank within each Model-Metric pair (higher is better)
    summary["Rank"] = summary.groupby(["Model", "Metric"])["mean"].rank(ascending=False)
    summary["IsBest"] = summary["Rank"] == 1
    return summary.sort_values(["Model", "Metric", "Rank"]).reset_index(drop=True)


def plot_mitigation_ranks(summary_df):
    """
    Plots a heatmap of mitigation rankings across models and metrics.
    Lower rank = better performance (darker color).
    """
    # Combine Model + Metric for columns
    heatmap_df = summary_df.copy()
    heatmap_df["Model+Metric"] = heatmap_df["Model"] + " | " + heatmap_df["Metric"]

    # Pivot the data for heatmap
    pivot_df = heatmap_df.pivot(index="Mitigation", columns="Model+Metric", values="Rank")

    # Dynamically adjust figure height based on number of mitigations
    num_mitigations = len(pivot_df.index)
    plt.figure(figsize=(max(12, len(pivot_df.columns) * 1.2), num_mitigations * 0.6))

    # Plot heatmap with reversed color scale (darker = better)
    sns.heatmap(
        pivot_df,
        annot=True,
        cmap="YlGnBu_r",  # <- _r reverses the color map
        cbar_kws={'label': 'Rank'},
        fmt=".0f"
    )

    plt.title("Mitigation Strategy Ranks per Model and Metric\n(Lower is Better = Darker)")
    plt.xlabel("Model + Metric")
    plt.ylabel("Mitigation")
    plt.tight_layout()
    plt.show()

def run_anova_by_model_metric(df):
    """
    Performs ANOVA and Tukey HSD for each (Model, Metric) group to test
    if differences between mitigation strategies are statistically significant.
    
    Returns:
        results_df (DataFrame): ANOVA p-values and significance per group.
        tukey_results (dict): Tukey HSD results per (Model, Metric).
    """
    anova_results = []
    tukey_results = {}

    grouped = df.groupby(["Model", "Metric"])

    for (model, metric), group in grouped:
        # Prepare data for ANOVA
        mitigation_groups = [sub["Score"].values for _, sub in group.groupby("Mitigation")]

        # Perform one-way ANOVA
        if len(mitigation_groups) >= 2:
            f_stat, p_value = stats.f_oneway(*mitigation_groups)
        else:
            f_stat, p_value = None, None

        anova_results.append({
            "Model": model,
            "Metric": metric,
            "F_statistic": f_stat,
            "p_value": p_value,
            "Significant": p_value is not None and p_value < 0.05
        })

        # Perform Tukey HSD if ANOVA was significant
        if p_value is not None and p_value < 0.05:
            tukey = pairwise_tukeyhsd(
                endog=group["Score"],
                groups=group["Mitigation"],
                alpha=0.05
            )
            tukey_results[(model, metric)] = tukey.summary()

    results_df = pd.DataFrame(anova_results)
    return results_df, tukey_results

def aggregate_fairness_results_by_model(df):
    """
    Aggregates fairness metric results per model, mitigation, and metric.
    Ranks mitigations based on proximity to 0 (ideal fairness).
    """
    summary = df.groupby(["Model", "Mitigation", "Metric"])["Score"].agg(["mean", "std"]).reset_index()

    # Compute absolute deviation from 0 (ideal fairness value)
    summary["abs_deviation"] = summary["mean"].abs()

    # Rank mitigations per metric and model (lower deviation = better fairness)
    summary["Rank"] = summary.groupby(["Model", "Metric"])["abs_deviation"].rank(ascending=True)

    return summary.sort_values(["Model", "Metric", "Rank"])

def aggregate_fairness_results_ranking(df):
    """
    Aggregates results across datasets to find the best performing mitigation per metric.
    """
    summary = df.groupby(["Mitigation", "Metric"])["Score"].agg(["mean", "std"]).reset_index()
    
    # Compute absolute deviation from 0 (ideal fairness value)
    summary["abs_deviation"] = summary["mean"].abs()

    # Rank mitigations per metric and model (lower deviation from 0 = better fairness)
    summary["Rank"] = summary.groupby("Metric")["abs_deviation"].rank(ascending=True)
    
    return summary.sort_values(["Metric", "Rank"])

def aggregate_fairness_results_percentage(df):
    """
    Aggregates results across datasets to find the best performing mitigation per metric,
    including the percentage of times each mitigation was the best.
    """
    # Compute mean and std of scores
    summary = df.groupby(["Mitigation", "Metric"])["Score"].agg(["mean", "std"]).reset_index()
    
    # Compute absolute deviation from 0 (ideal fairness value)
    summary["abs_deviation"] = summary["mean"].abs()

    # Rank mitigations per metric and model (lower deviation from 0 = better fairness)
    summary["Rank"] = summary.groupby("Metric")["abs_deviation"].rank(ascending=True)
    
    # Count how many times each mitigation was the best per dataset
    best_counts = df.loc[df.groupby(["Dataset", "Metric"])["Score"].idxmax()]
    best_counts = best_counts.groupby(["Mitigation", "Metric"]).size().reset_index(name="Best_Count")

    # Compute total number of datasets
    total_datasets = df["Dataset"].nunique()
    
    # Merge best count data with summary
    summary = summary.merge(best_counts, on=["Mitigation", "Metric"], how="left").fillna(0)
    
    # Compute percentage of times a mitigation was best
    summary["Best_Percentage"] = (summary["Best_Count"] / total_datasets) * 100
    
    return summary.sort_values(["Metric", "Rank"])

def aggregate_fairness_results_with_significance(df):
    """
    Aggregates results across datasets and determines if a mitigation method 
    is statistically significant compared to others.
    
    Parameters:
        df (pd.DataFrame): The input dataframe with columns:
                           ['Dataset', 'Mitigation', 'Metric', 'Score']
                           
    Returns:
        pd.DataFrame: Aggregated results with mean, std, rank, best %, and significance.
    """
    summary = df.groupby(["Mitigation", "Metric"])["Score"].agg(["mean", "std"]).reset_index()
    # Compute absolute deviation from 0 (ideal fairness value)
    summary["abs_deviation"] = summary["mean"].abs()

    # Rank mitigations per metric and model (lower deviation from 0 = better fairness)
    summary["Rank"] = summary.groupby("Metric")["abs_deviation"].rank(ascending=True)

    # Compute best percentage
    best_counts = (
        df.loc[df.groupby(["Dataset", "Metric"])["Score"].idxmax()]
        .groupby(["Mitigation", "Metric"])
        .size()
        .reset_index(name="Best_Count")
    )
    total_datasets = df["Dataset"].nunique()
    best_counts["Best_Percentage"] = (best_counts["Best_Count"] / total_datasets) * 100
    summary = summary.merge(best_counts, on=["Mitigation", "Metric"], how="left").fillna(0)

    # ANOVA + Tukey's HSD for statistical significance
    significance_results = []

    for metric in df["Metric"].unique():
        metric_data = df[df["Metric"] == metric]
        
        # Perform one-way ANOVA
        groups = [metric_data[metric_data["Mitigation"] == m]["Score"].values for m in metric_data["Mitigation"].unique()]
        anova_p_value = stats.f_oneway(*groups).pvalue

        # If ANOVA shows a significant difference, perform Tukey's HSD
        if anova_p_value < 0.05:
            tukey = pairwise_tukeyhsd(metric_data["Score"], metric_data["Mitigation"], alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])  # Convert to DataFrame

            # Identify statistically significant mitigations
            significant_methods = set(tukey_df.loc[tukey_df["reject"] == True, "group1"]) | set(
                tukey_df.loc[tukey_df["reject"] == True, "group2"]
            )

        else:
            significant_methods = set()  # No significant differences
        
        for mitigation in metric_data["Mitigation"].unique():
            significance_results.append({
                "Mitigation": mitigation,
                "Metric": metric,
                "Stat_Significance": "Significant" if mitigation in significant_methods else "Not Significant"
            })

    # Merge with summary table
    significance_df = pd.DataFrame(significance_results)
    summary = summary.merge(significance_df, on=["Mitigation", "Metric"], how="left")
    return summary.sort_values(["Metric", "Rank"])

def statistical_tests_fairness_results(full_df, title):
  anova_results = run_anova(full_df)

  # Run Tukey’s HSD only if ANOVA found significance
  if any(res["p-value"] < 0.05 for res in anova_results.values()):
      tukey_results = run_tukey(full_df)

  agg_results = aggregate_fairness_results_ranking(full_df)
  print('Overall ranking')
  display(agg_results)
  print('----------')
  
  agg_results = aggregate_fairness_results_percentage(full_df)
  print('Adding percentage')
  display(agg_results)  
  print('----------')

  agg_results = aggregate_fairness_results_with_significance(full_df)
  print('Adding signinficance test result')
  display(agg_results)
  print('----------')

  #plot_box_plots(full_df)
  print('Color-blind version')
  plot_box_plots_color_blind(full_df, title)

def join_fairness_performance(summary_perf, summary_fair):
    """
    Joins performance and fairness summaries on Mitigation, Model, and Metric.
    Adds suffixes to distinguish columns.
    """
    merged = pd.merge(
        summary_perf,
        summary_fair,
        on=["Mitigation", "Model", "Metric"],
        how="outer",  # 'inner' or "outer" if you want to keep everything
        suffixes=("_perf", "_fair")
    )
    return merged

def scatter_fair_vs_perf(merged_df, perf_metric, fair_metric):
    # Filter for the performance and fairness metric separately
    df = merged_df[
        (merged_df["Metric"] == perf_metric) |
        (merged_df["Metric"] == fair_metric)
    ]

    # Pivot so each metric becomes a column
    df_pivot = df.pivot_table(
        index=["Mitigation", "Model"],
        columns="Metric",
        values=["Score_perf", "Score_fair"],
        aggfunc="first"  # or 'mean' if needed
    ).reset_index()

    # Flatten column names
    df_pivot.columns = ['_'.join(col).strip('_') for col in df_pivot.columns.values]

    # Drop rows with missing values in both
    df_pivot = df_pivot.dropna(subset=[f"Score_perf_{perf_metric}", f"Score_fair_{fair_metric}"])

    if df_pivot.empty:
        print("No data available after filtering and pivoting.")
        return

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_pivot,
        x=f"Score_fair_{fair_metric}",
        y=f"Score_perf_{perf_metric}",
        hue="Mitigation",
        style="Model",
        s=100
    )
    plt.axvline(0, linestyle="--", color="grey", alpha=0.6)
    plt.title(f"Fairness vs Performance\nFair: {fair_metric} | Performance: {perf_metric}")
    plt.xlabel("Fairness (closer to 0 is better)")
    plt.ylabel("Performance (higher is better)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def scatter_fair_vs_perf_color_blind(merged_df, perf_metric, fair_metric, name, modality):
    # Filter for the selected performance and fairness metrics
    df = merged_df[
        (merged_df["Metric"] == perf_metric) |
        (merged_df["Metric"] == fair_metric)
    ]
    
    # Pivot so each metric becomes a column
    df_pivot = df.pivot_table(
        index=["Mitigation", "Model"],
        columns="Metric",
        values=["Score_perf", "Score_fair"],
        aggfunc="first"  # or 'mean' if values are repeated
    ).reset_index()

    # Flatten column names
    df_pivot.columns = ['_'.join(col).strip('_') for col in df_pivot.columns.values]

    # Drop rows missing either fairness or performance score
    perf_col = f"Score_perf_{perf_metric}"
    fair_col = f"Score_fair_{fair_metric}"
    #print("Columns in df_pivot:", df_pivot.columns.tolist())
    #print("Expected columns:", perf_col, fair_col)

    df_pivot = df_pivot.dropna(subset=[perf_col, fair_col])

    if df_pivot.empty:
        print("No data available after filtering and pivoting.")
        return

    # Set color-blind friendly palette
    sns.set_palette("colorblind")

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_pivot,
        x=fair_col,
        y=perf_col,
        hue="Mitigation",
        style="Model",
        s=100,
        edgecolor='black'
    )

    # Update legend labels with mapped mitigation names
    handles, labels = plt.gca().get_legend_handles_labels()

    # Separate handles for hue (Mitigation) and style (Model)
    # Usually, the first N handles are hue, and the rest are style (if present)
    unique_mitigations = df_pivot["Mitigation"].unique()
    unique_models= df_pivot["Model"].unique()
    mapped_labels = [
    map_acronym(label) if label in unique_mitigations
    else map_acronym(label) if label in unique_models
    else label
    for label in labels
]

    # Update legend
    plt.legend(handles=handles, labels=mapped_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)

    plt.axvline(0, linestyle="--", color="grey", alpha=0.6)
    plt.title(f"Fairness vs Performance\nFairness: {map_metric(fair_metric)} | Performance: {map_metric(perf_metric)}")
    plt.xlabel("Fairness (closer to 0 is better)", fontsize=11)
    plt.ylabel("Performance (closer to 1 is better)", fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    saved_file_name = f"{fair_metric}-{perf_metric}-{modality}.pdf"

    path = os.path.join(path_to_project, 'findings/tradeoff', name, saved_file_name)
    plt.savefig(path, bbox_inches="tight")
    plt.show()


# def plot_box_plots_color_blind(df):
#     """
#     Plots box plots for mitigation strategies across metrics using a color-blind friendly palette.
#     """
#     plt.figure(figsize=(12, 6))

#     # Set the color-blind friendly palette
#     sns.set_palette("colorblind")

#     sns.boxplot(x="Metric", y="Score", hue="Mitigation", data=df)
#     plt.title("Performance of Mitigation Strategies Across Metrics")
#     plt.xticks(rotation=45)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.show()
path_to_project = '/content/drive/MyDrive/FairAlgorithm'

def plot_box_plots_color_blind(df, file_name):
    """
    Plots box plots for mitigation strategies across metrics using a color-blind friendly palette.
    """
    # Create a copy and apply the mappings
    df = df.copy()
    df['MitigationMapped'] = df['Mitigation'].apply(map_acronym)
    df['MetricMapped'] = df['Metric'].apply(map_metric)

    plt.figure(figsize=(12, 6))

    # Set the color-blind friendly palette
    sns.set_palette("colorblind")

    # Plot with mapped labels
    sns.boxplot(x="MetricMapped", y="Score", hue="MitigationMapped", data=df)
    plt.title("Mitigation Strategies Across Metrics", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.xlabel("Metric", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.tight_layout()

    saved_file_name = f"{file_name}.pdf"

    path = os.path.join(path_to_project, 'findings', saved_file_name)
    plt.savefig(path, bbox_inches="tight")
    plt.show()
