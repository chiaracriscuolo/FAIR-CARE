import math
import random
import numpy as np
from rich import print
from rich.table import Table
from rich.panel import Panel
from rich.align import Align


def remove_missing_values(df):
  '''
  Remove missing values from the given dataframe
  '''
  df2 = df.dropna()
  if(df.shape == df2.shape):
    panel = Panel("",title="Searching missing values", subtitle="[green]NO MISSING VALUES FOUND")
  else:
    df = df2
    panel = Panel("",title="Searching missing values", subtitle="[green] MISSING VALUES FOUND AND REMOVED")
    
  print(panel)
  return df


def find_outliers_IQR(df, attribute):
  '''
  Search outliers for the given attribute using the IQR method (Interquartile Range)
  returns the outliers
  '''
  attribute_df = df[attribute]

  # Calculate the IQR
  q1=attribute_df.quantile(0.25)
  q3=attribute_df.quantile(0.75)
  IQR=q3-q1

  # Find outliers
  outliers = attribute_df[((attribute_df<(q1-1.5*IQR)) | (attribute_df>(q3+1.5*IQR)))]
  row = [str(attribute), str(outliers.shape[0]), str(outliers.max()), str(outliers.min())]

  return outliers, row


def search_and_remove_outliers(df, attributes):
  '''
  Search and remove outliers for the given attributes using the IQR method (Interquartile Range)
  returns the dataframe without outliers
  '''
  old_shape = df.shape
  table = Table(show_header=True, header_style="bold cyan")
  table.add_column('Attribute', justify="center")
  table.add_column('# Outliers', justify="center")
  table.add_column('MAX Outlier', justify="center")
  table.add_column('MIN Outlier', justify="center")
  for attribute in attributes:
    # Search outliers for the given attribute
    outliers, row = find_outliers_IQR(df, attribute)
    table.add_row(*row)
    # Remove the outliers
    df = df.drop(outliers.index)

  if(old_shape == df.shape):
    panel = Panel("",title="Searching outliers for the following attributes: {}".format(attributes), subtitle="[green]NO OUTLIERS FOUND")
  else:
    panel = Panel(Align.center(table, vertical="middle"), title="Searching outliers for the following attributes: {}".format(attributes), subtitle="[green]OUTLIERS REMOVED")

  print(panel)
  return df


def age_category(row):
  if(row["Age"]>25):
      return 1
  else:
      return 0


def standardizeData(data,colums_to_exclude=[]):
    """
        data is a dataframe stored all the data read from a csv source file
        colums_to_exclude is a array like data structure stored the attributes which should be ignored in the normalization process.
        return the standardized data
    """
    df = data.loc[:, data.columns.difference(colums_to_exclude)]# remove no weight attributes
    df_stand = (df - df.mean())/np.std(df)
    data.loc[:, data.columns.difference(colums_to_exclude)] = df_stand
    
    return data


def normalizeDataset(data,colums_to_exclude=[]):
    """
        data is a dataframe stored all the data read from a csv source file
        colums_to_exclude is a array like data structure stored the attributes which should be ignored in the normalization process.
        return the normalized data
    """
    df = data.loc[:,data.columns.difference(colums_to_exclude)] # remove no weight attributes
    norm_df = (df - df.min()) / (df.max() - df.min())
    data.loc[:,data.columns.difference(colums_to_exclude)] = norm_df

    return data