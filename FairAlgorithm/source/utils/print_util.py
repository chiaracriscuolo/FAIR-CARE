import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
import matplotlib.patches as patches
import numpy as np


def visualize_att_dist(_data, _att, _category=False):
  plt.figure(figsize=[12,5], dpi=100)
  sns.set(style="darkgrid")  
  sns.set(font_scale = 1.5)
  if _category: # for categorical attribute
      vis_data = _data[_att].value_counts()
      ax = sns.barplot(x=vis_data.index, y=vis_data.values);
      ax.set_xlabel(_att)
      if len(vis_data) >= 5:
          ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
  else:
      ax = sns.distplot(_data[_att], kde=False, color='steelblue');
  ax.set_ylabel("Count")
  plt.tight_layout()


def plot_fair_metrics(fair_metrics):
  fig, ax = plt.subplots(figsize=(20,4), ncols=5, nrows=1)

  plt.subplots_adjust(
      left    =  0.125, 
      bottom  =  0.1, 
      right   =  0.9, 
      top     =  0.9, 
      wspace  =  .5, 
      hspace  =  1.1
  )

  plt.suptitle("Fairness metrics", y = 1.2, fontsize=20)
  sns.set(style="dark")

  cols = fair_metrics.columns.values
  obj = fair_metrics.loc['objective']
  size_rect = [0.2,0.2,0.2,0.4,0.25]
  rect = [-0.1,-0.1,-0.1,0.8,0]
  bottom = [-1,-1,-1,0,0]
  top = [1,1,1,2,1]
  bound = [[-0.1,0.1],[-0.1,0.1],[-0.1,0.1],[0.8,1.2],[0,0.25]]
  for attr in fair_metrics.index[1:len(fair_metrics)].values:
      display(Markdown("### For the %s attribute :"%attr))
      check = [bound[i][0] < fair_metrics.loc[attr][i] < bound[i][1] for i in range(0,5)]
      display(Markdown("#### With default thresholds, bias against unprivileged group detected in **%d** out of 5 metrics"%(5 - sum(check))))

  for i in range(0,5):
      plt.subplot(1, 5, i+1)
      ax = sns.barplot(x=fair_metrics.index[1:len(fair_metrics)], y=fair_metrics.iloc[1:len(fair_metrics)][cols[i]])
      
      for j in range(0,len(fair_metrics)-1):
          a, val = ax.patches[j], fair_metrics.iloc[j+1][cols[i]]
          marg = -0.2 if val < 0 else 0.1
          ax.text(a.get_x()+a.get_width()/5, a.get_y()+a.get_height()+marg, round(val, 3), fontsize=15,color='black')

      plt.ylim(bottom[i], top[i])
      plt.setp(ax.patches, linewidth=0)
      ax.add_patch(patches.Rectangle((-5,rect[i]), 10, size_rect[i], alpha=0.3, facecolor="green", linewidth=1, linestyle='solid'))
      plt.axhline(obj[i], color='black', alpha=0.3)
      plt.title(cols[i])
      ax.set_ylabel('')    
      ax.set_xlabel('')


def print_heatmap(df):
   # Create a mask to display only the lower triangle of the matrix 
    mask = np.zeros_like(df)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(15,10))
    sns.heatmap(df, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5, annot=True)
    plt.yticks(rotation=0) 
    plt.xticks(rotation=90) 
    plt.show()


def visualize_diversity(_data, _att):
    fig, axs = plt.subplots(1, 2, figsize=(10,10))

    axs[1].pie(_data.head(10)[_att].value_counts(), autopct='%1.1f%%', labels=_data.head(10)[_att].value_counts().index);
    axs[1].set_title("First 10 values")

    axs[0].pie(_data[_att].value_counts(), autopct='%1.1f%%', labels=_data[_att].value_counts().index);
    axs[0].set_title("All data")

    plt.show()

    
    