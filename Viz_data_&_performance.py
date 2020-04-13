### DATA PLOT ------------------------------

import numpy as np
import pandas as pd
from sklearn import datasets
# user defined packages
import sys
sys.path.insert(0, "./code")
from Utils import multivariateGrid

# generate data
X, y = datasets.make_blobs(n_samples=500, centers=[[0.2, 0.8], [0.7, 0.1]],
                           n_features=2, center_box=(0, 1), cluster_std=0.2,
                           random_state=5432)

df = pd.DataFrame(X, columns=['$x_1$', '$x_2$'])
Y = np.where(y == 0, 'class 0', 'class 1')
df['kind'] = Y

# specify additional column for classes colors (optional)
group_color = np.where(y == 0, 'rebeccapurple', 'darkkhaki')
df['grp_col'] = group_color

# 2-D scatteplot (run without col_color for default colors)
multivariateGrid('$x_1$', '$x_2$', 'kind', df=df, col_color='grp_col')

### PERFORMANCE PLOT ------------------------------

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# read input data
opt_data = pd.read_csv('results/opt_data.csv')

# set save path for output figures (and create if necessary)
save_path = "./results/plots"
Path(save_path).mkdir(parents=True, exist_ok=True)

# define axes and legend fontsize
fs_labels = 17
fs_legend = 15.5

# define colors and linestyles for the curves
color_cost = "cadetblue"
linestyle_cost = "--"

color_train = "darkkhaki"
linestyle_train = "-."

color_test = "rebeccapurple"
linestyle_test = "-"

fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))

# plot test accuracy
ax1.plot(opt_data.iteration, opt_data.test, marker='', color=color_test,
         linewidth=2, linestyle=linestyle_test, label="Test accuracy",
         zorder=10)
# plot train accuracy
ax1.plot(opt_data.iteration, opt_data.train, marker='', color=color_train,
         linewidth=2, label="Train accuracy", linestyle=linestyle_train,
         zorder=5)

# set grid style
ax1.grid(alpha=0.3)
# set legend fontsize
ax1.legend(fontsize=fs_legend)

# set accuracy y-axis range
ax1.set_ylim(0.2, 1.2)
# set axes labels fontsize
[ticklab.set_fontsize(fs_labels) for ticklab in ax1.get_xticklabels()]
[ticklab.set_fontsize(fs_labels) for ticklab in ax1.get_yticklabels()]
# set axes titles
ax1.set_xlabel('Epochs', fontsize=fs_labels)
ax1.set_ylabel('Accuracy', fontsize=fs_labels)

# get additional y-axis for plotting the cost function
ax2 = ax1.twinx()
# set secondary y-axis below primary axis to avoid curve overlapping
ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)

# plot cost function
ax2.plot(opt_data.iteration, opt_data.cost, marker='',
         markerfacecolor='lightblue', markersize=12, color=color_cost,
         linewidth=2, label="Cost function", linestyle=linestyle_cost, zorder=0)
# set secondary y-axis style
ax2.set_ylabel(r"$SSE$", color=color_cost, rotation=90, labelpad=15,
               fontsize=fs_labels)
ax2.tick_params(axis='y', labelcolor=color_cost)
ax2.set_yticklabels(np.round(np.arange(0.4, 1.6, 0.2), 2), fontsize=fs_labels)

# set secondary y-axis range
ax2.set_ylim(0.45, 1.6)
# set legend options
ax2.legend(loc='lower left', fontsize=fs_legend)
plt.tight_layout()

fig.savefig(
    '{}/Performance_cost_col={}_cl_tr={}_col_tst={}.png'.format(save_path,
                                                                color_cost,
                                                                color_train,
                                                                color_test),
    dpi=300)
plt.show()
plt.close()

### OPTIMIZATION PLOT ------------------------------

# read data
df = pd.read_csv('results/data_multiple_runs.csv')

# preprocess
std_dev = np.arange(0.1, 0.8, 0.02)
train_mean = np.array(df.train_mean)
test_mean = np.array(df.test_mean)
cost_mean = np.array(df.cost_mean)
train_sd = np.array(df.train_sd)
test_sd = np.array(df.test_sd)
cost_sd = np.array(df.cost_sd)

df = pd.DataFrame(
    [train_mean, test_mean, cost_mean, train_sd, test_sd, cost_sd]).transpose()
df.columns = ['train_mean', 'test_mean', 'cost_mean', 'train_sd', 'test_sd',
              'cost_sd']
df.to_csv('results/data_multiple_runs_all.csv', index=False)

# set axes and legend fontsize
fs_labels = 11
fs_legend = 14

fig, ax1 = plt.subplots(1, 1, figsize=(8, 5.5))

# plot test accuracy with filled area for confidence bands
p2 = ax1.plot(std_dev, test_mean, linestyle=linestyle_test,
              label="Test accuracy", color=color_test, zorder=5)
ax1.fill_between(std_dev, test_mean - test_sd, test_mean + test_sd,
                 color=color_test, alpha=0.5)

# plot train accuracy with filled area for confidence bands
p1 = ax1.plot(std_dev, train_mean, linestyle=linestyle_train,
              label="Train accuracy", color=color_train)
ax1.fill_between(std_dev, train_mean - train_sd, train_mean + train_sd,
                 alpha=0.5, color=color_train, zorder=2)

# set accuracy plot style
ax1.grid(alpha=0.4)
ax1.set_xlabel(r'Standard deviation', fontsize=fs_labels)
ax1.set_ylabel(r'Accuracy', fontsize=fs_labels)
[ticklab.set_fontsize(fs_labels) for ticklab in ax1.get_xticklabels()]
[ticklab.set_fontsize(fs_labels) for ticklab in ax1.get_yticklabels()]

# get additional y-axis for plotting the cost function
ax2 = ax1.twinx()
# set secondary y-axis below primary axis to avoid curve overlapping
ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)

# plot cost function with filled area for confidence bands
p3 = ax2.plot(std_dev, cost_mean, linestyle=linestyle_cost,
              label="Cost function", color=color_cost)
ax2.fill_between(std_dev, cost_mean - cost_sd, cost_mean + cost_sd, alpha=0.5,
                 color=color_cost)

# set cost function style
ax2.set_ylabel(r"$SSE$", color=color_cost, fontsize=fs_labels)
ax2.tick_params(axis='y', labelcolor=color_cost)
ax2.set_yticklabels(np.round(np.arange(0.4, 2.3, 0.2), 2), fontsize=fs_labels)

# set ordered legend entries with formats
lns = p3 + p2 + p1
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, bbox_to_anchor=(.5, -.25), loc='lower center', ncol=3,
           fontsize=fs_legend)
# save output figure
fig.savefig(
    '{}/Optimization_cost_col={}_cl_tr={}_col_tst={}.png'.format(save_path,
                                                                 color_cost,
                                                                 color_train,
                                                                 color_test),
    dpi=300, bbox_inches = "tight")
plt.show()
plt.close()