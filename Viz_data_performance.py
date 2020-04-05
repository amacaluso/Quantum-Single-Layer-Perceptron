import seaborn as sns, numpy as np, matplotlib.pyplot as plt, pandas as pd
from sklearn import datasets


X, y = datasets.make_blobs(n_samples=500, centers=[[0.2, 0.8],[0.7, 0.1]],
                           n_features=2, center_box=(0, 1),
                           cluster_std = 0.2, random_state = 5432)


df = pd.DataFrame(X, columns=['$x_1$','$x_2$'])
Y = np.where(y == 0, 'class 0', 'class 1')
df['kind'] = Y

def multivariateGrid(col_x, col_y, col_k, df, k_is_color=False, scatter_alpha=.5):
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            plt.scatter(*args, **kwargs)

        return scatter

    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df
    )
    color = None
    legends=[]
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color=name
        g.plot_joint(
            colored_scatter(df_group[col_x],df_group[col_y],color),
        )
        sns.distplot(
            df_group[col_x].values,
            ax=g.ax_marg_x,
            color=color,
        )
        sns.distplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,
            color=color,
            vertical=True
        )
    # Do also global Hist:
    sns.distplot(
        df[col_x].values,
        ax=g.ax_marg_x,
        color='grey'
    )
    sns.distplot(
        df[col_y].values.ravel(),
        ax=g.ax_marg_y,
        color='grey',
        vertical=True
    )
    plt.tight_layout()
    plt.xlabel(r'$x_1$', fontsize = 18)
    plt.ylabel(r'$x_2$', fontsize=18, rotation = 0)
    plt.legend(legends, fontsize=14)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('Data.png', dpi = 300, bbox_inches = "tight")
    plt.show()
    plt.close()


multivariateGrid('$x_1$', '$x_2$', 'kind', df=df)



# iterations = range(100)
# opt_data = pd.DataFrame([pd.Series(iterations), pd.Series(train_vector),
#                          pd.Series(val_vector), pd.Series(cost_vector)]).transpose()
# opt_data.columns = ['iteration', 'train', 'test', 'cost']
# opt_data.to_csv('opt_data.csv', index = False)

# libraries

opt_data = pd.read_csv('opt_data.csv')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


fs_labels = 17
fs_legend = 13

#plt.figure(figsize=(5,5))
# multiple line plot
plt.figure(figsize=(6,5))
plt.plot(opt_data.iteration, opt_data.train, marker='', color='mediumseagreen', linewidth=2, label="Training")
plt.plot(opt_data.iteration,  opt_data.test, marker='', color='mediumseagreen', linewidth=2, linestyle='dashed', label="Testing")
plt.grid(alpha=0.3)
plt.legend(fontsize=fs_legend)
plt.ylim(0.2,1.2)
plt.xticks(fontsize=fs_labels)
plt.yticks(fontsize=fs_labels)
#plt.title('Performance of Quantum SLP classifier')
plt.xlabel('Epochs', fontsize=fs_labels)
plt.ylabel('Accuracy', fontsize=fs_labels)
plt2=plt.twinx()
# plt.plot(iterations, cost_vector, marker='', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4, label="Cost Function")
plt.plot(opt_data.iteration, opt_data.cost, marker='', markerfacecolor='lightblue',
         markersize=12, color='sandybrown', linewidth=2, label="Cost Function")
plt2.set_ylabel(r"$SSE$",color="sandybrown", rotation =270, labelpad=15, fontsize = fs_labels )
plt2.tick_params(axis='y', labelcolor='sandybrown')
#plt2.xticks(fontsize=20)
plt2.set_yticklabels(np.round(np.arange(0.4,1.6,0.2),2),fontsize=fs_labels)
plt.ylim(0.45,1.6)
plt2.legend(loc = 'lower right', fontsize=fs_legend)
plt.tight_layout()
plt.savefig('Performance.png', dpi = 400)
plt.show()
plt.close()

#training metric
