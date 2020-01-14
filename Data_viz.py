import seaborn as sns, numpy as np, matplotlib.pyplot as plt, pandas as pd

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
    plt.legend(legends)
    plt.grid(alpha=0.3)
    plt.savefig('Data.png')


multivariateGrid('$x_1$', '$x_2$', 'kind', df=df)
plt.close()


# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iterations = range(100)
#plt.figure(figsize=(5,5))
# multiple line plot
#plt.figure(figsize=(5,5))
plt.tight_layout()
plt.plot(iterations, train_vector, marker='', color='olive', linewidth=2, label="Training")
plt.plot(iterations, val_vector, marker='', color='olive', linewidth=2, linestyle='dashed', label="Testing")
plt.grid(alpha=0.3)
plt.legend()
plt.ylim(0.2,1.2)
#plt.title('Performance of Quantum SLP classifier')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt2=plt.twinx()
# plt.plot(iterations, cost_vector, marker='', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4, label="Cost Function")
plt.plot(iterations, cost_vector, marker='', markerfacecolor='lightblue', markersize=12, color='lightblue', linewidth=3, label="Cost Function")
plt2.set_ylabel(r"$SSE$",color="lightblue")
plt2.tick_params(axis='y', labelcolor='skyblue')
plt.ylim(0.45,1.6)
plt2.legend(loc = 'lower right')
plt.savefig('Performance.png')
plt.show()
plt.close()

#training metric
