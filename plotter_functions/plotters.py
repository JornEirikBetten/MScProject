import matplotlib.pyplot as plt
import matplotlib



def plot_lineplot(x,
                  ylist,
                  labels,
                  xlabel,
                  ylabel,
                  colorlist,
                  fig_path,
                  fig_name,
                  alpha=0.8,
                  xticks=None,
                  xticklabels=None,
                  yticks=None,
                  yticklabels=None):
    fig = plt.figure()
    for i, y in enumerate(ylist):
        plt.plot(x, y, color=colorlist[i], label=labels[i], alpha=alpha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if xticks != None:
        plt.xticks(xticks, xticklabels)
    if yticks != None:
        plt.yticks(yticks, yticklabels)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(fig_path + fig_name, format="pdf", bbox_inches="tight")
    return fig
