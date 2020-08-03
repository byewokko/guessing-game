import matplotlib.pyplot as plt
import numpy as np


def plot_colourline(x, y, c, ax, linewidth=2):
    # print(c)
    c = np.asarray(c)
    c = (c-np.nanmin(c))/((np.nanmax(c)-np.nanmin(c)) or 1)
    # print(c)
    c = plt.cm.get_cmap("bone")(c)
    for i in np.arange(len(x)-1):
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], c=c[i], linewidth=linewidth)
    return
