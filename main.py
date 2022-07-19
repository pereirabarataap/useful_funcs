import numpy as np
from matplotlib import pyplot as plt

def get_elbow_value(array, n_steps=1e6, plot=True):
    """
    Function that returns the optimal cutoff value (elbow in cum_prop)
    array: array or list
        - the array for which the cutoff is to be found
    n_steps: int or float
        - granularity of the search; larger values produce finer results
    plot: bool
        - to either plot or not to plot the results
    """
    n_steps= int(n_steps)
    value_steps = np.linspace(np.min(array), np.max(array), n_steps)
    cum_sum = np.concatenate(
        ([0], np.cumsum(np.histogram(array, value_steps)[0]))
    )
    cum_prop = cum_sum / cum_sum[-1]
    increase_th = np.linspace(0,1,n_steps)[1]
    th_locs = np.diff(cum_prop)>=increase_th
    # if there is not just 1 "elbow"
    if np.diff(th_locs).sum()!= 1:
        return get_cutoff_value(array, n_steps=n_steps*(1-1e-2), plot=plot)
    else:    
        cutoff_index = np.argmin(th_locs)
        cutoff_value = value_steps[cutoff_index]
    
    if plot:
        fig, ax = plt.subplots(1,1,dpi=100, figsize=(6,6))
        x=value_steps
        y=cum_prop
        ax.plot(x,y)
        cutoff_x = cutoff_value
        cutoff_y = cum_prop[cutoff_index]
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.vlines(x=cutoff_x, ymin=ymin, ymax=ymax, alpha=0.5, ls="dotted")
        ax.hlines(y=cutoff_y, xmin=xmin, xmax=xmax, alpha=0.5, ls="dotted")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.ylabel("Cumulative proportion")
        plt.xlabel("Value")
        plt.title("Cutoff point")
        plt.grid(alpha=0.2)
        plt.show()
        
    return cutoff_value
