def process_all_batches(batches):
    """
    Process all batches of items in parallel, updating a progress bar.

    Parameters:
    batches (list of lists): A list where each element is a batch (list) of items to be processed.

    Returns:
    list of lists: A list where each element corresponds to the processed results of the respective batch in the input.
    """

    from joblib import Parallel, delayed
    from tqdm.notebook import tqdm
    import time
    import threading
    import multiprocessing

    # Define a sample function that simulates some processing
    # Users should replace this function with their own processing logic
    def process_item(item, progress_list, index):
        """
        Simulate processing of an item.

        Parameters:
        item: The item to be processed.
        progress_list (list): A shared list to track progress.
        index (int): The index of the current batch.

        Returns:
        The processed item (for demonstration, item * 2).
        """
        time.sleep(1)  # Simulate a delay
        progress_list[index] += 1  # Update progress
        return item * 2

    # Define the function that processes a batch
    def process_batch(batch, progress_list, index):
        """
        Process a batch of items.

        Parameters:
        batch (list): The batch of items to be processed.
        progress_list (list): A shared list to track progress.
        index (int): The index of the current batch.

        Returns:
        list: The processed items in the batch.
        """
        results = [process_item(item, progress_list, index) for item in batch]
        return results

    # Define the function to update the progress bar
    def update_progress_bar(pbar, total_items, progress_list):
        """
        Update the progress bar based on the processing progress.

        Parameters:
        pbar (tqdm): The progress bar object.
        total_items (int): The total number of items to be processed.
        progress_list (list): A shared list to track progress.
        """
        while pbar.n < total_items:
            time.sleep(0.1)  # Refresh interval
            completed = sum(progress_list)  # Total completed items
            pbar.n = completed  # Update progress bar
            pbar.refresh()
        pbar.close()

    total_items = sum(len(batch) for batch in batches)  # Total number of items to process
    manager = multiprocessing.Manager()
    progress_list = manager.list([0] * len(batches))  # Shared list to track progress for each batch

    # Create the progress bar
    pbar = tqdm(total=total_items, position=0, leave=True)

    # Start the progress bar update thread
    progress_thread = threading.Thread(target=update_progress_bar, args=(pbar, total_items, progress_list))
    progress_thread.start()

    # Process each batch in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_batch)(batch, progress_list, i) for i, batch in enumerate(batches)
    )

    # Wait for the progress thread to finish
    progress_thread.join()

    return results
    

def make_distribution(
    size:int=100, 
    distribution_type:str="normal",  
    distribution_uniform_factor:float=0.5
):
    """
    Function which generates a distribution from which agents sample their random attributes
    
    Input:
        size: int -> [1, inf)
            number of variables/features to consider
        distribution_type: str -> {"normal", "powerlaw"}
            distribution shape to follow
        distribution_uniform_factor: float -> [0,1]
            how uniform the shape of the distribution should be, with 1 being completely uniform
    
    Output:
        numpy array of length 'size' of non-zero positive values which sum to 1
    """
    import numpy as np
    from scipy.stats import powerlaw, norm
    from sklearn.preprocessing import MinMaxScaler
    
    if distribution_type=="powerlaw":
        scaler = MinMaxScaler(feature_range=(0.1,1))
        power_uniform_factor = scaler.fit_transform(
            np.array([0, distribution_uniform_factor, 1]).reshape(-1,1)
        ).ravel()[1]
        materials_distribution = powerlaw.pdf(
            x=np.linspace(0,1, num=size+1)[1:],
            a=power_uniform_factor,
        )

    elif distribution_type=="normal":
        scaler = MinMaxScaler(feature_range=(-3,0))
        norm_uniform_factor = scaler.fit_transform(
            np.array([0, distribution_uniform_factor, 1]).reshape(-1,1)
        ).ravel()[1]
        materials_distribution = np.sort(norm.pdf(
            x=np.linspace(-norm_uniform_factor,norm_uniform_factor, num=size)
        ))[::-1]

    return materials_distribution / materials_distribution.sum()

class MissIndicator():

    import copy
    import numpy as np
    from sklearn.impute import MissingIndicator
    
    def __init__(self):
        self.is_fit = False
        
    def fit(self, X, y=None):
        self.mi = MissingIndicator(sparse=False, error_on_new=False)
        self.mi.fit(X)
        self.is_fit = True
        
    def transform(self, X, y=None):
        return np.concatenate([copy.deepcopy(X), self.mi.transform(X)], axis=1)
    
    def fit_transform(self, X, y=None):
        self.mi = MissingIndicator(sparse=False, error_on_new=False)
        self.mi.fit(X)
        self.is_fit = True
        return np.concatenate([copy.deepcopy(X), self.mi.transform(X)], axis=1)
    
class Clamper():

    import copy
    import numpy as np
    
    def __init__(self):
        self.is_fit = False
        self.values_to_keep = {}
        
    def _get_values_to_keep_from_value_counts(self, value_counts):
        values = value_counts.keys()
        counts = value_counts.values.astype(int)
        count_p = counts / sum(counts)
        min_p_increase = 1/len(values)
        index_to_keep = np.argmin(abs(count_p - min_p_increase))
        values_to_keep = values[:index_to_keep]
        return values_to_keep
    
    def fit_transform(self, X, y=None):
        transformed_X = copy.deepcopy(X)
        for column in X.columns:
            self.values_to_keep[column] = self._get_values_to_keep_from_value_counts(
                X[column].value_counts()
            )
            transformed_X.loc[
                ~(transformed_X[column].isin(self.values_to_keep[column])),
                column
            ] = "other"
        self.is_fit = True
        return transformed_X
    
    def fit(self, X, y=None):
        for column in X.columns:
            self.values_to_keep[column] = self._get_values_to_keep_from_value_counts(
                X[column].value_counts()
            )
        self.is_fit = True
        
    def transform(self, X, y=None):
        transformed_X = copy.deepcopy(X)
        for column in X.columns:
            transformed_X.loc[
                ~(transformed_X[column].isin(self.values_to_keep[column])),
                column
            ] = "other"
        
        return transformed_X

def get_values_to_keep_from_value_counts(value_counts, plot=False):
    values = value_counts.keys()
    counts = value_counts.values.astype(int)
    count_p = counts / sum(counts)
    min_p_increase = 1/len(values)
    index_to_keep = np.argmin(abs(count_p - min_p_increase))
    values_to_keep = values[:index_to_keep]
    
    if plot:
        fig, ax = plt.subplots(1,1, dpi=500, figsize=(len(values)//10,len(values)//10))    
        ax.plot(
            [""] + values.tolist(), 
            np.cumsum([0] + counts.tolist())
        )
        ax.scatter(
            [index_to_keep],
            [np.cumsum(counts.tolist())[index_to_keep-1]],
            c="C1",
            edgecolor="k"
        )
        ax.grid(alpha=0.5)
        ax.set_xlabel("Values")
        ytick_min = 0
        ytick_max = sum(counts)
        ax.set_ylabel("Proportion of samples retained")
        plt.title(f"{value_counts.index.name} values to keep based on proportion of samples retained")
        ax.set_yticks(
            ticks=np.round(np.linspace(ytick_min, ytick_max, len(values)+1)), 
            labels=np.round(np.linspace(0, 1, len(values)+1), 2).astype(float), 
            size=5
        )
        ax.set_xticks(
            ticks=range(len(values)+1),
            labels=[""] + values.tolist(),
            ha="center",
            fontsize=5,
            rotation=90,
        )
        ax.tick_params(axis='x', which='major', pad=1)
        plt.show()
    
    return values_to_keep

def plot_corr_df(corr_df, file_name=None, show=True):
    """
    corr_df must be a pandas.DataFrame().corr() object
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    from scipy.cluster.hierarchy import linkage

    labels = corr_df.columns

    # Initialize figure by creating upper dendrogram
    fig = ff.create_dendrogram(
        corr_df.values, 
        labels=labels,
        orientation='bottom', 
        linkagefun=lambda x: linkage(
            x, method="ward",optimal_ordering=True
        )
    )
    for i in range(len(fig['data'])):
        fig['data'][i]['yaxis'] = 'y2'

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(
        corr_df.values, 
        labels=labels,
        orientation='right',
        linkagefun=lambda x: linkage(
            x, method="ward",optimal_ordering=True
        )
    )
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'

    # Add Side Dendrogram Data to Figure
    for data in dendro_side['data']:
        fig.add_trace(data)

    # Create Heatmap
    dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
    # dendro_leaves = list(map(int, dendro_leaves))

    heatmap = [
        go.Heatmap(
            x = dendro_leaves,
            y = dendro_leaves,
            z = corr_df.loc[dendro_leaves, dendro_leaves],
            colorscale = 'tempo'
        )
    ]

    heatmap[0]['x'] = fig['layout']['xaxis']['tickvals']
    heatmap[0]['y'] = fig['layout']['xaxis']['tickvals'] #dendro_side['layout']['yaxis']['tickvals']

    # Add Heatmap Data to Figure
    for data in heatmap:
        fig.add_trace(data)

    # Edit Layout
    fig.update_layout(
        {
            "autosize": True,
            'width': 1000, 
            'height': 1000,
            'showlegend':False, 
            'hovermode': 'closest',
        }
    )

    # Edit xaxis
    fig.update_layout(xaxis={
        'domain': [.15, 1],
        'mirror': False,
        'showgrid': False,
        'showline': False,
        'zeroline': False,
        'ticktext': dendro_leaves,
        'ticks':"",
    })

    # Edit yaxis
    fig.update_layout(yaxis={
        'domain': [0, .85],
        'mirror': False,
        'showgrid': False,
        'showline': False,
        'zeroline': False,
        'showticklabels': False,
        'ticks': "",
        'ticktext': dendro_leaves,
        'tickvals': np.array(range(len(labels))) * 10 + 5
    })

    # Edit xaxis2
    fig.update_layout(xaxis2={
        'domain': [0, .15],
        'mirror': False,
        'showgrid': False,
        'showline': False,
        'zeroline': False,
        'showticklabels': False,
        'ticks':""
    })


    # Edit yaxis2
    fig.update_layout(yaxis2={
        'domain':[.825, 0.975],
        'mirror': False,
        'showgrid': False,
        'showline': False,
        'zeroline': False,
        'showticklabels': False,
        'ticks':""
    })
    
    if file_name:
        fig.write_html(file_name)
    
    # Plot!
    if show:
        fig.show()


def get_elbow_value(array, n_steps=1e6, plot=True):
    import numpy as np
    from matplotlib import pyplot as plt
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
        return get_elbow_value(array, n_steps=n_steps*(1-1e-2), plot=plot)
    else:    
        cutoff_index = np.argmin(th_locs)
        elbow_value = value_steps[cutoff_index]
    
    if plot:
        fig, ax = plt.subplots(1,1,dpi=100, figsize=(6,6))
        x=value_steps
        y=cum_prop
        ax.plot(x,y)
        cutoff_x = elbow_value
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
        
    return elbow_value
