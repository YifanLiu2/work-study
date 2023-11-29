import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def plot_cluster_data(labels, time, order, vertical_bars=None, x_limits=None):
    """
    Plots a scatter plot with time as the x-axis, merge order as the y-axis, 
    and color-coded by labels. Optionally adds multiple vertical bars with specified colors.

    Parameters:
    labels (list or array): The labels for each data point.
    time (list or array): The time order for each data point.
    order (list or array): The merge order for each data point.
    vertical_bars (list of tuples, optional): A list of tuples, where each tuple contains 
                                             an x-value and a color for the vertical bar.
    x_limits (tuple, optional): A tuple containing the min and max values for the x-axis range.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=time, y=order, hue=labels, palette='viridis', s=10)

    plt.title('Scatter Plot of Merge Order over Time')
    plt.xlabel('Time Order')
    plt.ylabel('Merge Order')

    if vertical_bars:
        for x_value, color in vertical_bars:
            plt.axvline(x=x_value, color=color)

    if x_limits:
        plt.xlim(x_limits)

    plt.legend(title='Labels')
    plt.show()


def plot_bar_with_confidence(df, labels, meta):
    """
    Plots a bar plot for a given metadata column grouped by labels with 95% confidence intervals.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    labels (array-like): The cluster labels.
    meta (str): The name of the metadata column to analyze.
    """
    # Create a copy of the DataFrame with only necessary columns
    plot_df = df[[meta]].copy()
    plot_df['labels'] = labels

    # Plot using seaborn's barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='labels', y=meta, data=plot_df, errorbar=('ci', 95))

    plt.title(f'Mean of {meta} by Cluster Labels with 95% CI')
    plt.xlabel('Cluster Labels')
    plt.ylabel(f'Mean of {meta}')
    plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cluster_time_distribution(df, labels, bin_width=10):
    """
    Plot the time distribution for each cluster in a given DataFrame, with histogram bars covering multiple years.

    Parameters:
    df (DataFrame): The DataFrame containing the data to be plotted. 
                    It must include a 'date' column and a column with cluster labels.
    labels (array-like): A list of unique identifiers for each cluster present in the DataFrame.
    bin_width (int): The width of each bin in the histogram, in years. Defaults to 5 years.
    """
    plot_df = df.copy()
    plot_df['label'] = labels

    plot_df['year'] = pd.to_datetime(plot_df['date']).dt.year
    for i in np.unique(labels):
        df_label = plot_df[plot_df['label'] == i]

        min_year = df_label['year'].min()
        max_year = df_label['year'].max()
        bins = np.arange(min_year, max_year + bin_width, bin_width)

        plt.figure(figsize=(10, 6))
        sns.histplot(df_label, x='year', bins=bins, kde=False)
        plt.xticks(bins)
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.title(f'Time Distribution for Cluster {i} - {bin_width}-Year Bins')
        plt.show()

