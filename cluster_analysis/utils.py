import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter
import itertools

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


def plot_cluster_time_distribution(df, labels, bin_width=20):
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
    plot_df.loc[plot_df['date'] == 0, 'date'] = 600

    for i in np.unique(labels):
        df_label = plot_df[plot_df['label'] == i]

        min_year = df_label['date'].min()
        max_year = df_label['date'].max()
        bins = np.arange(min_year, max_year + bin_width, bin_width)

        plt.figure(figsize=(10, 6))
        sns.histplot(df_label, x='date', bins=bins, kde=False)
        plt.xticks(bins)
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.title(f'Time Distribution for Cluster {i} - {bin_width}-Year Bins')
        plt.xticks(rotation=90)
        plt.show()


def plot_cluster_by_metadata(df, labels, meta_lst, cluster_num=None, anglo=None):
    """
    Plot the mean of binary metadata attributes for a specific cluster.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing the dataset with multiple metadata attributes.
    labels (list or array-like): A list or array of cluster labels corresponding to each row in `df`.
    cluster_num (int): The cluster number for which the metadata distribution is to be plotted.
    meta_lst (list): A list of strings representing the column names of the metadata attributes in `df`.
    anglo
    """
    plot_df = df.copy()
    plot_df['label'] = labels
    if cluster_num is not None:
        plot_df = plot_df[plot_df['label'] == cluster_num]
    
    if anglo is True:
        plot_df = plot_df[plot_df['date'] < 1066]
    if anglo is False:
        plot_df = plot_df[plot_df['date'] >= 1066]

    mean_values = plot_df[meta_lst].mean()
    mean_values_df = mean_values.reset_index()
    mean_values_df.columns = ['metadata', 'mean value']

    title = 'Mean of Metadata'
    if cluster_num is not None:
        title += f' in Cluster {cluster_num}'
    if anglo is not None:
        title += ' in Anglo-Saxon Period' if anglo else ' in Norman Period'

    plt.figure(figsize=(10, 6))
    sns.barplot(x='metadata', y='mean value', data=mean_values_df)
    plt.title(title)
    plt.xlabel('Metadata')
    plt.ylabel('Mean Value')
    plt.xticks(rotation=45)
    
    plt.show()


def plot_cluster_word_distribution(df, labels, cluster_num=None, n=3, top_n=20, anglo=None):
    """
    Plot the distribution of the first 'n' words from each phrase in a specified cluster.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing the dataset with phrases in one of the columns.
    labels (list or array-like): A list or array of cluster labels corresponding to each row in `df`.
    cluster_num (int, optional): The specific cluster number to analyze. If None, all clusters are considered.
    n (int, default 3): The number of initial words to consider from each phrase.
    top_n (int, default 20): The number of top frequent words to display in the plot.
    anglo (bool, optional): A boolean to filter the phrases based on historical date. 
                              True to consider only phrases before 1066, False for phrases from 1066 and onwards.
                              If None, no date-based filtering is applied.
    """
    plot_df = df.copy()
    plot_df['label'] = labels

    if cluster_num is not None:
        plot_df = plot_df[plot_df['label'] == cluster_num]

    if anglo is True:
        plot_df = plot_df[plot_df['date'] < 1066]
    elif anglo is False:
        plot_df = plot_df[plot_df['date'] >= 1066]

    words = plot_df['phrase'].str.split().str[: n].str.join(' ').to_list()
    word_counts = Counter(words)
    top_n_words = word_counts.most_common(top_n)
    words_df = pd.DataFrame(top_n_words, columns=['Word', 'Frequency'])

    title = f'Top {top_n} Frequent Words Among First {n} Words'
    if cluster_num is not None:
        title += f' in Cluster {cluster_num}'
    if anglo is not None:
        title += ' in Anglo-Saxon Period' if anglo else ' in Norman Period'

    plt.figure(figsize=(10, 6))
    plt.barh(words_df['Word'], words_df['Frequency'])
    plt.title(title)
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.gca().invert_yaxis()
    plt.show()



