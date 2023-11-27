import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def plot(labels, time, order, vertical_bars=None, x_limits=None):
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
    sns.scatterplot(x=time, y=order, hue=labels, palette='viridis')

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
