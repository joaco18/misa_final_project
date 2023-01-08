import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
import pandas as pd
from typing import List
from pathlib import Path
from matplotlib.ticker import MultipleLocator


def plot_dice(df: pd.DataFrame, partition: str, experiment: str):
    """
    Boxplot of dice scores across cases ablated by experiment name
    Args:
        df (pd.DataFrame): Dataframe should contain at least the following columns:
            ["tissue", "dice", "experiment_name"]
        partition (str): partition name to be placed in the title
    """
    _, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_title(
        f'Dice score across {partition} cases grouped by tissue '
        f'and experiment - {experiment}',
        fontsize=14
    )
    sns.boxplot(data=df, x="tissue", y="dice", hue="experiment_name", palette='Paired', ax=ax)
    sns.despine()
    plt.ylim([0, 1.1])
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    plt.grid(axis='y', which='both')
    plt.xlabel('Tissues', fontsize=12)
    plt.ylabel('Dice Score', fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Experiment/Method', fontsize=10)
    # plt.show(block=False)


def plot_hausorf(df: pd.DataFrame, partition: str, experiment: str):
    """
    Boxplot of hausdorf distances across cases ablated by experiment name
    Args:
        df (pd.DataFrame): Dataframe should contain at least the following columns:
            ["tissue", "hausdorff", "experiment_name"]
        partition (str): partition name to be placed in the title
    """
    _, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_title(
        f'Hausdorff distances across {partition} cases grouped by tissue '
        f'and experiment - {experiment}',
        fontsize=14
    )
    sns.boxplot(data=df, x="tissue", y="hausdorff", hue="experiment_name", palette='Paired', ax=ax)
    sns.despine()
    plt.xlabel('Tissues', fontsize=12)
    plt.ylabel('Hausdorff Distance', fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Models', fontsize=10)
    # plt.show(block=False)


def plot_ravd(df: pd.DataFrame, partition: str, experiment: str):
    """
    Boxplot of the relative absolute volume difference across cases ablated by experiment name
    Args:
        df (pd.DataFrame): Dataframe should contain at least the following columns:
            ["tissue", "ravd", "experiment_name"]
        partition (str): partition name to be placed in the title
    """
    _, ax = plt.subplots(1, 1, figsize=(10, 4))
    t = f'Relative absolute volume difference across {partition}'\
        f'cases grouped by tissue and experiment - {experiment}'
    ax.set_title(t, fontsize=14)
    sns.boxplot(data=df, x="tissue", y="ravd", hue="experiment_name", palette='Paired', ax=ax)
    sns.despine()
    plt.xlabel('Tissues', fontsize=12)
    plt.ylabel('Relative absolute volume difference', fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Models', fontsize=10)
    # plt.show(block=False)


def plot_avd(df: pd.DataFrame, partition: str, experiment: str):
    """
    Boxplot of the absolute volume difference across cases ablated by experiment name
    Args:
        df (pd.DataFrame): Dataframe should contain at least the following columns:
            ["tissue", "avd", "experiment_name"]
        partition (str): partition name to be placed in the title
    """
    _, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_title(
        f'Asolute volume difference across {partition} case grouped by '
        f'tissue and experiment - {experiment}',
        fontsize=14
    )
    sns.boxplot(data=df, x="tissue", y="avd", hue="experiment_name", palette='Paired', ax=ax)
    sns.despine()
    plt.xlabel('Tissues', fontsize=12)
    plt.ylabel('Absolute volume difference', fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Models', fontsize=10)
    # plt.show(block=False)

