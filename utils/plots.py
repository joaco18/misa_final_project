import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
import pandas as pd
from typing import List
from pathlib import Path
from matplotlib.ticker import MultipleLocator


def plot_dice(df: pd.DataFrame, partition: str):
    """
    Boxplot of dice scores across cases ablated by experiment name
    Args:
        df (pd.DataFrame): Dataframe should contain at least the following columns:
            ["tissue", "dice", "experiment_name"]
        partition (str): partition name to be placed in the title
    """
    _, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_title(
        f'Dice score across {partition} patients grouped by tissue and experiment',
        fontsize=14
    )
    sns.boxplot(data=df, x="tissue", y="dice", hue="experiment_name", palette='Paired', ax=ax)
    sns.despine()
    plt.ylim([0, 1.1])
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    plt.grid(axis='y', which='both')
    plt.xlabel('Tissues', fontsize=12)
    plt.ylabel('Dice Score', fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Models', fontsize=10)
    plt.show(block=False)


def plot_hausorf(df: pd.DataFrame, partition: str):
    """
    Boxplot of hausdorf distances across cases ablated by experiment name
    Args:
        df (pd.DataFrame): Dataframe should contain at least the following columns:
            ["tissue", "hausdorff", "experiment_name"]
        partition (str): partition name to be placed in the title
    """
    _, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_title(
        f'Hausdorff distances across {partition} patients grouped by tissue and experiment',
        fontsize=14
    )
    sns.boxplot(data=df, x="tissue", y="hausdorff", hue="experiment_name", palette='Paired', ax=ax)
    sns.despine()
    plt.xlabel('Tissues', fontsize=12)
    plt.ylabel('Hausdorff Distance', fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Models', fontsize=10)
    plt.show(block=False)


def plot_ravd(df: pd.DataFrame, partition: str):
    """
    Boxplot of the relative absolute volume difference across cases ablated by experiment name
    Args:
        df (pd.DataFrame): Dataframe should contain at least the following columns:
            ["tissue", "ravd", "experiment_name"]
        partition (str): partition name to be placed in the title
    """
    _, ax = plt.subplots(1, 1, figsize=(10, 4))
    t = f'Relative absolute volume difference across {partition}'\
        'patients grouped by tissue and experiment'
    ax.set_title(t, fontsize=14)
    sns.boxplot(data=df, x="tissue", y="ravd", hue="experiment_name", palette='Paired', ax=ax)
    sns.despine()
    plt.xlabel('Tissues', fontsize=12)
    plt.ylabel('Relative absolute volume difference', fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Models', fontsize=10)
    plt.show(block=False)


def plot_avd(df: pd.DataFrame, partition: str):
    """
    Boxplot of the absolute volume difference across cases ablated by experiment name
    Args:
        df (pd.DataFrame): Dataframe should contain at least the following columns:
            ["tissue", "avd", "experiment_name"]
        partition (str): partition name to be placed in the title
    """
    _, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_title(
        f'Asolute volume difference across {partition} patients grouped by tissue and experiment',
        fontsize=14
    )
    sns.boxplot(data=df, x="tissue", y="avd", hue="experiment_name", palette='Paired', ax=ax)
    sns.despine()
    plt.xlabel('Tissues', fontsize=12)
    plt.ylabel('Absolute volume difference', fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Models', fontsize=10)
    plt.show(block=False)


def brains_figure(
    cases: List[str], data_path: Path, segs_path: Path, exp_names: List[str],
    out_path: Path = None, slice_n: int = 125
):
    """
    Plot all segementations in plots of 7xn_cases. First row is t1 and second
    ground truth
    Args:
        img_path (Path): t1 path
        segs_path (Path): directory containing all segementations
        cases (List[str]): list of case names to use in the plot
        exp_names (List[str]): list of exp_names to include
        slice_n (int, optional): Axial slice to plot. Defaults to 25.
    """
    n_figures = int(np.ceil(len(exp_names) / 5))
    for n in range(n_figures):
        sublist_exp = exp_names[n*5:(n+1)*5]
        sublist_seg_p = segs_path[n*5:(n+1)*5]
        n_rows, n_cols = 2+len(sublist_exp), len(cases)
        _, ax = plt.subplots(n_rows, n_cols, figsize=(13, 3*(len(sublist_exp)+2)))
        for i, name in enumerate(['T1', 'GT']):
            for j, case in enumerate(cases):
                cmap = 'gray' if (i == 0) else 'viridis'
                img_name = f'{case}.nii.gz' if i == 0 else f'{case}_3C.nii.gz'
                img = sitk.ReadImage(str(data_path / case / img_name))
                img_array = sitk.GetArrayFromImage(img)
                if i == 0:
                    ax[i][j].set_title(case, fontsize=14)
                if (j == 0):
                    y_label = name
                    ax[i][j].set_ylabel(y_label, fontsize=14)
                ax[i][j].imshow(img_array[slice_n, :, :], cmap=cmap)
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
        for i, (name, seg_path) in enumerate(zip(sublist_exp, sublist_seg_p), 2):
            for j, case in enumerate(cases):
                img_name = f'{case}.nii.gz'
                img = sitk.ReadImage(str(seg_path / img_name))
                img_array = sitk.GetArrayFromImage(img)
                if (j == 0):
                    y_label = name
                    ax[i][j].set_ylabel(y_label, fontsize=14)
                ax[i][j].imshow(img_array[slice_n, :, :], cmap='viridis')
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])
                if i == n_rows:
                    ax[i][j].set_xlabel(case, fontsize=14)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)
        if out_path is not None:
            plt.savefig(out_path/f'brains_{n}.svg', bbox_inches='tight', format='svg')
