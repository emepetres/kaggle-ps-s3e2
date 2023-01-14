import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_count(
    df: pd.core.frame.DataFrame, col_list: list, title_name: str = "Train"
) -> None:
    """Draws the pie and count plots for categorical variables.

    Args:
        df: train or test dataframes
        col_list: a list of the selected categorical variables.
        title_name: 'Train' or 'Test' (default 'Train')

    Returns:
        subplots of size (len(col_list), 2)
    """
    f, axes = plt.subplots(len(col_list), 2, figsize=(15, 24))
    plt.subplots_adjust(wspace=0)

    for col_name, ax in zip(col_list, axes):
        s1 = df[col_name].value_counts()
        N = len(s1)

        outer_sizes = s1
        inner_sizes = s1 / N

        outer_colors = ["#9E3F00", "#eb5e00", "#ff781f", "#ff9752", "#ff9752"]
        inner_colors = ["#ff6905", "#ff8838", "#ffa66b"]

        ax[0].pie(
            outer_sizes,
            colors=outer_colors,
            labels=s1.index.tolist(),
            startangle=90,
            frame=True,
            radius=1.3,
            explode=([0.05] * (N - 1) + [0.3]),
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
            textprops={"fontsize": 12, "weight": "bold"},
        )

        textprops = {"size": 13, "weight": "bold", "color": "white"}

        ax[0].pie(
            inner_sizes,
            colors=inner_colors,
            radius=1,
            startangle=90,
            autopct="%1.f%%",
            explode=([0.1] * (N - 1) + [0.3]),
            pctdistance=0.8,
            textprops=textprops,
        )

        center_circle = plt.Circle((0, 0), 0.68, color="black", fc="white", linewidth=0)
        ax[0].add_artist(center_circle)

        x = s1
        y = s1.index.astype(str)

        sns.barplot(x=x, y=y, ax=ax[1], palette="YlOrBr_r", orient="horizontal")

        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[1].tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        for i, v in enumerate(s1):
            ax[1].text(
                v, i + 0.1, str(v), color="black", fontweight="bold", fontsize=12
            )

        plt.title(col_name)
        plt.setp(ax[1].get_yticklabels(), fontweight="bold")
        plt.setp(ax[1].get_xticklabels(), fontweight="bold")
        ax[1].set_xlabel(col_name, fontweight="bold", color="black")
        ax[1].set_ylabel("count", fontweight="bold", color="black")

    f.suptitle(f"{title_name} Dataset", fontsize=20, fontweight="bold")
    plt.tight_layout()
    plt.show()


def pair_plot(df: pd.core.frame.DataFrame, title_name: str, hue: str) -> None:
    """Draws the pairplot for the selected dataframe.

    Args:
        df: train, test or combined dataframes
        title_name: any string title
        hue: a specified categorical column name

    Returns:
        pairplots
    """
    s = sns.pairplot(df, hue=hue, palette=["#9E3F00", "#eb5e00"])
    s.fig.set_size_inches(16, 12)
    s.fig.suptitle(title_name, y=1.08)
    plt.show()


def plot_correlation_heatmap(
    df: pd.core.frame.DataFrame, title_name: str = "Train correlation"
) -> None:
    """Draws the correlation heatmap plot.

    Args:
        df: train or test dataframes
        title_name: 'Train' or 'Test' (default 'Train correlation')

    Returns:
        subplots of size (len(col_list), 2)
    """

    corr = df.corr()
    fig, axes = plt.subplots(figsize=(20, 10))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidths=0.5, cmap="YlOrBr_r", annot=True)
    plt.title(title_name)
    plt.show()
