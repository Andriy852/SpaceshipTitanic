import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def pct_heatmap(data: pd.DataFrame, ycolumn: str, 
                xcolumn: str, ax: plt.Axes) -> plt.Axes:
    """
    Creates a percentage heatmap showing the distribution of a 
    categorical variable (ycolumn) across another categorical variable
    (xcolumn). The values in the heatmap represent the percentage distribution
    of ycolumn values for each xcolumn category.

    Parameters:
    data : DataFrame
        The input data containing the categorical variables to plot.
    ycolumn : str
        The name of the categorical variable to be displayed on the y-axis.
    xcolumn : str
        The name of the categorical variable to be displayed on the x-axis.
    ax : plt.Axes
        The Matplotlib axis on which to plot the heatmap.

    Returns:
    plt.Axes
        The axis object with the generated heatmap.
    """
    count = pd.crosstab(data[ycolumn], data[xcolumn])
    pct = count.div(count.sum(axis=0), axis=1) * 100
    sns.heatmap(pct, annot=True, cmap='coolwarm', fmt=".1f", 
                cbar_kws={'label': 'Percentage'}, cbar=False, ax=ax)
    return ax

def heatmap_association(data: pd.DataFrame, xcolumn: str, 
                        ycolumn1: str, ycolumn2: str) -> None:
    """
    Parameters:
    data : pd.DataFrame
        The input dataset containing the categorical variables to visualize.
    xcolumn : str
        The name of the categorical variable to be displayed on the x-axis.
    ycolumn1 : str
        The name of the first categorical variable to be displayed 
        on the y-axis in the first heatmap.
    ycolumn2 : str
        The name of the second categorical variable to be displayed on 
        the y-axis in the second heatmap.

    Returns:
    None
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for i, column in enumerate([ycolumn1, ycolumn2]):
        pct_heatmap(data, column, xcolumn, ax[i])
        ax[i].set_title(column)
    plt.suptitle(f"Distribution of {ycolumn1}, {ycolumn2} depending"
                 f" on different {xcolumn}(in percents)", fontsize=16)

def get_scores(model: BaseEstimator, X: np.ndarray,
               y: np.ndarray, fit: bool = True) -> Dict[str, float]:
    """
    Compute performance scores on the data.

    Parameters:
    model (BaseEstimator): The machine learning model
    X (np.ndarray): The feature matrix used.
    y (np.ndarray): The target vector used.
    fit (bool): If True, the model will be fitted to the data. Default is True.

    Returns:
    Dict[str, float]: A dictionary containing accuracy, recall, precision, and f1 scores.
    """
    if fit:
        model.fit(X, y)

    model_predict = model.predict(X)

    scores = {
        "accuracy": accuracy_score(y, model_predict),
        "recall": recall_score(y, model_predict),
        "precision": precision_score(y, model_predict),
        "f1": f1_score(y, model_predict)
    }

    return scores

def customize_bar(position: str, axes, 
                  values_font=12, pct=False, round_to=0) -> None:
    """
    Function, which customizes bar chart
    Takes axes object and:
        - gets rid of spines
        - modifies ticks
        - adds value above each bar
    Parameters:
        - position(str): modify the bar depending on how the
        bars are positioned: vertically or horizontally
    Return: None
    """
    # get rid of spines
    for spine in axes.spines.values():
        spine.set_visible(False)
    # modify ticklabels
    if position == "v":
        axes.set_yticks([])
        for tick in axes.get_xticklabels():
            tick.set_rotation(0)
    if position == "h":
        axes.set_xticks([])
        for tick in axes.get_yticklabels():
            tick.set_rotation(0)
    # add height value above each bar
    for bar in axes.patches:
        if bar.get_width() == 0:
            continue
        if position == "v":
            text_location = (bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 1/100*bar.get_height())
            value = bar.get_height()
            location = "center"
        elif position == "h":
            text_location = (bar.get_width(),
                             bar.get_y() + bar.get_height() / 2)
            value = bar.get_width()
            location = "left"
        if pct:
            value = f"{round(value * 100, round_to)}%"
        elif round_to == 0:
            value = str(int(value))
        else:
            value = str(round(value, round_to))
        axes.text(text_location[0],
                text_location[1],
                str(value),
                fontsize=values_font,
                ha=location)
        
def plot_cat_columns(data: pd.DataFrame, columns: List[str], 
                     title: str, figsize: Tuple[int, int]) -> None:
    """
    Plots count plots for specified categorical columns in a DataFrame.

    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame containing the data to be plotted.
    columns : List[str]
        The list of column names to be plotted.
    title : str
        The title for the entire figure.
    figsize : Tuple[int, int]
        The size of the figure (width, height).

    Returns:
       fig, plt
    """
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16, y=0.91)
    
    for i, column in enumerate(columns):
        ax = fig.add_subplot(3, 3, i+1)
        sns.countplot(x=column, data=data, 
                      ax=ax, color="red")

        customize_bar(axes=ax, position="v", values_font=10)
        ax.set_xlabel("")
        ax.set_title(column.capitalize(), fontsize=12)

        # get rid of ylabels for middle or right axes
        if i % 3 != 0:
            ax.set_ylabel("")

    return fig, plt
        
def make_mi_scores(X: pd.DataFrame, y: pd.Series, 
                   discrete_features: Optional[List[bool]] = None,
                   random_state: Optional[int] = None) -> pd.Series:
    """
    Calculate Mutual Information scores for features in a dataset.

    Parameters:
    ----------
    X : pd.DataFrame
        The input features as a DataFrame where each column is a feature.
    y : pd.Series
        The target variable.
    discrete_features : Optional[List[bool]], optional
        A list indicating whether each column in X is discrete (True) or continuous (False). 
        If not provided, all features are assumed to be continuous.
    random_state : Optional[int], optional
        Random seed used to ensure reproducibility of the results. 
        If not provided, results may vary each time the function is run.

    Returns:
    -------
    pd.Series
        A Series containing the MI scores for each feature, sorted in descending order.
    """
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores