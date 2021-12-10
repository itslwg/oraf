import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error,
    precision_score,
    recall_score,
    confusion_matrix
)


def macro_averaged_mean_absolute_error(y_true, y_pred, average=None):
    """Computes macro-averaged MAD
    
    Set to merge in next imbalanced-learn release. Source and PR: 
    https://github.com/scikit-learn-contrib/imbalanced-learn/pull/780/files#diff-b826a80767761457bec8eb89eaa2190ca9864011b15ddca453228ed6a07eb2d3
    """
    all_mae = []
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    for class_to_predict in np.unique(y_true):
        index_class_to_predict = np.flatnonzero(y_true == class_to_predict)
        mae_class = mean_absolute_error(y_true[index_class_to_predict],
                                        y_pred[index_class_to_predict])
        all_mae.append(mae_class)
    ma_mae = sum(all_mae) / len(all_mae)
    return ma_mae


def macro_averaged_mean_squared_error(y_true, y_pred, average=None):
    """Computes macro-averaged MSE
    
    Set to merge in next imbalanced-learn release. Source and PR:
    https://github.com/scikit-learn-contrib/imbalanced-learn/pull/780/files#diff-b826a80767761457bec8eb89eaa2190ca9864011b15ddca453228ed6a07eb2d3
    """
    all_mse = []
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    for class_to_predict in np.unique(y_true):
        index_class_to_predict = np.flatnonzero(y_true == class_to_predict)
        mse_class = mean_squared_error(y_true[index_class_to_predict],
                                       y_pred[index_class_to_predict])
        all_mse.append(mse_class)
    ma_mse = sum(all_mse) / len(all_mse)
    return ma_mse


def report_performance(y_pred, y_true, approach):
    """Compile results on test set."""
    p = dict(
        y_pred=y_pred, 
        y_true=y_true, 
        average=None
    )
    metrics = [
        precision_score,
        recall_score,
        macro_averaged_mean_squared_error,
        macro_averaged_mean_absolute_error
    ]
    tbl = pd.concat([
        pd.Series(metric(**p)) 
        for metric in metrics
    ]).to_frame().round(3)
    labels = np.unique(y_pred)
    r = ["Rec_" + str(i) for i in labels]
    p = ["Prec_" + str(i) for i in labels]
    index = p + r + ["MMSE", "MMAD"] 
    
    tbl.columns = [approach]
    tbl.index = index
    
    return tbl

def plot_confusion_matrix(y_true, y_pred):
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9,9))
    sns.heatmap(cm,
                annot=True,
                fmt=".3f",
                linewidths=.5,
                square = True,
                xticklabels=labels,
                yticklabels=labels,
                cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix', size=15)