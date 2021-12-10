import typing

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.decomposition as mat_decomp
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    confusion_matrix
)

from scripts.metrics import (
    macro_averaged_mean_absolute_error,
    macro_averaged_mean_squared_error,
)

def generate_labels(y, labels):
    '''
    Given labels for a k-class ordinal classication problem, generates k-1 binary classification labels.
    '''
    k = labels.shape[0]

    ys = []
    for i in labels[:-1]:
        yl = np.where(y > i, 1, 0)
        ys.append(yl)

    return ys


class OrdinalClassifier:
    '''
    Implements an ordinal classifier with the method by Frank and Hall.
    Source: www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf
    '''

    def __init__(self, labels, classifier, params, preprocess=None):
        self.labels = labels
        self.k = len(labels)
        self.models = []
        self.preprocess = preprocess

        # Model options
        options = {
            'logistic': LogisticRegression,
            'tree': DecisionTreeClassifier,
            'forest': RandomForestClassifier,
            'svm': SVC,
            'xgb': XGBClassifier
        }

        # To specify each classifier independently
        clss = [classifier]*(self.k-1) if type(classifier) == str else classifier
        prms = [params]*(self.k-1) if type(params) == dict else params

        for i in range(self.k-1):
            self.models.append(options[clss[i]](**prms[i]))


    def train(self, X, y):
        ys = generate_labels(y, self.labels)
        if self.preprocess is not None:
            X = self.preprocess(X)
        for i in range(self.k-1):
            self.models[i].fit(X, ys[i])


    def predict(self, X):
        if self.preprocess is not None:
            X = self.preprocess(X)
        
        P = np.zeros((X.shape[0], self.k))

        P[:,0]        = self.models[0].predict_proba(X)[:,0]
        P[:,self.k-1] = self.models[-1].predict_proba(X)[:,1]

        #TODO: what to do about negative values... not discussed in the paper
        for i in range(1, self.k-1):
            P[:,i] = self.models[i-1].predict_proba(X)[:,1] \
                     - self.models[i].predict_proba(X)[:,1]

        idx = np.argmax(P, axis=1)
        pred = self.labels[idx]

        return pred, P


def create_sklearn_layout(df: pd.DataFrame,
                          y_label: str="dangerLevel"):
    y = df[y_label]
    X = df.drop(y_label, axis=1)
    return X, y


def prepare_data(df, test_split, drop_cat, PCA, verbose=True):
    cat_cols = [
            "station",
            "Is_month_end",
            "Is_month_start",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start"
        ]

    if drop_cat:
        if verbose:
            print("Drop_cat is set to true, dropping: {cat_cols}".format(cat_cols=cat_cols))
        df = df.drop(columns=cat_cols)
    else:
        if verbose:
            print("Drop_cat is set to false, using a one-hot encoding for: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Split and scale the samples
    mask = df.date < test_split
    X, y = create_sklearn_layout(df)
    X = X.drop("date", axis=1)
    X_train = X[mask]
    y_train = y[mask]
    mean_train = X_train.mean()
    std_train = X_train.std()
    X_train = (X_train - mean_train) / std_train
    X_test = X[~mask]
    y_test = y[~mask]
    X_test = (X_test - mean_train) / std_train
    
    if PCA:
        pca = mat_decomp.PCA(0.95).fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        
    return X_train, y_train, X_test, y_test
    

def fit_and_predict(df: pd.DataFrame,
                    test_split: str='2015-01-01',
                    drop_cat: bool=True,
                    PCA: bool=False,
                    params: list=None,
                    classifier: list=None,
                    verbose: bool=True,
                    return_P: bool=False):
    """Full modelling pipeline for the Frank et al. approach."""
    
    ## Setup params
    labels = np.unique(df.dangerLevel)
    k = labels.shape[0]

    ### Use default ensemble
    if classifier is None:
        classifier = ['xgb', 'xgb', 'tree']

    ### Use default parameters
    if params is None:
        defaults = {
            'logistic': {'max_iter':1000},
            'tree': {},
            'forest': {},
            'svm': {'probability':True},
            'xgb': {'verbosity':0},
        }
        params = [ defaults[c] for c in classifier ]
    
    # Split and prepare data
    X_train, y_train, X_test, y_test = prepare_data(df, test_split, drop_cat, PCA, verbose)
    
    ## Fit the model
    clf = OrdinalClassifier(labels, classifier, params)
    clf.train(X_train, y_train)

    # Predict on unseen data
    y_pred, P = clf.predict(X_test)

    if verbose:
        non_pos = np.count_nonzero(P < 0)
        print('P matrix:')
        print(f'  Non-positive: {non_pos} ({non_pos/P.size:.3%})')

        # Accuracy
        acc = balanced_accuracy_score(y_test, y_pred)
        print(f'Balanced Accuracy:\t{acc:.3f}\n')

        # Per-class Accuracy
        print("Per Class Accuracy")
        cm = confusion_matrix(y_test, y_pred)
        class_accuracy = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).diagonal()
        print(f"{pd.Series(index=labels, data=np.around(class_accuracy,3))}\n")

        # Metrics summary
        print(classification_report(y_test, y_pred))
        macmse = macro_averaged_mean_squared_error(y_test, y_pred, average=None)
        macmad = macro_averaged_mean_absolute_error(y_test, y_pred, average=None)
        print(f"\nMacro-averaged mean squared error: {macmse}")
        print(f"Macro-averaged mean absolute error: {macmad}")

        # Confusion matrix
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
        
        # Negative Probabilities
        data_P = pd.DataFrame(np.where(P < 0, P, 0), columns=['1','2','3','4'])
        counts = data_P.astype(bool).sum(axis=0)
        
        plt.figure()
        ax = sns.stripplot(data=data_P.iloc[:,1:3], orient='h', size=10)
        ax = sns.violinplot(data=data_P.iloc[:,1:3], orient='h', scale='count', inner=None, color='0.8')
        line1 = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=sns.color_palette()[0], markersize=10)
        line2 = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor=sns.color_palette()[1], markersize=10)
        plt.legend((line1, line2), (f'{counts[1]}',f'{counts[2]}'), loc='lower left')
        plt.title('Distribution of Negative Probabilities')
        plt.xlabel('Probability')
        plt.ylabel('Danger Level')
        plt.savefig('./figures/frank_negative_probabilities.pdf')
    
    if return_P:
        return y_pred, y_test, clf, P
    else:
        return y_pred, y_test, clf
