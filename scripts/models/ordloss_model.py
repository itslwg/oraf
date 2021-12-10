# Import packages

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sklearn.decomposition as mat_decomp
import sklearn.decomposition as mat_decomp
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder, add_dummy_feature
import random

import os, sys
from pathlib import Path as P
import time
from tqdm import tqdm

def random_seed(seed_value):
	"""Ensure reproducibility"""
	np.random.seed(seed_value)
	torch.manual_seed(seed_value) 
	random.seed(seed_value)

def neg_log_loss(theta, output, y, class_w):
    """Implements negative log likelihood for ordinal regression.
    From https://github.com/ebeckwith/Ordinal_Regression/blob/master/Ordinal_Regression.ipynb	
    """
    theta_km1 = torch.roll(theta, 1) #\theta_{k-1}
    theta_km1[0] = -99999 #\theta_0 = i\inf
    t1 = theta_km1[y]
    
    t0 = theta[y]
    
    eps = 1e-10 # Deal with potential issues in the log.
           
    l = torch.sigmoid(t0-output)-torch.sigmoid(t1-output)      
    l = torch.clamp(l, eps, 1-eps)
    nll = - torch.log(l)
    nll *= class_w[y]
    return torch.mean(nll)


class ordinalRegression(nn.Module):
    """Implements 2 layer fully connected neural network """
    def __init__(self, n_features, start, end, n_classes, nodes):
        super(ordinalRegression, self).__init__()
        
        l2nodes = int(nodes/2)

        self.l1 = nn.Linear(n_features, nodes) 
        self.l2 = nn.Linear(nodes, l2nodes)
        self.output = nn.Linear(l2nodes, 1)
	
        self.theta = nn.Parameter(
            torch.linspace(start=start, end=end,
                           steps=n_classes, dtype=torch.float)
            )        
        
        self.activation = nn.LeakyReLU()
        
        
    def forward(self, x):
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)        
        x = self.activation(x)
        return self.output(x), self.theta


def train(X_train, y_train, n_classes, nodes, class_weights, 
          epochs=10, batch_size=4, lr=0.01, start=-1,
          end=5, verbose=True, seed=1, **kwargs):
    """Implements training loop"""

    random_seed(seed) # Reproducibility

    # Convert X, y into Tensor. 
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    class_weights = torch.from_numpy(class_weights).float()
    
    # Training variables
    model = ordinalRegression(X_train.shape[1], start, end, n_classes, nodes)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_is = []
    
    for ep in range(epochs):
        for i in (tqdm(range(0,len(X_train), batch_size)) if verbose else range(0,len(X_train), batch_size)):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            output, theta = model(X_batch)
            loss = neg_log_loss(theta, output, y_batch, class_weights)

            loss.backward() # Compute gradients 

            optimizer.step() # Update w, theta
            optimizer.zero_grad()

            loss_is.append(loss)

            with torch.no_grad(): # Projected Gradient Descent, enforce constraint \theta_1 < \theta_2 < \theta_k
                for i in range(len(theta)-1):
                    theta[i].clamp_(-1.0e6, theta[i+1].data)

    return loss_is, model, theta.detach()


def plot_loss(losses, title="In sample loss", path="figures",save=False):
    """Plot the loss function at each iteration."""
    plt.plot(losses)
    plt.xlabel("# of iterations")
    plt.ylabel("Loss")
    plt.title(title)
    if save:
        plt.savefig(f"{path}/is_loss_{time.time():.0f}.pdf")
    plt.show()


def plot_theta(latent, y, theta, path="figures",save=False):
	"""Plot the thresholds and the output of the model for each class.
	https://github.com/ebeckwith/Ordinal_Regression/blob/master/Ordinal_Regression.ipynb
	"""
	data = pd.DataFrame(latent, columns=["y_hat"]) # Save latent variable
	data["class"] = y+1
	grid = sns.FacetGrid(data, hue="class", aspect=3)
	grid.map(sns.kdeplot, "y_hat", linewidth=0.1, fill=True)
	for t in theta:
		plt.axvline(x=t, color="black")
	grid.add_legend()
	plt.ylabel("Estimated Density")
	plt.xlabel(r"$y*$")
	if save:
		plt.savefig(f"./figures/Ord_Loss_theta_plot.pdf")
	plt.show()
	
def make_predictions(X, model, theta):
    """Make predictions from the model."""
    X_torch = torch.from_numpy(X).float()
    with torch.no_grad():
        latent,_ = model(X_torch)
    preds = torch.sum(latent>theta,dim=1)
    preds = torch.where(preds>=len(theta), len(theta)-1, preds) # Ensure class 4 is the max
    return preds, latent


def fit_and_predict(df_in: pd.DataFrame,
					opt_parameters:dict,
					test_split: str='2015-01-01'):
	"""Complete modelling pipeline for the OrdLoss approach."""

	df = df_in.copy() # Don't change the input df.

	to_drop=["dangerLevel", "date"]

	X = df.drop(columns=to_drop)  # TODO: Note that I added station to the data
	y = df["dangerLevel"]
	y -= 1 # So we can index by y.

    # Dummies
	cat_cols = [
		"station",
		"Is_month_end",
		"Is_month_start",
		"Is_quarter_end",
		"Is_quarter_start",
		"Is_year_end",
		"Is_year_start"
	]       
	X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
	
	# Split train & test
	mask = df.date < test_split
	X_train = X[mask].values
	y_train = y[mask].values
	X_test = X[~mask].values
	y_test = y[~mask].values

	# Scale data without bias
	scaler = StandardScaler()
	encoder = LabelEncoder()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test) 
	y_train = encoder.fit_transform(y_train) # Turn into integer
	y_test = encoder.transform(y_test)
	
	# Map PCA to train and test sets
	pca = mat_decomp.PCA(0.95).fit(X_train)
	X_train = pca.transform(X_train)
	X_test = pca.transform(X_test)

	# Classes
	n_classes = len(np.unique(y))
	class_weights = len(y) / (n_classes * np.bincount(y)) # Sklearn balanced

	# Train
	losses, model, theta = train(X_train, y_train, 
								 n_classes, **opt_parameters)

	# Make predictions
	y_pred_oos, latent_oos = make_predictions(X_test, model, theta)

	plot_theta(latent_oos, y_test, theta, save=True)

	# Output classification metrics
	print(metrics.classification_report(y_test+1, y_pred_oos+1))
	acc = metrics.balanced_accuracy_score(y_test, y_pred_oos.numpy())
	print(f'Balanced Accuracy:\t{acc:.3f}\n')

	return y_pred_oos, y_test
