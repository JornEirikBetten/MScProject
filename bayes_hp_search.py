import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import pandas as pd
from data_preprocessing import transform_data, load_data
from mlmodels import MLP, train_and_test
from plotter_functions import plot_lineplot
import time

def objective(learning_rate, depth, width):
    fig_path = os.getcwd() + "/results/figures/"
    data_path = "/home/jeb/Documents/MScProject/Project/datasets/data_Vaska"
    gp = "/gpVaska_vectors.csv"
    nbo = "/nboVaska_vectors.csv"

    target = "target_barrier"
    df, target = load_data(data_path+gp, target)

    # Data splitting
    x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.8, random_state=int(time.time()))
    y_train = y_train.to_numpy().reshape(len(y_train), 1)
    y_test = y_test.to_numpy().reshape(len(y_test), 1)
    x_train, x_test, y_train, y_test, y_scaler = transform_data(x_train, y_train, x_test, y_test, n_components=25, use_pca=False)
    x_train = torch.from_numpy(x_train); x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train); y_test = torch.from_numpy(y_test)

    n_features = x_train.shape[-1]
    n_targets = 1
    # Initiate model with proper width and depth
    model = MLP(n_features, n_targets, int(depth), int(width))

    # Loss
    criterion = torch.nn.MSELoss()
    # Define the optimizer with the given learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        output = model(x_test)
        loss = criterion(output, y_test)

    return -loss

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

# Define the search space for the hyperparameters
pbounds = {'learning_rate': (1e-4, 1e-0),
           'depth': (1, 6),
           'width': (16, 512)}


# Define the Gaussian process regression model for the Bayesian optimization
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
)

# Define the number of iterations for Bayesian optimization
num_iterations = 10

# Define the utility function to use (Expected Improvement)
utility = UtilityFunction(kind="poi", kappa=2.5, xi=0.0, kappa_decay=1.0)


#for i in range(num_iterations):
# Get the next hyperparameters to try from the Bayesian optimization model
next_hyperparameters = optimizer.suggest(utility)

# Evaluate the objective function with the next hyperparameters
objective_value = objective(**next_hyperparameters)

# Update the Bayesian optimization model with the new data point
optimizer.register(
    params=next_hyperparameters,
    target=objective_value,
)
# Report the best objective value found so far
optimizer.maximize(n_iter=15)

print("Final result:", optimizer.max)
