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
from numpy.random import default_rng

def objective(learning_rate, depth, width):
    fig_path = os.getcwd() + "/results/figures/"
    data_path = "/home/jeb/Documents/MScProject/Project/datasets/data_Vaska"
    gp = "/gpVaska_vectors.csv"
    nbo = "/nboVaska_vectors.csv"
    """
    rng = default_rng()
    x = rng.uniform(0, 1, size=(1000))
    y = lambda x: 2*np.exp(-x**2) + rng.normal(loc=0, scale=2.0)

    x_train, x_test, y_train, y_test = train_test_split(x.reshape(len(x), 1), y(x).reshape(len(x), 1), test_size=0.8, random_state=int(time.time()))
    x_train = torch.from_numpy(x_train); x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train); y_test = torch.from_numpy(y_test)
    """
    target = "target_barrier"
    df, target = load_data(data_path+gp, target)

    top_25_features = ['Z-3_FR_AA','chi-1_MS_AA','T-4_FR_AA','d-1_MD_BB','T-6_FD_AB',
                       'chi-1_MA_AA', 'chi-2_MR_AA','BO-1_MD_BBavg','Z-0_FA_AA','S-2_MD_AB',
                       'I-1_MD_AB', 'chi-3_MD_AA','BO-0_MR_AB','S-1_MR_AA','d-0_MS_BBavg',
                       'I-2_MD_AB', 'chi-1_MR_AA','d-0_MA_BBavg','d-0_MR_AB','BO-2_MD_BBavg',
                       'Z-1_FA_AA', 'chi-1_MD_AA','d-1_MD_BBavg','Z-2_FA_AA']

    df = df[top_25_features]

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
    for epoch in range(50):
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
           'depth': (1, 5),
           'width': (4, 512)}


# Define the Gaussian process regression model for the Bayesian optimization
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
)

# Define the number of iterations for Bayesian optimization
num_iterations = 10

# Define the utility function to use (Expected Improvement)
utility = UtilityFunction(kind="ucb", xi=0.0, kappa_decay=0.9)


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
optimizer.maximize(n_iter=500)

print(f"Final result:")
print(f"    loss={optimizer.max['target']:.4f}")
print(f"    depth={int(optimizer.max['params']['depth'])}")
print(f"    width={int(optimizer.max['params']['width'])}")
print(f"    learning_rate={optimizer.max['params']['learning_rate']:.4f}")
