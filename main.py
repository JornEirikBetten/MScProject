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


"""
plotting parameters
"""

fontsize = "x-large"
params = {"font.family": "serif",
          "font.sans-serif": ["Computer Modern"],
          "axes.labelsize": fontsize,
          "legend.fontsize": fontsize,
          "xtick.labelsize": fontsize,
          "ytick.labelsize": fontsize,
          "legend.handlelength": 2
          }

plt.rcParams.update(params)



fig_path = os.getcwd() + "/results/figures/"
data_path = "/home/jeb/Documents/MScProject/Project/datasets/data_Vaska"
gp = "/gpVaska_vectors.csv"
nbo = "/nboVaska_vectors.csv"

target = "target_barrier"
df, target = load_data(data_path+gp, target)

# Data splitting
x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.8, random_state=123)
y_train = y_train.to_numpy().reshape(len(y_train), 1)
y_test = y_test.to_numpy().reshape(len(y_test), 1)
x_train, x_test, y_train, y_test, y_scaler = transform_data(x_train, y_train, x_test, y_test, n_components=25, use_pca=False)
x_train = torch.from_numpy(x_train); x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train); y_test = torch.from_numpy(y_test)

layers = [50, 50]
n_features = x_train.shape[-1]
n_targets = 1
model = MLP(n_features, n_targets, layers)

if torch.cuda.is_available():
    x_train = x_train.cuda(); x_test = x_test.cuda()
    y_train = y_train.cuda(); y_test = y_test.cuda()
    model = model.cuda()

# Loss and optimizer
lr = 1e-3
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

epochs = 1000
train_err, test_err, preds, output = train_and_test(x_train, x_test, y_train, y_test, epochs, model, criterion, optimizer, scheduler)
y_train = y_train.cpu(); y_test = y_test.cpu()
preds = preds.cpu().numpy(); output = output.cpu().detach().numpy()
R2_Train = r2_score(y_train, output)
R2_test = r2_score(y_test, preds)
print(f"R2 Train: {R2_Train}")
print(f"R2 Test: {R2_test}")

#r2=plot_lineplot(np.array([i for i in range(epochs)]), [R2_Train, R2_test],["Train", "Test"], "Epoch", "R2", ["tab:red", "tab:blue"], fig_path, "R2.pdf")
MSE=plot_lineplot(np.array([i for i in range(epochs)]), [train_err, test_err], ["Train", "Test"], "Epoch", "MSE", ["tab:red", "tab:blue"], fig_path, "MSE.pdf")
