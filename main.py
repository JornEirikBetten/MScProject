import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import pandas as pd
from data_preprocessing import transform_data, load_data, scale_features
from mlmodels import MLP, train_and_test, train_and_test_w_batching
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
"""
#wanted_features = ['Z-5_FS_AA','I-3_FD_AB','d-1_MR_BB','Z-0_FD_AB','Z-3_FR_AA','d-2_MA_BB',
                   'BO-4_FA_BB','Z-1_FD_AB','d-1_MR_AB','Z-5_FD_AB','chi-0_MS_AB',
                   'd-1_MD_BB','T-2_MR_AA','chi-2_MR_AA','Z-2_FR_AA','S-1_MS_AA',
                   'd-1_MS_BBavg','S-1_MD_AA','S-1_MA_AA','chi-0_MD_AB','S-2_MR_AA',
                   'I-0_MR_AB','Z-1_MR_AA','d-1_MA_BBavg','I-1_MD_AB','Z-1_MA_AA',
                   'chi-1_MS_AA','chi-3_MD_AA','S-2_MD_AB','chi-0_FA_AB','T-6_FD_AB',
                   'Z-2_MR_AA','chi-1_MR_AA','d-0_MR_AB','Z-0_FA_AA','d-1_MR_BBavg',
                   'S-1_MR_AA','chi-1_MA_AA','I-2_MD_AB','d-0_MS_BBavg','BO-1_MD_BBavg',
                   'S-0_MD_AB','BO-2_MD_BBavg','d-0_MA_BBavg','Z-1_FA_AA','BO-0_MR_AB',
                   'Z-2_FA_AA','d-1_MD_BBavg','chi-1_MD_AA']
"""
wanted_features = ['chi-2_MR_AA','chi-3_MD_AA','Z-0_FA_AA','S-2_MD_AB','BO-1_MD_BBavg',
                   'chi-1_MA_AA','I-1_MD_AB','S-1_MR_AA','BO-0_MR_AB','I-2_MD_AB',
                   'd-0_MS_BBavg','d-0_MA_BBavg','chi-1_MR_AA','d-0_MR_AB','chi-1_MD_AA',
                   'BO-2_MD_BBavg','Z-1_FA_AA','d-1_MD_BBavg','Z-2_FA_AA']
#df=df[wanted_features]

# Data splitting
x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=1233)
x_train = x_train.to_numpy(); x_test = x_test.to_numpy()
y_train = y_train.to_numpy().reshape(len(y_train), 1)
y_test = y_test.to_numpy().reshape(len(y_test), 1)
x_train, x_test = scale_features(x_train, x_test)
x_train = torch.from_numpy(x_train); x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train); y_test = torch.from_numpy(y_test)


width = 247
depth = 4
n_features = x_train.shape[-1]
n_targets = 1
print(f"Running on {n_features}/1030 features.")
model = MLP(n_features, n_targets, depth, width)

if torch.cuda.is_available():
    x_train = x_train.cuda(); x_test = x_test.cuda()
    y_train = y_train.cuda(); y_test = y_test.cuda()
    model = model.cuda()

# Loss and optimizer
lr = 1e-3
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

epochs = 2000
train_err, test_err, preds, output, r2_test, r2_train= train_and_test(x_train, x_test, y_train, y_test, epochs, model, criterion, optimizer, scheduler)
y_train = y_train.cpu(); y_test = y_test.cpu()
preds = preds.cpu().numpy(); output = output.cpu().detach().numpy()
model.eval()
output = model(x_train)
R2_Train = r2_score(y_train, output.cpu().detach().numpy())
R2_test = r2_score(y_test, preds)
print(f"R2 Train: {R2_Train}")
print(f"R2 Test: {R2_test}")

#r2=plot_lineplot(np.array([i for i in range(epochs)]), [R2_Train, R2_test],["Train", "Test"], "Epoch", "R2", ["tab:red", "tab:blue"], fig_path, "R2.pdf")
MSE=plot_lineplot(np.array([i for i in range(epochs)]),
                  [train_err, test_err],
                  ["Train", "Test"],
                  "Epoch",
                  "MSE",
                  ["tab:red", "tab:blue"],
                  fig_path,
                  "MSE.pdf")
R2=plot_lineplot(np.array([i for i in range(200, epochs, 1)]),
                 [r2_train[199:-1], r2_test[199:-1]],
                 ["R2 Train", "R2 Test"],
                 "Epoch",
                 "R2",
                 ["tab:red", "tab:orange"],
                 fig_path,
                 "R2.pdf")
