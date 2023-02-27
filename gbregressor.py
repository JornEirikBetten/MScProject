from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from data_preprocessing import transform_data, load_data, scale_features
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

fig_path = os.getcwd() + "/results/figures/"
data_path = "/home/jeb/Documents/MScProject/Project/datasets/data_Vaska"
gp = "/gpVaska_vectors.csv"
nbo = "/nboVaska_vectors.csv"

target = "target_barrier"
df, target = load_data(data_path+gp, target)

# Data splitting
x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.1, random_state=123)
x_train, x_test = scale_features(x_train, x_test)
#x_train, x_test, y_train, y_test, y_scaler = transform_data(x_train, y_train, x_test, y_test, n_components=25, use_pca=False)


params = {
    "n_estimators": 2500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

reg = GradientBoostingRegressor(**params)
reg.fit(x_train, y_train)
test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
r2_train = np.zeros((params["n_estimators"],), dtype=np.float64)
r2_test = np.zeros((params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(x_test)):
    test_score[i] = mean_squared_error(y_test, y_pred)
    r2_test[i] = r2_score(y_test, y_pred)

for i, y_pred in enumerate(reg.staged_predict(x_train)):
    r2_train[i] = r2_score(y_train, y_pred)


fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.plot(
    np.arange(params["n_estimators"]) + 1,
    reg.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
)
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
fig.tight_layout()
plt.savefig(fig_path + "Deviance.pdf", format="pdf")

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.plot(
    np.arange(params["n_estimators"]) + 1,
    r2_train,
    "b-",
    label="Training Set R2",
)
plt.plot(
    np.arange(params["n_estimators"]) + 1, r2_test, "r-", label="Test Set R2"
)
plt.legend(loc="lower right")
plt.xlabel("Boosting Iterations")
plt.ylabel("R2")
fig.tight_layout()
plt.savefig(fig_path + "R2.pdf", format="pdf")

feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)[-20:-1]
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(df.columns)[sorted_idx])
plt.title("Feature Importance (MDI)")

result = permutation_importance(
    reg, x_test, y_test, n_repeats=25, random_state=42, n_jobs=-1
)
sorted_idx = result.importances_mean.argsort()[-25:-1]
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(df.columns)[sorted_idx],
)
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.savefig(fig_path + "feature_importance.pdf", format="pdf")

print(np.array(df.columns)[sorted_idx])
