import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import seaborn as sns
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
import time
import random
from prettytable import PrettyTable
from datetime import date    
today = date.today().isoformat()

# Helper files with all ML model helpers :)
from predict_ml_model_helpers import *

# Dynamically load configs :)
import argparse
from dynamic_config_load import inject_config
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Name of the config file (without .py)")
args = parser.parse_args()

inject_config(args.config, globals())

# Set seeds if the run is deterministic
if DETERMINISTIC:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

figure_counter = 0

# --------- BEGIN Preprocessing ---------
dataset_ml_models = os.path.join('../data', RUN_NAME, "ml_models")
# Create ML Library
if not os.path.exists(dataset_ml_models):
    os.makedirs(dataset_ml_models)

run_metrics_filename = 'dynamic_energy_model_analysis_' + today + '.csv'
metrics_output_filepath = os.path.join(dataset_ml_models, run_metrics_filename)

# Find full dataset and put into dataframe
dataset_csv_filepath = os.path.join('../data', RUN_NAME, DF_FILENAME)
spike_data_df = pd.read_csv(dataset_csv_filepath)
spike_data_df['Latency'] = spike_data_df['Latency'] * 10**9     # Get into ns
spike_data_df['Energy'] = spike_data_df['Energy'] * 10**12      # Get into pJ

# Get the standard scaler
std_scaler = produce_or_load_common_standard_scalar(spike_data_df, LIST_OF_COLUMNS_X, dataset_ml_models, "Run_Number", TRAIN_TEST_SPLIT, VALIDATION_SPLIT, random_state=42)

# Preprocess step
spike_data_df = spike_data_df[spike_data_df['Event_Type'] == 'in-out']

# Runwise train test split :)
train_df, test_df, val_df = runwise_train_test_split(spike_data_df, test_size=TRAIN_TEST_SPLIT, val_size=VALIDATION_SPLIT, random_state=42)
X_train = train_df[LIST_OF_COLUMNS_X]
y_train = train_df[["Energy"]]
X_test = test_df[LIST_OF_COLUMNS_X]
y_test = test_df[["Energy"]]
X_val = val_df[LIST_OF_COLUMNS_X]
y_val = val_df[["Energy"]]

# Logging
print("Train Run Numbers")
print(train_df["Run_Number"].unique())
print("Validation Run Numbers")
print(val_df['Run_Number'].unique())
print("Test Run Numbers")
print(test_df["Run_Number"].unique())

print("Number of Train Samples: {}".format(X_train.shape[0]))
print("Number of Validation Samples: {}".format(X_val.shape[0]))
print("Number of Test Samples: {}".format(X_test.shape[0]))

# --------- END Preprocessing ---------
# Create table to make everything easier to visualize what the hell is going on
table = PrettyTable()
table.field_names = ["Regressor", "Train Time", "Inference Time", "MSE", "MAE", "MAPE", "R-Squared", "Average Error", "Predicted Dynamic Energy Total", "Real Dynamic Energy Total"]


# Super Baseline
# Super baseline is to just compare the MSE with using the mean value for the training dataset and apply that to the other one.
start_time = time.time()
train_y_mean = y_train.mean()
end_time = time.time()
train_time = end_time - start_time

start_time = time.time()
baseline_vec = np.full_like(y_test, fill_value=train_y_mean)
end_time = time.time()
test_time = end_time - start_time

baseline_metrics = calculate_metrics(y_test, baseline_vec)

table.add_row(["Mean Baseline", f"{train_time:.6f}", f"{test_time:.6f}"]+baseline_metrics)

# ---------------------
# Nearest-Neighbor Interpolator (Table-based method)
table_y_pred, train_time, test_time = interpolate(X_train, X_test, X_val, y_train, y_test, y_val)
baseline_metrics = calculate_metrics(y_test, table_y_pred)

table.add_row(["NN Interpolation", f"{train_time:.6f}", f"{test_time:.6f}"]+baseline_metrics)

# ----------------------
# Print Linear Regression Stuff
ols_y_pred, train_time, test_time = train_linear_regression(X_train, X_test, X_val, y_train, y_test, y_val, std_scaler)
baseline_metrics = calculate_metrics(y_test, ols_y_pred)

table.add_row(["OLS", f"{train_time:.6f}", f"{test_time:.6f}"]+baseline_metrics)

# -------------------------
# CatBoost
# NOTE: Decision Tree Based Models do not need scaled data :O
catboost_params = {
    'iterations': 500,
    'learning_rate': 0.1,
    'depth': 10,
    'l2_leaf_reg': 5,
    'subsample': 0.5,
    'verbose': False,
    'eval_metric':'RMSE'
}

if DETERMINISTIC:
    catboost_params['random_seed'] = RANDOM_SEED

catboost_model_save_name = "catboost_dynamic_energy_11_7"
cat_y_pred, train_time, test_time = run_catboost_regression(X_train, X_test, X_val, y_train, y_test, y_val, catboost_params, SAVE_CATBOOST_MODEL, os.path.join(dataset_ml_models, catboost_model_save_name),SAVE_CATBOOST_CPP)
baseline_metrics = calculate_metrics(y_test, cat_y_pred)

table.add_row(["CatBoost", f"{train_time:.6f}", f"{test_time:.6f}"]+baseline_metrics)

# ----------------------------------------
# MLP
hyperparameters_mlp = {
    'hidden_layer_sizes': (100,50),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,
    'learning_rate_init': 0.01,
    'tol':1e-5,
    'early_stopping':True,
    'validation_fraction': VALIDATION_SPLIT
}

if DETERMINISTIC:
    hyperparameters_mlp['random_state'] = RANDOM_SEED

mlp_model_save_name = "mlp_dynamic_energy_11_8"
mlp_y_pred, train_time, test_time = train_mlp_regression(X_train, X_test, X_val, np.ravel(y_train), np.ravel(y_test), np.ravel(y_val), hyperparameters_mlp, std_scaler, SAVE_MLP_MODEL, os.path.join(dataset_ml_models, mlp_model_save_name))
baseline_metrics = calculate_metrics(y_test, mlp_y_pred)
table.add_row(["MLP", f"{train_time:.6f}", f"{test_time:.6f}"]+baseline_metrics)


plt.figure(figure_counter)
plt.gca().set_aspect('equal', adjustable='box')
figure_counter+=1    
plt.scatter(mlp_y_pred, y_test, marker='x', linewidth=2)
plt.xlabel("Predicted Energy (pJ)",fontsize=22,labelpad=10)
plt.ylabel("SPICE Energy (pJ)",fontsize=22,labelpad=10)
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
plt.gca().yaxis.set_major_locator(MultipleLocator(1))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))
plt.gca().tick_params(width=2.5, length=9, which='major',pad=10)  # Set linewidth and length for major ticks
plt.gca().tick_params(width=2, length=6, which='minor')  # Set linewidth and length for minor ticks
plt.plot([0.25,3.75], [0.25,3.75], '--', color='black', linewidth=3.5)
plt.xlim(0.0,4)
plt.ylim(0,4)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
for spine in plt.gca().spines.values():
    spine.set_linewidth(2.5)
plt.tight_layout()

if SAVE_FIGS:
    plt.savefig('../results/mlp_dynamic_energy_model_correlation_plot_'+today+'.pdf', format='pdf')

# -----------------
# Print and write the table to the file
print(table)
write_prettytable(metrics_output_filepath, table)

if PLOT_MATPLOTLIB_FIGS:
    plt.show()

