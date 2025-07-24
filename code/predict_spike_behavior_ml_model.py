import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import seaborn as sns
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
import time
from prettytable import PrettyTable
from datetime import date    
today = date.today().isoformat()

# Helper files with all ML model helpers :)
from predict_ml_model_helpers import *
from config import *

# TODO: Fix seed so that always the same :) (Not sure if we actually need to do this but yeah)


figure_counter = 0

# --------- BEGIN Preprocessing ---------
# Create ML Library
dataset_ml_models = os.path.join('logs', RUN_NAME, "ml_models")

if SAVE_CATBOOST_MODEL or SAVE_MLP_MODEL:
    if not os.path.exists(dataset_ml_models):
        os.makedirs(dataset_ml_models)


run_metrics_filename = 'spike_or_not_model_analysis_' + today + '.csv'
metrics_output_filepath = os.path.join(dataset_ml_models, run_metrics_filename)

# Find full dataset and put into dataframe
dataset_csv_filepath = os.path.join('logs', RUN_NAME, DF_FILENAME)

spike_data_df = pd.read_csv(dataset_csv_filepath)
spike_data_df['Latency'] = spike_data_df['Latency'] * 10**9
spike_data_df['Energy'] = spike_data_df['Energy'] * 10**12

# Only get the things that we need
spike_data_df = spike_data_df[spike_data_df['Event_Type'].isin(['in-out', 'in-no_out'])]
spike_data_df['Spike'] = spike_data_df['Event_Type'].apply(lambda x: 1 if x == 'in-out' else 0)

# Need to actually look at the relative distribution of 0, and 1
total_spikes = spike_data_df['Spike'].sum()
total_points = spike_data_df.shape[0]
print("Number of Spikes: {} / {}".format(total_spikes, total_points))
print("Number of No Spikes: {} / {}".format(total_points-total_spikes, total_points))

# Get the standard scaler in play :O
std_scaler = produce_or_load_common_standard_scalar(spike_data_df, LIST_OF_COLUMNS_X, dataset_ml_models, "Run_Number", TRAIN_TEST_SPLIT, VALIDATION_SPLIT, random_state=42)

# Train Test Split
train_df, test_df, val_df = runwise_train_test_split(spike_data_df, test_size=TRAIN_TEST_SPLIT, val_size=VALIDATION_SPLIT, random_state=42)
X_train = train_df[LIST_OF_COLUMNS_X]
y_train = train_df[["Spike"]]
X_test = test_df[LIST_OF_COLUMNS_X]
y_test = test_df[["Spike"]]
X_val = val_df[LIST_OF_COLUMNS_X]
y_val = val_df[["Spike"]]

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

# -------------------
# Create table to make everything easier to visualize what the hell is going on
table = PrettyTable()
table.field_names = ["Classifier", "Train Time", "Inference Time", "Accuracy", "ROC AUC", "Precision", "Recall", "F1 Score", "MSE"]

# Super Naive Baseline
#Train
start_time = time.time()
train_y_mean = y_train.mode()
end_time = time.time()
train_time = end_time - start_time

# Inference
start_time = time.time()
baseline_vec = np.full_like(y_test, fill_value=train_y_mean)
baseline_vec = np.squeeze(baseline_vec)
end_time = time.time()
test_time = end_time - start_time

baseline_metrics = calculate_binary_classification_metrics(y_test, baseline_vec)
print(baseline_metrics)

table.add_row(["Mean Baseline", f"{train_time:.6f}", f"{test_time:.6f}"]+baseline_metrics)

# ---------------------

# Nearest-Neighbor Interpolator (Table-based method)
print("Training NN Interpolation")
table_y_pred, train_time, test_time = interpolate(X_train, X_test, X_val, y_train, y_test, y_val)
baseline_metrics = calculate_binary_classification_metrics(y_test, table_y_pred)
table.add_row(["NN Interpolation", f"{train_time:.6f}", f"{test_time:.6f}"]+baseline_metrics)

# ----------------------
# Logistic Regression
print("Training Logistic Regression")
logistic_y_pred, train_time, test_time = calculate_logistic_regression(X_train, X_test, X_val, y_train, y_test, y_val, std_scaler)
baseline_metrics = calculate_binary_classification_metrics(y_test, logistic_y_pred)
table.add_row(["Logistic Regression", f"{train_time:.6f}", f"{test_time:.6f}"]+baseline_metrics)

# -------------------------
# CatBoost
# NOTE: Decision Tree Based Models do not need scaled data :O
catboost_params = {
    'iterations': 300,
    'learning_rate': 0.1,
    'depth': 10,
    'l2_leaf_reg': 5,
    'subsample': 0.5,
    'verbose': False,
    'eval_metric':'Logloss'
}

catboost_model_save_name = "catboost_spike_or_not_11_7"
print("Training CatBoost")
cat_y_pred, train_time, test_time = run_catboost_classify(X_train, X_test, X_val, y_train, y_test, y_val, catboost_params, SAVE_CATBOOST_MODEL, os.path.join(dataset_ml_models, catboost_model_save_name),SAVE_CATBOOST_CPP)
baseline_metrics = calculate_binary_classification_metrics(y_test, cat_y_pred)
table.add_row(["CatBoost",  f"{train_time:.6f}", f"{test_time:.6f}"]+baseline_metrics)

#------------------------------------------

# Sklearn MLP
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

print("Training MLP")
mlp_model_save_name = "mlp_spike_or_not_11_8"
mlp_y_pred, train_time, test_time = train_mlp_classify(X_train, X_test, X_val, np.ravel(y_train), np.ravel(y_test), np.ravel(y_val), hyperparameters_mlp, std_scaler, SAVE_MLP_MODEL, os.path.join(dataset_ml_models, mlp_model_save_name))
baseline_metrics = calculate_binary_classification_metrics(y_test, mlp_y_pred)
table.add_row(["MLP", f"{train_time:.6f}", f"{test_time:.6f}"]+baseline_metrics)

# -----------------
# Print and write the table to the file
print(table)
write_prettytable(metrics_output_filepath, table)

if PLOT_MATPLOTLIB_FIGS:
    plt.show()
