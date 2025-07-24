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

# Dynamically load configs :)
import argparse
from dynamic_config_load import inject_config
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Name of the config file (without .py)")
args = parser.parse_args()

inject_config(args.config, globals())


figure_counter = 0

# TODO: Fix seed so that always the same :) (Not sure if we actually need to do this but yeah)

def adc(voltage, v_min=-0.8, v_max=0.8, bits=8):
    """
    Converts an analog voltage to a digital ADC value with saturation.
    """
    levels = 2 ** bits
    step_size = (v_max - v_min) / (levels - 1)

    # Saturate voltage
    voltage = np.clip(voltage, v_min, v_max)


    return np.round((voltage - v_min) / step_size).astype(int)


def dac(adc_value, v_min=-0.8, v_max=0.8, bits=8):
    """
    Converts digital ADC values (scalar or array) back to analog voltages.
    """
    levels = 2 ** bits
    step_size = (v_max - v_min) / (levels - 1)

    adc_value = np.clip(adc_value, 0, levels - 1)  # element-wise clipping

    return v_min + adc_value * step_size


# --------- BEGIN Preprocessing ---------
# Create ML Library
dataset_ml_models = os.path.join('logs', RUN_NAME, "ml_models")
if not os.path.exists(dataset_ml_models):
    os.makedirs(dataset_ml_models)

run_metrics_filename = 'output_model_analysis_' + today + '.csv'
metrics_output_filepath = os.path.join(dataset_ml_models, run_metrics_filename)

# Find full dataset and put into dataframe
dataset_csv_filepath = os.path.join('logs', RUN_NAME, DF_FILENAME)

spike_data_df = pd.read_csv(dataset_csv_filepath)
spike_data_df['Latency'] = spike_data_df['Latency'] * 10**9
spike_data_df['Energy'] = spike_data_df['Energy'] * 10**12

print(spike_data_df['Event_Type'].value_counts())


# Get the standard scaler in play :O
std_scaler = produce_or_load_common_standard_scalar(spike_data_df, LIST_OF_COLUMNS_X_MAC, dataset_ml_models, "Run_Number", TRAIN_TEST_SPLIT, VALIDATION_SPLIT,random_state=42)

# Runwise train test split :)
train_df, test_df, val_df = runwise_train_test_split(spike_data_df, test_size=TRAIN_TEST_SPLIT, val_size=VALIDATION_SPLIT, random_state=42)
X_train = train_df[LIST_OF_COLUMNS_X_MAC]
y_train = train_df[["Output_Value"]]
X_test = test_df[LIST_OF_COLUMNS_X_MAC]
y_test = test_df[["Output_Value"]]
X_val = val_df[LIST_OF_COLUMNS_X_MAC]
y_val = val_df[["Output_Value"]]

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


# Initialize a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1 row, 3 columns

# Plot the "Output Value" histogram
axes[0].hist(spike_data_df["Output_Value"], bins=100)
axes[0].set_title("Output Value Histogram Distribution")

# Plot the "Latency" histogram
axes[1].hist(spike_data_df["Latency"], bins=100)
axes[1].set_title("Latency Histogram Distribution")

# Plot the "Energy" histogram
axes[2].hist(spike_data_df["Energy"], bins=100)
axes[2].set_title("Energy Histogram Distribution")

# Adjust layout for better spacing
plt.tight_layout()

if PLOT_MATPLOTLIB_FIGS:
    plt.show()


# --------- END Preprocessing ---------

# Create table to make everything easier to visualize what the hell is going on
table = PrettyTable()
table.field_names = ["Regressor", "Train Time", "Inference Time", "MSE", "MAE", "MAPE", "R-Squared", "Average Error", "Predicted Output Total", "Real Output Total"]

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
print("Trained Baseline Metric")

# ---------------------

# Nearest-Neighbor Interpolator (Table-based method)
table_y_pred, train_time, test_time = interpolate(X_train, X_test, X_val, y_train, y_test, y_val)
baseline_metrics = calculate_metrics(y_test, table_y_pred)
table.add_row(["NN Interpolation", f"{train_time:.6f}", f"{test_time:.6f}"]+baseline_metrics)
print("Trained NN Interpolation")

# ----------------------

# Print Linear Regression Stuff
ols_y_pred, train_time, test_time = train_linear_regression(X_train, X_test, X_val, y_train, y_test, y_val, std_scaler)
baseline_metrics = calculate_metrics(y_test, ols_y_pred)

table.add_row(["OLS", f"{train_time:.6f}", f"{test_time:.6f}"]+baseline_metrics)
print("Trained OLS")

# ----------------------
# XGBoost
# NOTE: Decision Tree Based Models do not need scaled data :O
hyperparams = {
    'learning_rate': 0.03,
    'max_depth': 10,
    'n_estimators': 500,
    'subsample': 0.7,
    'lambda': 1
}


# -------------------------
# CatBoost
catboost_params = {
    'iterations': 500,
    'learning_rate': 0.1,
    'depth': 10,
    'l2_leaf_reg': 5,
    'subsample': 0.5,
    'eval_metric':'RMSE',
    'verbose': False
}

catboost_model_name = "mac_catboost_output_11_9"
cat_y_pred, train_time, test_time = run_catboost_regression(X_train, X_test, X_val, y_train, y_test, y_val, catboost_params, SAVE_CATBOOST_MODEL, os.path.join(dataset_ml_models, catboost_model_name),SAVE_CATBOOST_CPP, early_stopping=True)
baseline_metrics = calculate_metrics(y_test, cat_y_pred)
table.add_row(["CatBoost", f"{train_time:.6f}", f"{test_time:.6f}"]+baseline_metrics)


# Convert to ADC
V_MAX = 0.8
V_MIN = -0.8

for i in range(4,17):
    y_test_digital = dac(adc(y_test.to_numpy(), v_min=V_MIN, v_max=V_MAX, bits=i), v_min=V_MIN, v_max=V_MAX, bits=i)
    y_pred_digital = dac(adc(cat_y_pred, v_min=V_MIN, v_max=V_MAX, bits=i), v_min=V_MIN, v_max=V_MAX, bits=i)
    baseline_metrics = calculate_metrics(y_test_digital, y_pred_digital)
    table.add_row([f"CatBoost ADC ({i} Bit)", f"{train_time:.6f}", f"{test_time:.6f}"]+baseline_metrics)

print("Trained Catboost")

# ----------------------------

# Assuming X_train_scaled, X_test_scaled, y_train, y_test are your standardized train-test split data

hyperparameters_mlp = {
    'hidden_layer_sizes': (100, 50),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,
    'learning_rate_init': 0.01,
    'tol':1e-5,
    'early_stopping':True,
    'validation_fraction': VALIDATION_SPLIT
}

mlp_model_name = "mac_mlp_output_11_9"
mlp_y_pred, train_time, test_time = train_mlp_regression(X_train, X_test, X_val, np.ravel(y_train), np.ravel(y_test), np.ravel(y_val), hyperparameters_mlp, std_scaler, SAVE_MLP_MODEL, os.path.join(dataset_ml_models, mlp_model_name))
baseline_metrics = calculate_metrics(y_test, mlp_y_pred)
table.add_row(["MLP", f"{train_time:.6f}", f"{test_time:.6f}"]+baseline_metrics)
print("Trained MLP")

# -----------------

# Show correlation plots for MLP and CatBoost

plt.figure(figure_counter)
figure_counter+=1    
plt.scatter(cat_y_pred, y_test, marker='x', linewidth=2)
plt.plot([-1.95,1.95], [-1.95,1.95], '--', color='black', linewidth=3.5)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.xlabel("Predicted Output (V)", fontsize=22)
plt.ylabel("SPICE Output (V)", fontsize=22)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
plt.gca().yaxis.set_major_locator(MultipleLocator(1))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))
plt.gca().tick_params(width=2.5, length=9, which='major',pad=10)  # Set linewidth and length for major ticks
plt.gca().tick_params(width=2, length=6, which='minor')  # Set linewidth and length for minor ticks
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)


for spine in plt.gca().spines.values():
    spine.set_linewidth(2.5)
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig('figure_src/mac_catboost_behavior_model_correlation_plot_'+today+'.svg', format='svg')
    plt.savefig('figure_src/mac_catboost_behavior_model_correlation_plot_'+today+'.pdf', format='pdf')

# -------------------------------
# Print and write the table to the file
print(table)
write_prettytable(metrics_output_filepath, table)

if PLOT_MATPLOTLIB_FIGS:
    plt.show()