from prettytable import PrettyTable
import pandas as pd
import os
from collections import defaultdict
from predict_ml_model_helpers import *

'''
Helper script that runs after the training and testing of all of the ML LASANA models which recreates the tables that
we see in the paper.
'''

# Helper to extract times safely
def get_times(df, label):
    if label in df.index:
        row = df.loc[label]
        return float(row["Train Time"]), float(row["Inference Time"])
    else:
        return 0.0, 0.0

# Read in the files
PCM_FOLDER_CSV_FILES = '../data/pcm_crossbar_diff_30_run/ml_models'
SPIKING_NEURON_FOLDER_CSV_FILES = '../data/spiking_neuron_run/ml_models'

# Get relevant CSVs
pcm_latency = pd.read_csv(os.path.join(PCM_FOLDER_CSV_FILES, 'latency_model_analysis.csv'), index_col="Regressor")
pcm_dynamic_energy = pd.read_csv(os.path.join(PCM_FOLDER_CSV_FILES, 'dynamic_energy_model_analysis.csv'), index_col="Regressor")
pcm_static_energy = pd.read_csv(os.path.join(PCM_FOLDER_CSV_FILES, 'static_energy_model_analysis.csv'), index_col="Regressor")
pcm_output = pd.read_csv(os.path.join(PCM_FOLDER_CSV_FILES, 'output_model_analysis.csv'), index_col="Regressor")
pcm_tables = [pcm_latency, pcm_dynamic_energy, pcm_static_energy, pcm_output]

# Get relevant CSVs
lif_latency = pd.read_csv(os.path.join(SPIKING_NEURON_FOLDER_CSV_FILES, 'latency_model_analysis.csv'), index_col="Regressor")
lif_dynamic_energy = pd.read_csv(os.path.join(SPIKING_NEURON_FOLDER_CSV_FILES, 'dynamic_energy_model_analysis.csv'), index_col="Regressor")
lif_static_energy = pd.read_csv(os.path.join(SPIKING_NEURON_FOLDER_CSV_FILES, 'static_energy_model_analysis.csv'), index_col="Regressor")
lif_state = pd.read_csv(os.path.join(SPIKING_NEURON_FOLDER_CSV_FILES, 'neuron_state_model_analysis.csv'), index_col="Regressor")
lif_output = pd.read_csv(os.path.join(SPIKING_NEURON_FOLDER_CSV_FILES, 'spike_or_not_model_analysis.csv'), index_col="Classifier")
lif_tables = [lif_latency, lif_dynamic_energy, lif_static_energy, lif_state, lif_output]

# Create Timing Table
labels = ["Mean Baseline", "NN Interpolation", "OLS", "CatBoost", "MLP"]
columns = ["Model", "PCM Crossbar Train (s)", "PCM Crossbar Test (s)", "LIF Train (s)", "LIF Test (s)"]

# Get timing information
timing_table = defaultdict(float)

for df in pcm_tables:
    for label in labels:
        train_time, test_time = get_times(df, label)

        timing_table[label+"_pcm_train"] += train_time
        timing_table[label+"_pcm_test"] += test_time

for df in lif_tables:
    for label in labels:
        train_time, test_time = get_times(df, label)

        timing_table[label+"_lif_train"] += train_time
        timing_table[label+"_lif_test"] += test_time

# Create table
table_i = PrettyTable()
table_i.field_names = columns

# Add rows
table_i.add_row(["Mean", round(timing_table["Mean Baseline_pcm_train"], 3), round(timing_table["Mean Baseline_pcm_test"], 4),
               round(timing_table["Mean Baseline_lif_train"], 3), round(timing_table["Mean Baseline_lif_test"], 4)])

table_i.add_row(["Table", round(timing_table["NN Interpolation_pcm_train"], 3), round(timing_table["NN Interpolation_pcm_test"], 4),
               round(timing_table["NN Interpolation_lif_train"], 3), round(timing_table["NN Interpolation_lif_test"], 4)])

table_i.add_row(["OLS", round(timing_table["OLS_pcm_train"], 3), round(timing_table["OLS_pcm_test"], 4),
               round(timing_table["OLS_lif_train"], 3), round(timing_table["OLS_lif_test"], 4)])

table_i.add_row(["CatBoost (d=10)", round(timing_table["CatBoost_pcm_train"], 3), round(timing_table["CatBoost_pcm_test"], 4),
               round(timing_table["CatBoost_lif_train"], 3), round(timing_table["CatBoost_lif_test"], 4)])

table_i.add_row(["MLP (100,50)", round(timing_table["MLP_pcm_train"], 3), round(timing_table["MLP_pcm_test"], 4),
               round(timing_table["MLP_lif_train"], 3), round(timing_table["MLP_lif_test"], 4)])

# Print table
print(table_i)

# Save pretty table
write_prettytable("../results/table_i.csv", table_i)

# ---------

# Create Table II
def create_table_ii_row(label):
    data = [
        round(pcm_latency.loc[label]["MSE"] * 10**6, 1), 
        round(pcm_latency.loc[label]["MAPE"], 3),
        round(pcm_dynamic_energy.loc[label]["MSE"] * 10**6, 1),
        round(pcm_dynamic_energy.loc[label]["MAPE"], 2),
        round(pcm_static_energy.loc[label]["MSE"] * 10**6, 1),
        round(pcm_output.loc[label]["MSE"], 4),
        round(lif_latency.loc[label]["MSE"], 4),
        round(lif_latency.loc[label]["MAPE"], 2),
        round(lif_dynamic_energy.loc[label]["MSE"], 3),
        round(lif_dynamic_energy.loc[label]["MAPE"], 2),
        round(lif_static_energy.loc[label]["MSE"], 4),
        round(lif_state.loc[label]["MSE"], 4),
        round(lif_output.loc[label]["MSE"], 4),
        round(lif_output.loc[label]["Accuracy"]*100, 2)
    ]
    return data

table_ii = PrettyTable()

# Redefine columns with unique, clear names
columns = [
    "Model",
    "PCM M_L MSE(ps²)", "PCM M_L MAPE(%)","PCM M_ED MSE(fJ²)", "PCM M_ED MAPE(%)","PCM M_ES MSE(fJ²)", "PCM M_O MSE(V²)",
    "LIF M_L MSE(ns²)", "LIF M_L MAPE(%)","LIF M_ED MSE(pJ²)", "LIF M_ED MAPE(%)","LIF M_ES MSE(pJ²)", "LIF M_V MSE(V²)", "LIF M_O MSE(V²)", "LIF M_O Acc. (%)"
]

table_ii.field_names = columns

# Data rows
data_rows = [
    ["Mean"] + create_table_ii_row("Mean Baseline"),
    ["Table"] + create_table_ii_row("NN Interpolation"),
    ["Linear"] + create_table_ii_row("OLS"),
    ["CatBoost (d=10)"] + create_table_ii_row("CatBoost"),
    ["MLP (100,50)"] + create_table_ii_row("MLP"),
]

for row in data_rows:
    table_ii.add_row(row)

print(table_ii)

# Save table II away
write_prettytable("../results/table_ii.csv", table_ii)

# ---------

# Create Table III
oracle_classifier_table = pd.read_csv('../data/ml_inference_wrapper_intermediate_results/ml_inference_wrapper_classifier_oracle.csv', index_col="Value")
oracle_regressor_table = pd.read_csv('../data/ml_inference_wrapper_intermediate_results/ml_inference_wrapper_regressor_oracle.csv', index_col="Value")
predicted_classifier_table = pd.read_csv('../data/ml_inference_wrapper_intermediate_results/ml_inference_wrapper_classifier_predicted.csv', index_col="Value")
predicted_regressor_table = pd.read_csv('../data/ml_inference_wrapper_intermediate_results/ml_inference_wrapper_regressor_predicted.csv', index_col="Value")

table_iii = PrettyTable()

# Set column headers
table_iii.field_names = [
    "Model",
    "MSE M_L (ns²)", "MAPE M_L (%)",
    "MSE M_ED (pJ²)", "MAPE M_ED (%)",
    "MSE M_ES (pJ²)",
    "MSE M_V (V²)",
    "MSE M_O (V²)",
    "Acc. M_O  (%)"
]
row1 = [
    "LASANA-O",
    round(oracle_regressor_table.loc["Latency"]["MSE"], 4),
    round(oracle_regressor_table.loc["Latency"]["MAPE"], 2),
    round(oracle_regressor_table.loc["Energy"]["MSE"], 3),
    round(oracle_regressor_table.loc["Energy"]["MAPE"], 2),
    round(oracle_regressor_table.loc["Static Energy"]["MSE"], 4),
    round(oracle_regressor_table.loc["Neuron State (V(C_mem))"]["MSE"], 4),
    round(oracle_classifier_table.loc["Output Spike or Not"]["MSE"], 4),
    round(oracle_classifier_table.loc["Output Spike or Not"]["Acc"], 2),
]

row2 = [
    "LASANA-P",
    round(predicted_regressor_table.loc["Latency"]["MSE"], 4),
    round(predicted_regressor_table.loc["Latency"]["MAPE"], 2),
    round(predicted_regressor_table.loc["Energy"]["MSE"], 3),
    round(predicted_regressor_table.loc["Energy"]["MAPE"], 2),
    round(predicted_regressor_table.loc["Static Energy"]["MSE"], 4),
    round(predicted_regressor_table.loc["Neuron State (V(C_mem))"]["MSE"], 4),
    round(predicted_classifier_table.loc["Output Spike or Not"]["MSE"], 4),
    round(predicted_classifier_table.loc["Output Spike or Not"]["Acc"], 2),
]

# Add rows
table_iii.add_row(row1)
table_iii.add_row(row2)

print(table_iii)

# Send to Results
write_prettytable("../results/table_iii.csv", table_iii)