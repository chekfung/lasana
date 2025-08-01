import numpy as np
import pandas as pd
import os 
from catboost import CatBoostRegressor, CatBoostClassifier
import time
from predict_ml_model_helpers import write_prettytable
from collections import defaultdict
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from prettytable import PrettyTable
import joblib

def regression_metrics(y_true, y_pred, decimal_places=7):
    # Note: MAPE does not work due to zeroes in the data
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Format the metrics with the specified number of decimal places
    format_string = "{:." + str(decimal_places) + "f}"
    formatted_metrics = [format_string.format(metric) for metric in [mse, mae, r2]]
    return formatted_metrics

def per_circuit_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    return np.array([mse, mae, mape])

def remove_outliers_zscore(column, z_score):
    z_scores = (column - column.mean()) / column.std()
    return column[abs(z_scores) < z_score]  # Keeping values within 3 standard deviations

def classification_metrics(y_true, y_pred, decimal_places=7):
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    format_string = "{:." + str(decimal_places) + "f}"
    formatted_metrics = [format_string.format(metric) for metric in [acc, roc_auc, prec, recall]]
    return formatted_metrics

figure_counter = 0
pd.options.mode.chained_assignment = None  # default='warn'

# --------------------------------------
# Hyperparameters
# Here we are using the same 1000 runs for runtime purposes and not for accuracy purposes. 
RUN_NAME = 'pcm_crossbar_diff_30_run'
MODEL_RUN_NAME = 'pcm_crossbar_diff_30_run'
CSV_NAME = "pcm_crossbar_gain_30_dataset.csv"
CLOCK_PERIOD = 4 * 10**-9
SHOW_FIGS = False
SAVE_FIGS = False
SCALE_MULTIPLIER_OF_CIRCUITS = [0.01, 0.1, 1, 3, 5, 20]

LIST_OF_COLUMNS_X = ["Run_Number", "Input_Total_Time"]
LIST_OF_INPUTS = []

# Input Voltages
NUMBER_OF_INPUTS = 32
for i in range(NUMBER_OF_INPUTS):
    LIST_OF_COLUMNS_X.append(f"v{i}")
    LIST_OF_INPUTS.append(f"v{i}")

CIRCUIT_PARAMS = []
# Input Weights
for i in range(NUMBER_OF_INPUTS):
    CIRCUIT_PARAMS.append(f"weight_{i+1}")
CIRCUIT_PARAMS.append("bias_1")


LIST_OF_COLUMNS_X += CIRCUIT_PARAMS
NUM_CIRCUIT_PARAMS = len(CIRCUIT_PARAMS)

# --------------------------------------
# Find full dataset and put into dataframe
dataset_csv_filepath = os.path.join('logs', RUN_NAME, CSV_NAME)
model_filepath = os.path.join('logs', MODEL_RUN_NAME, 'ml_models')
data_df = pd.read_csv(dataset_csv_filepath)
data_df = data_df.drop(columns=['Run_Name', 'Event_Start_Index', 'Event_End_Index', 'Energy', 'Latency', 'Output_Value'])
data_df = data_df.astype({col: 'float32' for col in data_df.select_dtypes(include='float64').columns})

# -------------------------------------
# Load in Catboost Models
e_static_model = CatBoostRegressor()
e_static_model.load_model(os.path.join(model_filepath, 'pcm_crossbar_catboost_static_energy.cbm'))

e_model = CatBoostRegressor()
e_model.load_model(os.path.join(model_filepath, 'pcm_crossbar_catboost_dynamic_energy.cbm'))

l_model = CatBoostRegressor()
l_model.load_model(os.path.join(model_filepath, 'pcm_crossbar_catboost_latency.cbm'))

output_model = CatBoostRegressor()
output_model.load_model(os.path.join(model_filepath, 'pcm_crossbar_catboost_output.cbm'))

timing_table = PrettyTable()
timing_table.field_names = ["Number of 32-Input Crossbar Rows", "LASANA Runtime"]

for multiplier in SCALE_MULTIPLIER_OF_CIRCUITS:
    # Increase the scale of things :)
    total_number_of_circuits = max(data_df["Run_Number"]) + 1
    scale_list = [data_df]
    run_number_index = total_number_of_circuits

    if multiplier > 1:
        for i in range(1, multiplier):
            # Make a copy
            copied_df = data_df.copy()
            copied_df["Run_Number"] = data_df["Run_Number"] + run_number_index

            scale_list.append(copied_df)

            # Increase run_number_index
            run_number_index += total_number_of_circuits

        data_df_big = pd.concat(scale_list)
    else:
        # multiplier smaller
        total_num_circuits = int(total_number_of_circuits * multiplier)
        data_df_big = data_df[data_df["Run_Number"] < total_num_circuits]

    print("Input Dataset Created")

    num_time_steps = int(data_df["Digital_Time_Step"].max()) + 1

    # Get a specific run and sort by index
    number_of_circuits = data_df_big["Run_Number"].max()+1

    number_of_time_steps_input = []    

    # For each timestep, generate a df
    for time_step_id in range (num_time_steps):
        df = data_df_big[data_df_big["Digital_Time_Step"] == time_step_id]
        inputs = df[(df["Event_Type"] == 'in-out') | (df["Event_Type"] == 'in-no_out')]
        inputs = inputs[LIST_OF_COLUMNS_X]
        number_of_time_steps_input.append(inputs)

    assert(len(number_of_time_steps_input) == num_time_steps)

    # ----

    # Set up database to know circuit parameters 
    circuit_params = np.zeros((number_of_circuits, NUM_CIRCUIT_PARAMS))
    possible_circuit_ids = np.arange(number_of_circuits)

    for i in range(number_of_circuits):
        params_i = data_df_big[data_df_big["Run_Number"]== i].iloc[0]
        circuit_params[i,:] = np.array(params_i[CIRCUIT_PARAMS])
        
    # ----
    # Get rid of data_df_big for memory reasons
    del data_df_big

    # Declare Global Neuron States 
    time_since_last_update = np.ones(number_of_circuits) * -1
    last_output_tracking = np.zeros(number_of_circuits)
    leak_events_energy_per_time_step = defaultdict(list)
    events_energy_per_time_step = defaultdict(list)
    events_latency_per_time_step = defaultdict(list)
    events_output_per_time_step = defaultdict(list)

    run_start_time = time.time()
    for time_step_id in range(num_time_steps):
        # Batch all the inputs together
        events = number_of_time_steps_input[time_step_id]
        circuit_ids = events["Run_Number"]
        
        if time_step_id == num_time_steps-1:
            # Need to process everything left that does not have a spiking event at the end
            ids_not_spiking = np.setdiff1d(possible_circuit_ids, circuit_ids)
            input_voltages = np.zeros((ids_not_spiking.shape[0], NUMBER_OF_INPUTS))
            times_to_include_in_timing_event = ((time_step_id - time_since_last_update[ids_not_spiking]) * CLOCK_PERIOD).reshape(-1,1)
            leak_params = circuit_params[ids_not_spiking]
            last_outputs = last_output_tracking[ids_not_spiking].reshape(-1,1)
            all_params_for_ml = np.concatenate((times_to_include_in_timing_event, last_outputs, input_voltages, leak_params), axis=1)

            # Handle the guys here.
            energy = e_static_model.predict(all_params_for_ml)
            leak_events_energy_per_time_step[time_step_id].append((ids_not_spiking, energy))

        # Now go through all the input events
        if not events.empty:
            time_since_last = time_since_last_update[circuit_ids]
            mask = time_since_last < (time_step_id - 1)
            circuits_with_leak = circuit_ids[mask]

            if circuits_with_leak.shape[0] > 0:
                times = ((time_step_id - time_since_last[mask] - 1) * CLOCK_PERIOD).reshape(-1,1)
                input_voltages = np.zeros((circuits_with_leak.shape[0], NUMBER_OF_INPUTS))
                leak_params = circuit_params[circuits_with_leak]
                last_outputs = last_output_tracking[circuits_with_leak].reshape(-1,1)
                all_params_for_ml = np.concatenate((times, last_outputs, input_voltages, leak_params), axis=1)

                energy = e_static_model.predict(all_params_for_ml)
                leak_events_energy_per_time_step[time_step_id].append((circuits_with_leak, energy))

            params = circuit_params[circuit_ids]
            times = np.ones((circuit_ids.shape[0], 1)) * CLOCK_PERIOD
            inputs = np.array(events[LIST_OF_INPUTS])
            last_outputs = last_output_tracking[circuit_ids].reshape(-1,1)
            all_together = np.concatenate((times, last_outputs, inputs, params), axis=1)

            # behavior
            output = output_model.predict(all_together)
            # Energy and Latency
            energy = e_model.predict(all_together) 
            latency = l_model.predict(all_together)

            last_output_tracking[circuit_ids] = output
            time_since_last_update[circuit_ids] = time_step_id

            # Housekeeping
            events_energy_per_time_step[time_step_id].append((circuit_ids, energy))
            events_latency_per_time_step[time_step_id].append((circuit_ids, latency))
            events_output_per_time_step[time_step_id].append((circuit_ids, output))

    run_end_time = time.time()
    full_time_for_run = run_end_time - run_start_time

    print("{:7f}".format(full_time_for_run))

    timing_table.add_row([total_number_of_circuits, full_time_for_run])

write_prettytable(os.path.join('../results', f"ml_inference_wrapper_pcm_crossbar_diff_30_timing_table.csv"), timing_table)