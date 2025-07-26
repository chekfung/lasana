import os 
from collections import defaultdict
from prettytable import PrettyTable
from datetime import date    
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None  
import joblib
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from stat_helpers import normalize
from predict_ml_model_helpers import write_prettytable
import argparse

# Added Argparse for top script to be able to call with oracle on here to run and produce table III LASANA-O
parser = argparse.ArgumentParser()
parser.add_argument('--oracle', action='store_true', help='Enable oracle mode')
args = parser.parse_args()
ORACLE = args.oracle
print("ORACLE:", ORACLE)

# Initialize some global timer variables that are used
today = date.today().isoformat()
figure_counter = 0

timers = {}
loop_timers = defaultdict(list)

# -------------
# Hyperparameters # FIXME: NEED TO FIX THIS GUY UP :)
RUN_NAME = 'spiking_20000_runs_4_10'
MODEL_RUN_NAME = "larger_weight_range"
CSV_NAME = 'test1.csv'
LIST_OF_COLUMNS_X = ["Run_Number", "Cap_Voltage_At_Input_Start", "Weight", "Input_Total_Time"]
NEURON_PARAMS = ["V_sf", "V_adap", "V_leak", "V_rtr"]
NUM_NEURONS = 20000    
PERIOD = 5 * 10**-9  
LOAD_IN_MLP_MODELS = True

SHOW_FIGS = False
SAVE_FIGS = True   # For just the MSE plot

LIST_OF_COLUMNS_X += NEURON_PARAMS
NUM_NEURON_PARAMS = len(NEURON_PARAMS)

# --------------------------------------

# Global Timing Functions to assess timing.
def start_timer(name="default"):
    timers[name] = time.time()

def stop_timer(name="default"):
    if name in timers:
        elapsed_time = time.time() - timers[name]
        print(f"{name}: {elapsed_time:.6f} seconds")
        del timers[name]
    else:
        print(f"No timer found for '{name}'")

def stop_timer_loop(name="default"):
    if name in timers:
        elapsed_time = time.time() - timers[name]
        del timers[name]

        # Put elapsed time into the loop_timers
        loop_timers[name].append(elapsed_time)
    else:
        print(f"No timer found for '{name}'")

def pretty_print_loop_timers():
    if not loop_timers:
        print("No loop timers recorded.")
        return

    table = PrettyTable()
    table.field_names = ["Timer Name", "Average Time (s)", "Min Time (s)", "Max Time (s)", "Total Time (s)"]

    for name, times in loop_timers.items():
        average_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        total_time = sum(times)

        table.add_row([name, f"{average_time:.6f}", f"{min_time:.6f}", f"{max_time:.6f}", f"{total_time:.6f}"])

    print(table)

# --- Metrics Definitions ---

def mape_metric(y_true,y_pred):
    number_of_guys = y_true.shape[0]

    if number_of_guys == 0:
        mape = 0
    else:
        # Sometimes, there are problems where something will come in with y_true = 0. Deal with it now
        mask = y_true != 0

        guy = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        mape = np.sum(guy) / guy.shape[0]
    return mape

def regression_metrics_no_array(y_true,y_pred, print_mape=False):
    if y_true.shape[0] == 0:
        mse = 0
        mae = 0
        mape = 0
        r2 = 0
    else: 
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        r2 = r2_score(y_true, y_pred)

    if print_mape:
        for i in range(y_true.shape[0]):
            error_percentage = np.abs(y_true[i] - y_pred[i]) / y_true[i]
            print("{:.3f} {:.3f} {:.3f}".format(y_true[i], y_pred[i], error_percentage))

        if y_true.shape[0] != 0:
            # Manually calculate MAPE.
            number_of_values = y_true.shape[0]
            temp = np.abs((y_true - y_pred) / y_true)
            print("Difference: {:.3f}, Number of Guys: {}".format(np.sum(temp), number_of_values))

    return mse, mae, mape, r2

def regression_metrics(y_true, y_pred, decimal_places=7):
    # Note: MAPE does not work due to zeroes in the data
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mape_metric(np.array(y_true), np.array(y_pred)) * 100
    r2 = r2_score(y_true, y_pred)
    average_error = float(np.abs(np.sum(np.array(y_true)) - np.sum(np.array(y_pred))) / np.sum(np.array(y_true)) * 100)

    # Format the metrics with the specified number of decimal places
    format_string = "{:." + str(decimal_places) + "f}"
    formatted_metrics = [format_string.format(metric) for metric in [mse, mae, mape, r2, average_error]]
    return formatted_metrics

def classification_metrics(y_true, y_pred, decimal_places=7):
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1_score = 2 * ((prec * recall) / (prec + recall))
    mse = mean_squared_error(y_true*1.5, np.array(y_pred)*1.5)

    format_string = "{:." + str(decimal_places) + "f}"
    formatted_metrics = [format_string.format(metric) for metric in [acc, roc_auc, prec, recall, f1_score, mse]]
    return formatted_metrics

# --------------------------------------
# Start of Script
# Load in the Test Dataset that will be run

# Find full dataset and put into dataframe
figure_src_directory = os.path.join('results')
dataset_csv_filepath = os.path.join('../data', RUN_NAME, CSV_NAME)
model_filepath = os.path.join('../data', MODEL_RUN_NAME, 'ml_models')

# --------
start_timer("Load CSV and Prepare Data")
spike_data_df = pd.read_csv(dataset_csv_filepath)
spike_data_df = spike_data_df[spike_data_df["Run_Number"] < NUM_NEURONS]

# Convert to Convert to Easy to Understand Things for Output
spike_data_df['Output_Spike'] = spike_data_df['Event_Type'].apply(lambda x: 1 if x == 'in-out' else 0)
spike_data_df['Latency'] = spike_data_df['Latency'] * 10**9
spike_data_df['Energy'] = spike_data_df['Energy'] * 10**12

# Get a specific run and sort by index
spike_data_df = spike_data_df.sort_values(by=['Run_Number', 'Digital_Time_Step', "Event_Type"], ascending=[True, True, False])

neuron_runs = {}
number_of_neurons = spike_data_df["Run_Number"].max()+1

for neuron_id in range(number_of_neurons):
    run_df = spike_data_df[spike_data_df['Run_Number'] == neuron_id]
    data_df = run_df[['Output_Spike', 'Energy', "Latency", "Cap_Voltage_At_Output_End"]]
    neuron_runs[neuron_id] = data_df

assert(number_of_neurons == len(neuron_runs))

stop_timer("Load CSV and Prepare Data")

# --------
start_timer("Generate Time Step DataFrame Input Spikes")

# Break dataframe per time step
number_of_time_steps_leaks = []
number_of_time_steps_spikes = []
number_of_time_steps_answers = {}

num_time_steps = int(spike_data_df["Digital_Time_Step"].max()) + 1
print(f"Num Time Steps: {num_time_steps}")

# For each timestep, generate a df
for time_step_id in range (num_time_steps):
    df = spike_data_df[spike_data_df["Digital_Time_Step"] == time_step_id]

    spikes = df[(df["Event_Type"] == 'in-out') | (df["Event_Type"] == 'in-no_out')]
    spikes = spikes[LIST_OF_COLUMNS_X]
    #spikes_y = spikes[["Energy", "Latency", "Cap_Voltage_At_Output_End"]]
    leak = df[(df["Event_Type"] == 'leak') | (df["Event_Type"] == 'leak-2')]
    leak = leak[LIST_OF_COLUMNS_X]

    # Get per timestep, but sort by having leak events, then sort by run_number
    answers_per_timestep = df[["Run_Number","Event_Type", 'Output_Spike', 'Energy', "Latency", "Cap_Voltage_At_Output_End"]]

    # Rename all in-out to spike, and all leak to leak
    answers_per_timestep['Event_Type'] = answers_per_timestep['Event_Type'].replace({'in-out': 'spike', 'in-no_out': 'spike', 'leak-1': 'leak', 'leak-2': 'leak'})

    answers_per_timestep = answers_per_timestep.sort_values(by=['Event_Type', "Run_Number"], ascending=[True, True])
    number_of_time_steps_answers[time_step_id] = answers_per_timestep

    if time_step_id == 5:
        print("Timestep trace for timestep 5")
        print(answers_per_timestep)

    number_of_time_steps_leaks.append(leak)
    number_of_time_steps_spikes.append(spikes)

assert(len(number_of_time_steps_leaks) == num_time_steps)
assert(len(number_of_time_steps_spikes) == num_time_steps)

stop_timer("Generate Time Step DataFrame Input Spikes")

# ------------
start_timer("Load Models")

# Load in Catboost Models
e_static_model = CatBoostRegressor()
e_static_model.load_model(os.path.join(model_filepath, 'catboost_static_energy_11_7.cbm'))

e_model = CatBoostRegressor()
e_model.load_model(os.path.join(model_filepath, 'catboost_dynamic_energy_11_7.cbm'))

l_model = CatBoostRegressor()
l_model.load_model(os.path.join(model_filepath, 'catboost_latency_11_7.cbm'))

neuron_state_model = CatBoostRegressor()
neuron_state_model.load_model(os.path.join(model_filepath, 'catboost_neuron_state_11_7.cbm'))

spike_or_not_model = CatBoostClassifier()
spike_or_not_model.load_model(os.path.join(model_filepath, 'catboost_spike_or_not_11_7.cbm'))

if LOAD_IN_MLP_MODELS:
    print("Load MLP Models")
    # Load in the std scaler
    random_seed = 42
    std_scaler_name = 'ml_standard_scalar_random_seed_' + str(random_seed) + ".joblib"
    full_path = os.path.join(model_filepath, std_scaler_name)

    # Check if the path exists
    if os.path.exists(full_path):
        std_scaler = joblib.load(full_path)
    else:
        print(f"ERROR: Standard Scaler Path at {full_path} does not exist!")
        exit(1)

    # Note 8_5 is the fast one and accurate one: "mlp_spike_or_not_8_5.joblib"
    e_static_model = joblib.load(os.path.join(model_filepath, 'mlp_static_energy_11_8.joblib'))
    e_model = joblib.load(os.path.join(model_filepath, 'mlp_dynamic_energy_11_8.joblib'))
    l_model = joblib.load(os.path.join(model_filepath, 'mlp_latency_11_8.joblib'))
    neuron_state_model = joblib.load(os.path.join(model_filepath, "mlp_neuron_state_11_8.joblib"))
    spike_or_not_model = joblib.load(os.path.join(model_filepath, "mlp_spike_or_not_11_8.joblib"))

print("ML Models Loaded In")
stop_timer("Load Models")

# Test how fast one of these models would run with some baby inpuct
times = np.zeros((1, 1))
weights = np.zeros((1, 1))
leak_params = np.zeros((1, 4))
neuron_states = np.zeros((1, 1))
all_params_for_ml = np.concatenate((neuron_states, weights, times, leak_params), axis=1)

start_timer("Dynamic Energy Model 1 Sample")
e_model.predict(all_params_for_ml)
stop_timer("Dynamic Energy Model 1 Sample")

start_timer("Latency Model 1 Sample")
l_model.predict(all_params_for_ml)
stop_timer("Latency Model 1 Sample")

start_timer("Static Energy Model 1 Sample")
e_static_model.predict(all_params_for_ml)
stop_timer("Static Energy Model 1 Sample")

# ----------
start_timer("Initialize Algorithm to zeros")

# Set up the guy to know what the neuron parameters are :O
neuron_params = np.zeros((number_of_neurons, NUM_NEURON_PARAMS))
possible_neuron_ids = np.arange(number_of_neurons)

for i in range(number_of_neurons):
    guy = spike_data_df[spike_data_df["Run_Number"]== i].iloc[0]
    neuron_params[i,:] = np.array(guy[NEURON_PARAMS])

# Declare Global Neuron States 
global_neuron_state = np.zeros(number_of_neurons)
time_since_last_update = np.ones(number_of_neurons) * -1

leak_events_neuron_state_per_time_step = defaultdict(list)
leak_events_energy_per_time_step = defaultdict(list)

spike_events_neuron_state_per_time_step = defaultdict(list)
spike_events_energy_per_time_step = defaultdict(list)
spike_events_latency_per_time_step = defaultdict(list)
spike_events_spike_or_not_per_time_step = defaultdict(list)

stop_timer("Initialize Algorithm to zeros")

# ---------------
start_timer("Algorithm Run")
for time_step_id in range(num_time_steps):
    # Batch all the input spikes together
    spike_events = number_of_time_steps_spikes[time_step_id]
    spike_neuron_ids = spike_events["Run_Number"]
    
    if time_step_id == num_time_steps-1:
        # Need to process everything left that does not have a spiking event at the end
        ids_not_spiking = np.setdiff1d(possible_neuron_ids, spike_neuron_ids)

        weights = np.zeros((ids_not_spiking.shape[0], 1))
        times_to_include_in_timing_event = ((time_step_id - time_since_last_update[ids_not_spiking]) * PERIOD).reshape(-1,1)
        neuron_states = global_neuron_state[ids_not_spiking].reshape(-1,1)
        leak_params = neuron_params[ids_not_spiking]
        all_params_for_ml = np.concatenate((neuron_states, weights, times_to_include_in_timing_event, leak_params), axis=1)

        if LOAD_IN_MLP_MODELS:
            all_params_for_ml = std_scaler.transform(all_params_for_ml)

        # Handle the guys here.
        next_neuron_state = neuron_state_model.predict(all_params_for_ml)
        energy = e_static_model.predict(all_params_for_ml)

        global_neuron_state[ids_not_spiking] = next_neuron_state

        leak_events_neuron_state_per_time_step[time_step_id].append((ids_not_spiking, next_neuron_state))
        leak_events_energy_per_time_step[time_step_id].append((ids_not_spiking, energy))

    # Now go through all the input events
    if not spike_events.empty:
        time_since_last = time_since_last_update[spike_neuron_ids]
        mask = time_since_last < (time_step_id - 1)
        neurons_with_leak = spike_neuron_ids[mask]

        start_timer("Leak Event Processing")
        if neurons_with_leak.shape[0] > 0:
            start_timer("Leak Event Create Input Batch")
            times = ((time_step_id - time_since_last[mask] - 1) * PERIOD).reshape(-1,1)
            weights = np.zeros((neurons_with_leak.shape[0], 1))
            leak_params = neuron_params[neurons_with_leak]

            if not ORACLE:
                neuron_states = global_neuron_state[neurons_with_leak].reshape(-1,1)
            else: 
                # Experiment to change neuron state usage
                if time_step_id == num_time_steps-1:
                    neuron_states = global_neuron_state[neurons_with_leak].reshape(-1,1)
                else:
                    neuron_states = np.array(number_of_time_steps_leaks[time_step_id]["Cap_Voltage_At_Input_Start"]).reshape(-1,1)
                # End Experiment

            all_params_for_ml = np.concatenate((neuron_states, weights, times, leak_params), axis=1)
            stop_timer_loop("Leak Event Create Input Batch")

            # ----
            start_timer("Leak Event ML Models")
            if LOAD_IN_MLP_MODELS:
                all_params_for_ml = std_scaler.transform(all_params_for_ml)

            start_timer("Leak Neuron State Inference")
            next_neuron_state = neuron_state_model.predict(all_params_for_ml)
            stop_timer_loop("Leak Neuron State Inference")

            start_timer("Leak Static Energy Inference")
            energy = e_static_model.predict(all_params_for_ml)
            stop_timer_loop("Leak Static Energy Inference")

            stop_timer_loop("Leak Event ML Models")

            # ----

            start_timer("Leak Event House Keeping")
            global_neuron_state[neurons_with_leak] = next_neuron_state
            leak_events_neuron_state_per_time_step[time_step_id].append((neurons_with_leak, next_neuron_state))
            leak_events_energy_per_time_step[time_step_id].append((neurons_with_leak, energy))
            stop_timer_loop("Leak Event House Keeping")

        stop_timer_loop("Leak Event Processing")

        # --------

        start_timer("Spike Event Processing")

        start_timer("Spike Event Input Batch Creation")
        params = neuron_params[spike_neuron_ids]
        times = np.ones((spike_neuron_ids.shape[0], 1)) * PERIOD
        weights = np.array(spike_events["Weight"]).reshape(-1,1)

        if not ORACLE:
            neuron_states = global_neuron_state[spike_neuron_ids].reshape(-1,1)
        else:
            # Experiment to change neuron state usage
            neuron_states = np.array(spike_events["Cap_Voltage_At_Input_Start"]).reshape(-1,1)

        all_together = np.concatenate((neuron_states, weights, times, params), axis=1)
        stop_timer_loop("Spike Event Input Batch Creation")

        # --------

        start_timer("Spike ML Model Inference")

        if LOAD_IN_MLP_MODELS:
            all_together = std_scaler.transform(all_together)

        start_timer("Spike Static Energy Inference")
        static_energy = e_static_model.predict(all_together)
        stop_timer_loop("Spike Static Energy Inference")

        # Spike or Not?
        start_timer("Spike Behavior Inference")
        spike_or_not = spike_or_not_model.predict(all_together)
        stop_timer_loop("Spike Behavior Inference")

        # Calculate Next State
        start_timer("Spike Neuron State Inference")
        next_neuron_state = neuron_state_model.predict(all_together)
        stop_timer_loop("Spike Neuron State Inference")

        # Energy and Latency
        start_timer("Spike Dynamic Energy Inference")
        energy = e_model.predict(all_together) * spike_or_not
        stop_timer_loop("Spike Dynamic Energy Inference")

        start_timer("Spike Add Spike Energies Together")
        energy = energy + (static_energy * np.logical_not(spike_or_not))
        stop_timer_loop("Spike Add Spike Energies Together")

        start_timer("Spike Latency Inference")
        latency = l_model.predict(all_together) * spike_or_not
        stop_timer_loop("Spike Latency Inference")

        stop_timer_loop("Spike ML Model Inference")

        # --------

        start_timer("Spike House Keeping")

        # Update Global State
        global_neuron_state[spike_neuron_ids] = next_neuron_state
        time_since_last_update[spike_neuron_ids] = time_step_id

        # Housekeeping
        spike_events_neuron_state_per_time_step[time_step_id].append((spike_neuron_ids, next_neuron_state))
        spike_events_energy_per_time_step[time_step_id].append((spike_neuron_ids, energy))
        spike_events_latency_per_time_step[time_step_id].append((spike_neuron_ids, latency))
        spike_events_spike_or_not_per_time_step[time_step_id].append((spike_neuron_ids, spike_or_not))

        stop_timer_loop("Spike House Keeping")

        # -----

        stop_timer_loop("Spike Event Processing")

stop_timer("Algorithm Run")

# --------------
# Do Post Processing :)
# Print out all the timings aggregated
pretty_print_loop_timers()

# Post Processing to make sense of everything
energy_per_neuron_event = defaultdict(list)
latency_per_neuron_event = defaultdict(list)
neuron_state_per_neuron_event = defaultdict(list)
spike_or_not_per_neuron_event = defaultdict(list)
event_timeline = defaultdict(list)

per_timestep_guy = defaultdict(list)

for time_step_id in range(num_time_steps):
    # Handle Leak Event First
    leak_time = leak_events_neuron_state_per_time_step[time_step_id]
    energy_guys = leak_events_energy_per_time_step[time_step_id]

    if len(leak_time) > 0:
        # Combine all arrays
        combined_arrays = [np.concatenate(arrays) for arrays in zip(*leak_time)]
        energy_combined = [np.concatenate(arrays) for arrays in zip(*energy_guys)]

        # Combine with energy, and then sort by the neuron ID
        combined_arrays.append(energy_combined[1])
        stacked = np.stack(combined_arrays, axis=-1)
        sorted_combined_2d_array = stacked[np.argsort(stacked[:,0])]

        for i in range(sorted_combined_2d_array.shape[0]):
            leak_id = sorted_combined_2d_array[i,0]
            next_neuron_state = sorted_combined_2d_array[i,1]
            energy = sorted_combined_2d_array[i,2]

            energy_per_neuron_event[leak_id].append(energy)
            latency_per_neuron_event[leak_id].append(0)
            neuron_state_per_neuron_event[leak_id].append(next_neuron_state)
            spike_or_not_per_neuron_event[leak_id].append(0)
            event_timeline[leak_id].append("leak")

            # Put the leak timestep guy in here
            timestep_leak_info = (leak_id, "leak", 0, energy, 0, next_neuron_state)
            per_timestep_guy[time_step_id].append(timestep_leak_info)

    # Handle Spike Events
    spike_time = spike_events_neuron_state_per_time_step[time_step_id]

    if len(spike_time) > 0:
        spike_ids, neuron_state_s = spike_time[0]
        spike_ids, energy = spike_events_energy_per_time_step[time_step_id][0]
        spike_ids, latency = spike_events_latency_per_time_step[time_step_id][0]
        spike_ids, spike_or_not = spike_events_spike_or_not_per_time_step[time_step_id][0]
        spike_ids = np.array(spike_ids)

        for i in range (spike_ids.shape[0]):
            spike_id = spike_ids[i]
            energy_per_neuron_event[spike_id].append(energy[i])
            latency_per_neuron_event[spike_id].append(latency[i])
            neuron_state_per_neuron_event[spike_id].append(neuron_state_s[i])
            spike_or_not_per_neuron_event[spike_id].append(spike_or_not[i])
            event_timeline[spike_id].append('spike')

            # Put spike timestep guy in here
            timestep_spike_info = (spike_id, "spike", spike_or_not[i], energy[i], latency[i], neuron_state_s[i])
            per_timestep_guy[time_step_id].append(timestep_spike_info)

# For each timestep guy, convert into a dataframe
predicted_timestep_dfs = {}

# Convert each list of tuples into a DataFrame
for key, value in per_timestep_guy.items():
    df = pd.DataFrame(value, columns=["Run_Number","Event_Type", 'Output_Spike', 'Energy', "Latency", "Cap_Voltage_At_Output_End"])
    predicted_timestep_dfs[key] = df

# Calculate Metrics
predicted_energy = []
predicted_latency = []
predicted_neuron_state = []
predicted_spike_or_not = []

# Note: Output in this order of columns ['Output_Spike', 'Energy', "Latency", "Cap_Voltage_At_Output_End"]
spice_results = []

# Stack everything together
# Get per neuron energy :)
neuron_energies = np.zeros(number_of_neurons)

for neuron_id in range(number_of_neurons):
    # Predicted
    neuron_energies[neuron_id] = np.sum(energy_per_neuron_event[neuron_id])
    predicted_energy = predicted_energy + energy_per_neuron_event[neuron_id]
    predicted_latency = predicted_latency + latency_per_neuron_event[neuron_id]
    predicted_neuron_state = predicted_neuron_state + neuron_state_per_neuron_event[neuron_id]
    predicted_spike_or_not = predicted_spike_or_not + spike_or_not_per_neuron_event[neuron_id]

    real_results = np.array(neuron_runs[neuron_id])
    spice_results.append(real_results)

total_neuron_energies = np.sum(predicted_energy)

# Vstack everything for spice_results into one big array
spice_results_together = np.vstack(spice_results)

# Get statistics per timestep :)
latency_statistics = np.zeros((num_time_steps, 4))
energy_statistics = np.zeros((num_time_steps, 4))
static_energy_statistics = np.zeros((num_time_steps, 4))
neuron_state_statistics = np.zeros((num_time_steps, 4))

for time_step_id in range(num_time_steps):

    if time_step_id in predicted_timestep_dfs:
        # Get each of real and predicted
        predicted_guys = predicted_timestep_dfs[time_step_id]
        real_guys = number_of_time_steps_answers[time_step_id]

        # Determine all the times that we spike correctly by making a bitmask
        pred_spikes = np.array(predicted_guys["Output_Spike"]) 
        real_spikes = np.array(real_guys["Output_Spike"])

        bit_mask = (real_spikes == 1)
        static_bit_mask = (real_spikes == 0) 

        # Calculate Regression Metrics
        static_energy_statistics[time_step_id, :] = regression_metrics_no_array(np.array(real_guys["Energy"])[static_bit_mask], np.array(predicted_guys["Energy"])[static_bit_mask])
        energy_statistics[time_step_id, :] = regression_metrics_no_array(np.array(real_guys["Energy"])[bit_mask], np.array(predicted_guys["Energy"])[bit_mask])
        latency_statistics[time_step_id, :] = regression_metrics_no_array(np.array(real_guys["Latency"][bit_mask]), np.array(predicted_guys["Latency"])[bit_mask])
        neuron_state_statistics[time_step_id, :] = regression_metrics_no_array(np.array(real_guys["Cap_Voltage_At_Output_End"]), np.array(predicted_guys["Cap_Voltage_At_Output_End"]))

    else:
        static_energy_statistics[time_step_id, :] = (0,0,0,0)
        energy_statistics[time_step_id, :] = (0,0,0,0)
        latency_statistics[time_step_id, :] = (0,0,0,0)
        neuron_state_statistics[time_step_id, :] = (0,0,0,0)

# -------------
# Figure save make directory
if SAVE_FIGS:
    # Make necessary directories by using the most nested directory
    if not os.path.exists(figure_src_directory):
        os.makedirs(figure_src_directory)
        print(f"Directory '{figure_src_directory}' created successfully.")

# Create figures for per-timestep guys
time_vector = np.linspace(0, 500, 100)
plt.figure(figure_counter, figsize=(5.5,2))
figure_counter+=1
plt.plot(time_vector,normalize(energy_statistics[:, 0]), linewidth=1.75)
plt.plot(time_vector,normalize(static_energy_statistics[:, 0]), linewidth=1.75, linestyle='--')
plt.plot(time_vector,normalize(latency_statistics[:, 0]), linewidth=1.75, linestyle='-.')
plt.plot(time_vector,normalize(neuron_state_statistics[:, 0]), linewidth=1.75, linestyle=':', color='black')
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel("Time Step (ns)", fontsize=9)
plt.ylabel("Normalized MSE", fontsize=9)
#plt.title("MSE Vs. Time")
plt.legend(['Dynamic Energy', "Static Energy","Latency", "State"],ncol=4, prop={'size': 8},loc='lower center', framealpha=0.5)
plt.tight_layout()
if SAVE_FIGS:
    figure_name = "MSE_over_time" + today
    plt.savefig(os.path.join(figure_src_directory, figure_name+'.pdf'), format='pdf')

# Get Metrics
# For regressors
regressor_table = PrettyTable()
regressor_table.field_names = ["Value", "MSE", "MAE", "MAPE", "R-Squared", "Average Error"]

# Get bitmask of just the spikes for energy and latency :)
bit_mask = (np.array(spice_results_together[:,0]) ==1)
static_bit_mask = (np.array(spice_results_together[:,0]) ==0)
latency_bit_mask = (np.array(spice_results_together[:,0]) == np.array(predicted_spike_or_not)) & (np.array(spice_results_together[:,0]) ==1)

metrics = regression_metrics(np.array(spice_results_together[:, 1])[static_bit_mask], np.array(predicted_energy)[static_bit_mask])
regressor_table.add_row(["Static Energy"]+metrics)
metrics = regression_metrics(np.array(spice_results_together[:, 1])[bit_mask], np.array(predicted_energy)[bit_mask])
regressor_table.add_row(["Energy"]+metrics)
metrics = regression_metrics(np.array(spice_results_together[:, 2])[bit_mask], np.array(predicted_latency)[bit_mask])
regressor_table.add_row(["Latency"]+metrics)
metrics = regression_metrics(spice_results_together[:, 3], predicted_neuron_state)
regressor_table.add_row(["Neuron State (V(C_mem))"]+metrics)
print(regressor_table)

if ORACLE:
    tag = "oracle"
else:
    tag = "predicted"

write_prettytable(os.path.join('results', f"batch_transient_analysis_regressor_table_{tag}.csv"), regressor_table)

# For classifiers
classifier_table = PrettyTable()
classifier_table.field_names = ["Value", "Acc", "ROC AUC", "Precision", "Recall", "F1", "MSE"]

metrics = classification_metrics(spice_results_together[:, 0], predicted_spike_or_not)
classifier_table.add_row(["Output Spike or Not"]+metrics)

print(classifier_table)
write_prettytable(os.path.join('results', f"batch_transient_analysis_classifier_table_{tag}.csv"), classifier_table)

# ----
print(f"Total Predicted Energy: {total_neuron_energies}")
total_real_energies = np.sum(spice_results_together[:,1])
print(f"Total SPICE Energy: {total_real_energies}")
average_error = np.abs(total_real_energies - total_neuron_energies) /total_real_energies * 100
print(f"Average Error: {average_error:2f}")

if SHOW_FIGS:
    plt.show(block=False)
    plt.pause(0.001) # Pause for interval seconds.
    input("hit[enter] to end.")
    plt.close('all') # all open plots are correctly closed after each run