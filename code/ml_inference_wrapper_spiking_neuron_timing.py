import numpy as np
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import os 
from catboost import CatBoostRegressor, CatBoostClassifier
import time
from collections import defaultdict
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from prettytable import PrettyTable
from predict_ml_model_helpers import write_prettytable
from scipy.stats import zscore
from datetime import date    
today = date.today().isoformat()
import joblib

figure_counter = 0
pd.options.mode.chained_assignment = None  # default='warn'

# --------------------------------------
# Hyperparameters 
RUN_NAME = 'spiking_20000_runs'
MODEL_RUN_NAME = "spiking_neuron_run"
CSV_NAME = 'spiking_neuron_dataset.csv'
LIST_OF_COLUMNS_X = ["Run_Number", "Cap_Voltage_At_Input_Start", "Weight", "Input_Total_Time"]
NEURON_PARAMS = ["V_sf", "V_adap", "V_leak", "V_rtr"]
LIST_OF_COLUMNS_X += NEURON_PARAMS
NUM_NEURON_PARAMS = len(NEURON_PARAMS)
period = 5 * 10**-9
LOAD_IN_MLP_MODELS = True
num_neurons = [10, 100, 1000, 3000, 5000, 20000]


# --------------------------------------

# Load dataset that we will use beforehand
dataset_csv_filepath = os.path.join('../data', RUN_NAME, CSV_NAME)
model_filepath = os.path.join('../data', MODEL_RUN_NAME, 'ml_models')
spike_data_df = pd.read_csv(dataset_csv_filepath)

# Fill in NA
spike_data_df.fillna(0, inplace=True)

# Convert to Convert to Easy to Understand Things for Output
spike_data_df['Output_Spike'] = spike_data_df['Event_Type'].apply(lambda x: 1 if x == 'in-out' else 0)
spike_data_df['Latency'] = spike_data_df['Latency'] * 10**9
spike_data_df['Energy'] = spike_data_df['Energy'] * 10**12

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
    e_static_model = joblib.load(os.path.join(model_filepath, 'spiking_neuron_mlp_static_energy.joblib'))
    e_model = joblib.load(os.path.join(model_filepath, 'spiking_neuron_mlp_dynamic_energy.joblib'))
    l_model = joblib.load(os.path.join(model_filepath, 'spiking_neuron_mlp_latency.joblib'))
    neuron_state_model = joblib.load(os.path.join(model_filepath, "spiking_neuron_mlp_neuron_state.joblib"))
    spike_or_not_model = joblib.load(os.path.join(model_filepath, "spiking_neuron_mlp_spike_or_not.joblib"))
else:
    # Load in Catboost Models
    e_static_model = CatBoostRegressor()
    e_static_model.load_model(os.path.join(model_filepath, 'spiking_neuron_catboost_static_energy.cbm'))

    e_model = CatBoostRegressor()
    e_model.load_model(os.path.join(model_filepath, 'spiking_neuron_catboost_dynamic_energy.cbm'))

    l_model = CatBoostRegressor()
    l_model.load_model(os.path.join(model_filepath, 'spiking_neuron_catboost_latency.cbm'))

    neuron_state_model = CatBoostRegressor()
    neuron_state_model.load_model(os.path.join(model_filepath, 'spiking_neuron_catboost_neuron_state.cbm'))

    spike_or_not_model = CatBoostClassifier()
    spike_or_not_model.load_model(os.path.join(model_filepath, 'spiking_neuron_catboost_spike_or_not.cbm'))

# --------------------------------------------------
# Create pretty table to export easily later
timing_table = PrettyTable()
timing_table.field_names = ["Number of Neurons", "LASANA Runtime"]


# Start of all the runs
for neurons in num_neurons:
    # Only use so many neurons
    spike_data_df_run = spike_data_df[spike_data_df["Run_Number"] < neurons]

    # Get a specific run and sort by index
    spike_data_df_run = spike_data_df_run.sort_values(by=['Run_Number', 'Digital_Time_Step', "Event_Type"], ascending=[True, True, False])

    # Break dataframe per time step
    number_of_time_steps_leaks = []
    number_of_time_steps_spikes = []

    num_time_steps = int(spike_data_df_run["Digital_Time_Step"].max()) + 1

    # For each timestep, generate a df
    for time_step_id in range (num_time_steps):
        df = spike_data_df_run[spike_data_df_run["Digital_Time_Step"] == time_step_id]

        spikes = df[(df["Event_Type"] == 'in-out') | (df["Event_Type"] == 'in-no_out')]
        spikes = spikes[LIST_OF_COLUMNS_X]
        #spikes_y = spikes[["Energy", "Latency", "Cap_Voltage_At_Output_End"]]
        leak = df[(df["Event_Type"] == 'leak') | (df["Event_Type"] == 'leak-2')]
        leak = leak[LIST_OF_COLUMNS_X]

        number_of_time_steps_leaks.append(leak)
        number_of_time_steps_spikes.append(spikes)

    assert(len(number_of_time_steps_leaks) == num_time_steps)
    assert(len(number_of_time_steps_spikes) == num_time_steps)

    # Set up the guy to know what the neuron parameters are :O
    neuron_params = np.zeros((neurons, NUM_NEURON_PARAMS))
    possible_neuron_ids = np.arange(neurons)

    for i in range(neurons):
        guy = spike_data_df[spike_data_df["Run_Number"]== i].iloc[0]
        neuron_params[i,:] = np.array(guy[NEURON_PARAMS])

    # Declare Global Neuron States 
    global_neuron_state = np.zeros(neurons)
    time_since_last_update = np.ones(neurons) * -1

    leak_events_neuron_state_per_time_step = defaultdict(list)
    leak_events_energy_per_time_step = defaultdict(list)

    spike_events_neuron_state_per_time_step = defaultdict(list)
    spike_events_energy_per_time_step = defaultdict(list)
    spike_events_latency_per_time_step = defaultdict(list)
    spike_events_spike_or_not_per_time_step = defaultdict(list)

    run_start_time = time.time()
    for time_step_id in range(num_time_steps):
        # Batch all the input spikes together
        spike_events = number_of_time_steps_spikes[time_step_id]
        spike_neuron_ids = spike_events["Run_Number"]
        
        if time_step_id == num_time_steps-1:
            # Need to process everything left that does not have a spiking event at the end
            ids_not_spiking = np.setdiff1d(possible_neuron_ids, spike_neuron_ids)

            weights = np.zeros((ids_not_spiking.shape[0], 1))
            times_to_include_in_timing_event = ((time_step_id - time_since_last_update[ids_not_spiking]) * period).reshape(-1,1)
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

        # Now go through all the spike events
        if not spike_events.empty:
            time_since_last = time_since_last_update[spike_neuron_ids]
            mask = time_since_last < (time_step_id - 1)
            neurons_with_leak = spike_neuron_ids[mask]

            if neurons_with_leak.shape[0] > 0:
                times = ((time_step_id - time_since_last[mask] - 1) * period).reshape(-1,1)
                weights = np.zeros((neurons_with_leak.shape[0], 1))
                neuron_states = global_neuron_state[neurons_with_leak].reshape(-1,1)

                leak_params = neuron_params[neurons_with_leak]
                all_params_for_ml = np.concatenate((neuron_states, weights, times, leak_params), axis=1)

                if LOAD_IN_MLP_MODELS:
                    all_params_for_ml = std_scaler.transform(all_params_for_ml)

                next_neuron_state = neuron_state_model.predict(all_params_for_ml)
                energy = e_static_model.predict(all_params_for_ml)

                global_neuron_state[neurons_with_leak] = next_neuron_state

                leak_events_neuron_state_per_time_step[time_step_id].append((neurons_with_leak, next_neuron_state))
                leak_events_energy_per_time_step[time_step_id].append((neurons_with_leak, energy))

            params = neuron_params[spike_neuron_ids]
            times = np.ones((spike_neuron_ids.shape[0], 1)) * period
            neuron_states = global_neuron_state[spike_neuron_ids].reshape(-1,1)

            weights = np.array(spike_events["Weight"]).reshape(-1,1)
            all_together = np.concatenate((neuron_states, weights, times, params), axis=1)

            if LOAD_IN_MLP_MODELS:
                all_together = std_scaler.transform(all_together)

            # Calculate Next State
            next_neuron_state = neuron_state_model.predict(all_together)

            # Spike or Not?
            spike_or_not = spike_or_not_model.predict(all_together)

            # Energy and Latency
            energy = e_model.predict(all_together) * spike_or_not
            static_energy = e_static_model.predict(all_together) * np.logical_not(spike_or_not)
            energy = energy + static_energy

            latency = l_model.predict(all_together) * spike_or_not

            # Update Global State
            global_neuron_state[spike_neuron_ids] = next_neuron_state

            time_since_last_update[spike_neuron_ids] = time_step_id

            # Housekeeping
            spike_events_neuron_state_per_time_step[time_step_id].append((spike_neuron_ids, next_neuron_state))
            spike_events_energy_per_time_step[time_step_id].append((spike_neuron_ids, energy))
            spike_events_latency_per_time_step[time_step_id].append((spike_neuron_ids, latency))
            spike_events_spike_or_not_per_time_step[time_step_id].append((spike_neuron_ids, spike_or_not))

    run_end_time = time.time()
    full_time_for_run = run_end_time - run_start_time
    print("{:7f}".format(full_time_for_run))

    timing_table.add_row([neurons, full_time_for_run])

write_prettytable(os.path.join('../results', f"ml_inference_wrapper_spiking_neuron_timing_table.csv"), timing_table)