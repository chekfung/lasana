# General Imports 
import numpy as np
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
from stat_helpers import min_max_normalization, normalize
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool
import math
import gc

# Imports from Helper Files
from SpikeData import *
from tools_helper import *

# Dynamically load config :)
import argparse
from dynamic_config_load import inject_config
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Name of the config file (without .py)")
args = parser.parse_args()

inject_config(args.config, globals())


'''
LASANA Dataset Creation (File 2 of creation of the dataset)
Author: Jason Ho, SLAM Lab, UT Austin
'''


figure_counter = 0

# Reads in a SPICE file to precache and pickle the PSF object if necessary from the CAD tools
def analyze_run(i):
    print("Precache Run {}".format(i))

    # Read in a SPICE run file
    spice_file = spice_runs_dir + "_{}".format(i) + ".sp"
    raw_obj = read_simulation_file(spice_file, simulator=SIMULATOR)
    del raw_obj
    gc.collect()

# -------------------------------------
# Read in Hyperparameters from the config files and from the previous testbench_generation runs

run_directory = os.path.join('../data', RUN_NAME)
HYPERPARAMETERS_CSV_FILE = os.path.join(run_directory, "randomized_output_log.csv")
DATASET_CREATION_FILENAME = os.path.join(run_directory, DF_FILENAME)

df = pd.read_csv(HYPERPARAMETERS_CSV_FILE)
df = df.set_index("Run_Number", drop=False)

HYPERPARAMETERS_FILEPATH = os.path.join(run_directory, "hyperparameters.txt")
hyperparameters_dict = {}

with open(HYPERPARAMETERS_FILEPATH, 'r') as file:
    for line in file:
        key, value = line.strip().split(': ')
        # Convert value to appropriate type if needed
        if value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)  # Handles integers, decimals, and scientific notation
            except ValueError:
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
        hyperparameters_dict[key] = value

# Print out the hyperparams to stdout
if VERBOSE:
    print("Hyperparameters from the run!")
    for a in hyperparameters_dict.keys():
        print(f"{a}: {hyperparameters_dict[a]}")

# Get SPICE Raw Files
spice_model_filename = os.path.splitext(os.path.basename(hyperparameters_dict["MODEL_FILEPATH"]))[0]
spice_runs_dir = os.path.join(run_directory, 'spice_runs', spice_model_filename)

# Write out what is going on here
if hyperparameters_dict["DIGITAL_INPUT_ENFORCEMENT"]:
    digital_freq = hyperparameters_dict["DIGITAL_FREQUENCY"]
else: 
    digital_freq = None

SIMULATOR = hyperparameters_dict["SIMULATOR"]
SPIKING_INPUT= hyperparameters_dict["SPIKING_INPUT"]
CIRCUIT_STATEFUL = hyperparameters_dict["CIRCUIT_STATEFUL"]

if not SPIKING_INPUT:
    # Since multiple inputs, need to just include all things into the columns
    columns += INPUT_NAME

    # Open up input guide and pivot table such that all the inputs for each run, timestep are on one line :)
    inputs_fp = os.path.join(run_directory, 'input_tracking.csv')

    inputs_df = pd.read_csv(inputs_fp)
    # Pivot the DataFrame
    
    collapsed_df = inputs_df.pivot_table(
        index=["Run_Number", "Digital_Timestep"],
        columns="Input_Net_Name",
        values="Value"
    ).reset_index()

    # Flatten the column headers (optional, in case of MultiIndex)
    collapsed_df.columns.name = None
    collapsed_df.columns = [str(col) for col in collapsed_df.columns]

    non_numeric_cols = ["Run_Number", "Digital_Timestep"]
    numeric_cols = sorted([col for col in collapsed_df.columns if col.startswith('v')], key=lambda x: int(x[1:]))

    # Reorder the DataFrame columns
    input_per_timestep = collapsed_df[non_numeric_cols + numeric_cols]
    #input_per_timestep = input_per_timestep.set_index(["Run_Number", "Digital_Timestep"])
    
    # Save it :)
    input_pivot_fp = os.path.join(run_directory, "input_pivoted.csv")
    input_per_timestep.to_csv(input_pivot_fp, index=False)


total_simulation_time = hyperparameters_dict["TOTAL_TIME_NS"] * 10**-9
number_of_timesteps = int(total_simulation_time * digital_freq)
one_time_period = total_simulation_time / number_of_timesteps
NUMBER_OF_RAW_RUNS = int(hyperparameters_dict["NUMBER_OF_RUNS"])
OUTPUT_SPIKE_THRESHOLD = 1 * 10**-6      # use this value to determine if a trace has any spikes (i.e. if nothing above 1 uA, then no spike in whole thing)
events_list = []

# Before Script Starts, load everything in parallel so we are not bottlenecked in the future.
if SIMULATOR == 'hspice' or SIMULATOR == 'spectre':
    with Pool() as pool:
        pool.map(analyze_run, range(NUMBER_OF_RAW_RUNS))


# -----------
# Start of Dataset Creation Script

for i in range(NUMBER_OF_RAW_RUNS):
    # Load in Raw file and get input, output
    print("Analyzing Run {}".format(i))

    # Read in a spice run file
    spice_file = spice_runs_dir + "_{}".format(i) + ".sp"
    raw_obj = read_simulation_file(spice_file, simulator=SIMULATOR)

    if VERBOSE:
        print_signal_names(raw_obj, simulator=SIMULATOR)

    # Grab run parameter that were randomized
    run_randomized_parameters = df.loc[i]
    
    # Get Sweep
    raw_time = get_signal("time", raw_obj, simulator=SIMULATOR)

    if SIMULATOR == 'ltspice':
        # Need to mask out negative time for some reason
        time_vec = mask_negative_time(raw_time, raw_time)
    else:
        time_vec = raw_time

    if not SPIKING_INPUT and not CIRCUIT_STATEFUL:
        # Traditional analog case where not spiking output and not stateful :)

        # Get output nets
        output_signal = get_signal(OUTPUT_NAME, raw_obj, simulator=SIMULATOR)

        # Load in VDD, VSS nets for power calculation
        vdd = get_signal(VDD_VOLTAGE, raw_obj,simulator=SIMULATOR)[0]
        vdd_current = get_signal(VDD_CURRENT, raw_obj,simulator=SIMULATOR)
        instantaneous_power_vdd = np.abs(vdd * vdd_current)

        vss = get_signal(VSS_VOLTAGE, raw_obj, simulator=SIMULATOR)[0]
        vss_current = get_signal(VSS_CURRENT, raw_obj, simulator=SIMULATOR)
        instantaneous_power_vss = np.abs(vss * vss_current)

        instantaneous_power = instantaneous_power_vdd + instantaneous_power_vss

        if PLOT_RUNS:
            plt.plot(time_vec*10**9, normalize(instantaneous_power), label='pwr')
            plt.plot(time_vec*10**9, normalize(output_signal), label='output')

            for a in range(number_of_timesteps):
                plt.axvline(a*one_time_period*10**9, c='black')
                plt.text(a*one_time_period*10**9, .5, f'{a}', rotation=90, verticalalignment='bottom', fontsize=8)

        # Try something out
        run_df = input_per_timestep[input_per_timestep["Run_Number"] == i].sort_values(by="Digital_Timestep")

        # Select only the input columns (assuming input columns start from column 2 onward)
        input_columns = run_df.columns.difference(["Run_Number", "Digital_Timestep"])

        # Calculate the difference between each timestep and the previous one
        # Keep only rows where there is no difference (all values are zero)
        no_change_timesteps = run_df.loc[(run_df[input_columns].diff().fillna(1) == 0).all(axis=1),"Digital_Timestep"]

        # Convert to list of timesteps with no change
        no_change_timesteps = no_change_timesteps.tolist()

        # Get list of unique events, separated into static and dynamic events.
        timestep_guys = np.arange(number_of_timesteps)
        events_tuple_list = create_event_list_no_spike(no_change_timesteps, timestep_guys)

        # Convert into easier way to get access to things
        input_per_timestep_easy_index = input_per_timestep.set_index(["Run_Number", "Digital_Timestep"])

        # Get knob parameters
        run_weights = {}

        for knob in KNOB_NAMES:
            run_weights[knob] = run_randomized_parameters[knob]

        for event_id, (start_timestep, end_timestep, event_type) in enumerate(events_tuple_list):
            if VERBOSE:
                print(f"Run #{i} Event Start Timestep: {start_timestep}, Event End Timestep: {end_timestep}, Event Type: {event_type}")

            # Get input data in order
            input_voltages_per_timestep = input_per_timestep_easy_index.loc[(i, start_timestep)].to_dict()

            start_of_event = start_timestep * one_time_period
            
            if end_timestep != 'End':
                end_of_event = end_timestep * one_time_period
            else: 
                end_of_event = total_simulation_time
                end_timestep = number_of_timesteps

            # Convert to the indices that work with our SPICE simulation.
            bounds_mask = (time_vec >= start_of_event) & (time_vec <= end_of_event)
            valid_timestep_indices = np.where(bounds_mask)

            # Fix the end index to be plus one (so that we slightly overlap things.)
            start_index = valid_timestep_indices[0][0]  
            end_index_i = valid_timestep_indices[0][-1]+1
            end_index = np.min([end_index_i, len(time_vec)-1])      # Join the end of the index with the start of the new guy :)

            if PLOT_RUNS:
                plt.axvline(time_vec[start_index]*10**9, c='green')
                plt.axvline(time_vec[end_index]*10**9, c='red')

            # Look at energy
            event_energy = np.trapz(instantaneous_power[start_index:end_index+1], time_vec[start_index:end_index+1])

            # Look at latency
            if event_type == 'leak':
                event_latency = 0
                end_dynamic = start_index
            else: 
                # Dynamic 
                # Get dynamic after it reaches end
                subset_output = output_signal[start_index:end_index]

                # Find Voltage Swing 
                initial_voltage = output_signal[start_index]
                end_voltage = output_signal[end_index-2]
                voltage_swing = end_voltage-initial_voltage
                threshold = 0.9

                # Latency defined as when output voltage hits 90% of final
                threshold_voltage = initial_voltage + (voltage_swing * threshold)

                if voltage_swing >= 0:
                    crossing_index = np.where(subset_output >= threshold_voltage)[0]
                else:
                    crossing_index = np.where(subset_output <= threshold_voltage)[0]

                p = len(crossing_index) - 1

                while p > 0 and crossing_index[p] - crossing_index[p - 1] == 1:
                    p -= 1
                crossing_index = crossing_index[p]

                # Interpolation between i-1 and i
                subset_time = time_vec[start_index:end_index]
                
                if crossing_index == 0:
                    # Can't interpolate before first point, fallback to index 0
                    crossing_time = subset_time[crossing_index]
                else:
                    # Interpolate between points to get closer
                    x0, y0 = subset_time[crossing_index-1], subset_output[crossing_index-1]
                    x1, y1 = subset_time[crossing_index], subset_output[crossing_index]
                    y_thresh = threshold_voltage

                    if y1-y0 == 0:
                        crossing_time = subset_time[crossing_index]
                    else:
                        crossing_time = x0 + (y_thresh - y0) / (y1 - y0) * (x1 - x0)

                end_dynamic = start_index + crossing_index
                event_latency = crossing_time - time_vec[start_index]

                if event_latency == 0.0:
                    print(f"Latency of 0 detected. Adjusting for Run: {i}, Start Timestep: {start_timestep}")
                    event_latency = time_vec[start_index+1] - time_vec[start_index]


                if math.isinf(event_latency) or math.isnan(event_latency):
                    print("Warning: invalid latency, skipping")
                    latency = -1  # or some flag value like -1 or 'error'

            if VERBOSE:
                print(f'Run #{i}, Event ID: {event_id}, Type: {event_type}, Event Bounds: {start_index}:{end_index}, Energy: {event_energy}, Latency: {event_latency}')

            event = {}
            event["Run_Name"] = RUN_NAME
            event["Run_Number"] = i
            event["Event_Start_Index"] = start_index 
            event["Event_End_Index"]  = end_index
            event["Event_Type"] = event_type
            event["Digital_Time_Step"] = start_timestep
            event["Input_Total_Time"] = (end_timestep - start_timestep) * one_time_period  
            event['Latency'] = float(event_latency)
            event['Energy'] = event_energy
            event['Output_Value'] = subset_output[-1]
            event['Last_Output_Value'] = output_signal[start_index]

            if PLOT_RUNS:
                plt.axvline(time_vec[end_index-1]*10**9, color='red')
                plt.axvline(time_vec[end_dynamic]*10**9, color='orange')

            # Add in weights and inputs
            event.update(run_weights)
            event.update(input_voltages_per_timestep)

            events_list.append(event)

        if PLOT_RUNS:
            plt.legend()
            plt.show()

    else:
        # Get I/O (In spiking case, assume that there is just one input)
        input_spikes = get_signal(INPUT_NAME[0], raw_obj, simulator=SIMULATOR)
        circuit_state = get_signal(STATE_NET, raw_obj, simulator=SIMULATOR)
        output_spikes = get_signal(OUTPUT_NAME, raw_obj, simulator=SIMULATOR)

        # Get Power Ingredients
        vdd = get_signal(VDD_VOLTAGE, raw_obj,simulator=SIMULATOR)[0]
        vdd_current = get_signal(VDD_CURRENT, raw_obj,simulator=SIMULATOR)
        instantaneous_power = np.abs(vdd * vdd_current)

        if SIMULATOR == 'ltspice':
            # NOTE: For some reason, LTSpice has negative time sometimes. We use this to mask it out. 
            # Not sure what causes this error.
            input_spikes = mask_negative_time(raw_time, input_spikes)
            circuit_state = mask_negative_time(raw_time, circuit_state)
            output_spikes = mask_negative_time(raw_time, output_spikes)
            instantaneous_power = mask_negative_time(raw_time, instantaneous_power)

        # ----

        # Get gradients (first derivative)
        input_gradients = np.gradient(input_spikes, time_vec)
        output_gradients = np.gradient(output_spikes, time_vec)

        out_mask = output_spikes > OUTPUT_SPIKE_THRESHOLD
        num_values_over_thresh_output = np.sum(out_mask)

        # Remove Drift :)
        z = baseline_als(output_spikes, 10000, 0.01)
        filtered_output = output_spikes-z
        
        # Remove Baseline
        if not INPUT_WELL_DEFINED:
            z1 = baseline_als(np.abs(input_spikes), 10000, 0.01)
            filtered_input = input_spikes-z
        else:
            filtered_input = np.abs(input_spikes)

        # Get Mask of the situation (inputs and outputs)
        normalized_filtered_input = normalize(filtered_input)
        input_mask = (normalized_filtered_input > 0.01*3).astype(int)

        normalized_filtered_output = normalize(filtered_output)

        # output_mask = where output becomes larger than the baseline * 2; out_mask = whether parts of the algo greater than the threshold; 
        output_mask = (normalized_filtered_output > normalized_filtered_output[0] * 3)

        # gradient_spike_mask = where gradient goes larger than one percent of baseline (NOTE: If this is not that noise tolerant, than we can adjust, but tentatively, no noise in the system)
        gradient_spike_mask = get_gradient_where_spike(output_spikes, output_gradients, time_vec)

        combine_all_masks = output_mask & out_mask & gradient_spike_mask   
    
        #Plot everything right now with the timestep bounds :)
        if PLOT_RUNS:
            plt.figure(figure_counter)
            figure_counter+=1
            plt.plot(time_vec*10**9,normalize(input_spikes), color='green', label='in')
            plt.plot(time_vec*10**9,normalize(output_spikes), color='red', label='out')
            plt.plot(time_vec*10**9,input_mask,color='purple', label='in_mask')
            plt.plot(time_vec*10**9, combine_all_masks, color='purple', ls='--',label='out_spike')
            #plt.plot(time_vec*10**9, output_mask, color='black', linewidth=2, label='out mask')
            plt.plot(time_vec*10**9, circuit_state, label='circuit_state', color='black', ls='--')
            #plt.plot(time_vec*10**9,normalize(output_gradients), color='orange')
            #plt.plot(time_vec*10**9, meow, color='black')

            knob_vals = {}
            for knob in KNOB_NAMES:
                knob_vals[knob] = run_randomized_parameters[knob]
                
            plt.title(f"Run {i}, Run Knobs: {knob_vals}")

        timesteps_without_events = 0

        # Loop through each of the timesteps now :)
        for j in range(number_of_timesteps):
            if VERBOSE:
                print(f"Timestep {j}")
            start_of_timestep = j * one_time_period
            end_of_timestep = (j + 1) * one_time_period

            # Convert to the indices that work with our SPICE simulation.
            bounds_mask = (time_vec >= start_of_timestep) & (time_vec <= end_of_timestep)
            valid_timestep_indices = np.where(bounds_mask)

            if valid_timestep_indices[0].shape[0] == 0:
                timesteps_without_events+=1
                continue

            # Fix the end index to be plus one (so that we slightly overlap things.)
            start_index = valid_timestep_indices[0][0]  
            end_index = valid_timestep_indices[0][-1]+1
            end_index = np.min([end_index, len(time_vec)-1])      # Join the end of the index with the start of the new guy :)

            # Continue writing on that plot to show where these things are :)
            if PLOT_RUNS:
                #plt.axvline(start_of_timestep*10**9, color='black')
                plt.axvline(time_vec[start_index]*10**9, color='pink')
                plt.text(time_vec[start_index]*10**9, .5, f'{j}', rotation=90, verticalalignment='bottom', fontsize=8)

            # First look for input spike. If there is no input spike, there is no output spike...
            spike_found, input_start_spike_index, input_end_spike_index = identify_nice_input_spike(input_mask, np.max([start_index-1,0]), np.min([end_index+1, len(time_vec)-1]))

            if spike_found:
                # Spike found, that means that if there are any timing events, they end here.
                # This also means that this spike event will become the event this timestep (along with the leakage event)
                
                if timesteps_without_events > 0:
                    # Need to account for leakage events.
                    timestep_since_last_spike = (j - timesteps_without_events) * one_time_period

                    # Gather all of the information to create the event
                    # Create dictionary of the things to put in
                    timing_event = {}
                    timing_event_start_index = np.where(time_vec >= timestep_since_last_spike)[0][0]
                    timing_event_end_index = start_index # Note: Do not include this in the timing event. No Plus 1 there, I think

                    timing_event["Run_Name"] = RUN_NAME
                    timing_event["Run_Number"] = i
                    timing_event["Event_Start_Index"] = timing_event_start_index 
                    timing_event["Event_End_Index"]  = timing_event_end_index
                    timing_event["Event_Type"] = 'leak'
                    timing_event["Digital_Time_Step"] = j
                    timing_event["Input_Peak_Amplitude"] = 0
                    timing_event["Input_Total_Charge"] = 0
                    timing_event["Input_Total_Time"] = timesteps_without_events * one_time_period  
                    timing_event["Weight"] = 0
                    timing_event["Cap_Voltage_At_Input_Start"] = circuit_state[timing_event_start_index]
                    timing_event["Cap_Voltage_At_Output_End"] = circuit_state[timing_event_end_index]
                    timing_event['Latency'] = 0
                    timing_event['Energy'] = np.trapz(instantaneous_power[timing_event_start_index:timing_event_end_index+1], time_vec[timing_event_start_index:timing_event_end_index+1])

                    # Put in Knobs
                    for knob in KNOB_NAMES:
                        knob_value = run_randomized_parameters[knob]
                        timing_event[knob] = knob_value

                    events_list.append(timing_event)

                    #print(run_randomized_parameters)
                    if VERBOSE:
                        print(f"Timing Event {timing_event_start_index}:{timing_event_end_index}")

                # After processing the timing events, if it exists, now we can process the spike event.
                timesteps_without_events = 0

                # --------------- Process Spike Event Now ----------------
                if VERBOSE:
                    print(f"Spike Event Timestep: {start_index}:{end_index}")

                # Get the peak index for the spike for latency calculations :)
                spike_peak_index = np.argmax(np.abs(input_spikes[input_start_spike_index:input_end_spike_index+1])) + input_start_spike_index # Note: make sure to plus one to include input_end_spike_index in there

                plt.axvline(time_vec[spike_peak_index]*10**9, color='blue')

                spike_event = {}

                # House Keeping
                spike_event["Run_Name"] = RUN_NAME
                spike_event["Run_Number"] = i
                spike_event["Event_Start_Index"] = start_index 
                spike_event["Event_End_Index"]  = end_index
                spike_event["Digital_Time_Step"] = j

                # Input Spike Details
                footprint_spike_charge = hyperparameters_dict["FOOTPRINT_CHARGE"]
                spike_charge =  np.trapz(input_spikes[input_start_spike_index:input_end_spike_index+1], time_vec[input_start_spike_index:input_end_spike_index+1])
                spike_event["Input_Peak_Amplitude"] = input_spikes[spike_peak_index]
                spike_event["Input_Total_Charge"] = spike_charge
                spike_event["Input_Total_Time"] = (timesteps_without_events+1) * one_time_period   # Need to include all of 124 inside of 124, since no step 125 :)
                spike_event["Weight"] = spike_charge / footprint_spike_charge      
                #print(f"Spike Event: Run: {i}, Timestep: {j}, Weight: {spike_charge / footprint_spike_charge}")

                # I/O
                spike_event["Cap_Voltage_At_Input_Start"] = circuit_state[start_index]
                spike_event["Cap_Voltage_At_Output_End"] = circuit_state[end_index]
                spike_event['Energy'] = np.trapz(instantaneous_power[start_index:end_index+1], time_vec[start_index:end_index+1])

                # Put in Knobs
                for knob in KNOB_NAMES:
                    knob_value = run_randomized_parameters[knob]
                    spike_event[knob] = knob_value


                # Is there an output spike in the same timestep? 
                output_spike_found, output_start_index = identify_if_output_spike(combine_all_masks, output_mask, start_index, end_index,j)
                if VERBOSE:
                    print(f"Output Spike?: {output_spike_found}")

                if output_spike_found:
                    if output_start_index < start_index:
                        # Not real spike :()
                        event_type = 'in-no_out'
                        latency = 0
                        
                        print(f"WARNING: NOT A REAL SPIKE :o AT RUN: {i}, TIMESTEP: {j}")

                    elif output_start_index == start_index:
                        # Real spike, but I will just move it over a little bit so it is not something crazy :)
                        #event_type = 'in-out'
                        #output_spike_peak+=4
                        #latency = time_vec[output_spike_peak] - time_vec[spike_peak_index]
                        event_type='in-no_out'
                        latency=0

                        print(f"WARNING: SPIKE SAME INDICES AT RUN: {i}, TIMESTEP: {j}")

                    else:
                        # Anything else :)
                        latency = time_vec[output_start_index] - time_vec[start_index]
                        event_type = 'in-out'

                    plt.axvline(time_vec[output_start_index]*10**9, color='red')

                else:
                    # No output spike, in-no_out
                    latency = 0
                    event_type = 'in-no_out'

                # These two are dependent on if we have a spike or not :)
                spike_event["Event_Type"] = event_type
                spike_event['Latency'] = latency

                events_list.append(spike_event)

            else:
                # If spike not found, we need to check if this is the last timestep. If so, we need to make a timing event for this last timestep or more (significant leakage events)
                if j == number_of_timesteps - 1:
                    # If last timestep, but no spike, we need to evaluate this as a timing event :)
                    timestep_since_last_spike = (j - timesteps_without_events) * one_time_period

                    timing_event = {}
                    timing_event_start_index = np.where(time_vec >= timestep_since_last_spike)[0][0]
                    timing_event_end_index = end_index # Note: This is end index because there is no event, we process this timing event all the way to the end :)
                    if VERBOSE:
                        print(f"Timing Event Length: {len(time_vec)}, End Index: {end_index}")

                    timing_event["Run_Name"] = RUN_NAME
                    timing_event["Run_Number"] = i
                    timing_event["Event_Start_Index"] = timing_event_start_index 
                    timing_event["Event_End_Index"]  = timing_event_end_index
                    timing_event["Event_Type"] = 'leak'
                    timing_event["Digital_Time_Step"] = j
                    timing_event["Input_Peak_Amplitude"] = 0
                    timing_event["Input_Total_Charge"] = 0
                    timing_event["Input_Total_Time"] = (timesteps_without_events+1) * one_time_period   # Need to include all of 124 inside of 124, since no step 125 :)
                    timing_event["Weight"] = 0
                    timing_event["Cap_Voltage_At_Input_Start"] = circuit_state[timing_event_start_index]
                    timing_event["Cap_Voltage_At_Output_End"] = circuit_state[timing_event_end_index]
                    timing_event['Latency'] = 0
                    timing_event['Energy'] = np.trapz(instantaneous_power[timing_event_start_index:timing_event_end_index+1], time_vec[timing_event_start_index:timing_event_end_index+1])

                    # Put in Knobs
                    for knob in KNOB_NAMES:
                        knob_value = run_randomized_parameters[knob]
                        timing_event[knob] = knob_value

                    events_list.append(timing_event)
                
                # if no spike, add one to the timesteps_without_events :)
                timesteps_without_events+=1

        if PLOT_RUNS:
            plt.legend()
            plt.show()


# After the end of everything, put everything into a dataframe and export to csv
event_df = pd.DataFrame(events_list, columns=columns)
print("Dataframe Created!")

# Drop everything from first timestep (since can be garbage at startup)
event_df = event_df[event_df["Digital_Time_Step"] != 0]

if VERBOSE:
    print(event_df.to_string())

    event_df = event_df.sort_values(by=['Run_Number', 'Digital_Time_Step', "Event_Type"], ascending=[True, True, False])

    # Print any crazies first :)
    filtered_df = event_df[event_df.groupby(['Run_Number', 'Event_End_Index'])['Event_End_Index'].transform('count') > 1]
    print("---------------- Preliminary Red Flags -----------------")
    print("Any Rows that share the same run number, and event index (should never happen)")
    print(filtered_df.to_string())


    print("---------------- Negative Energies ------------------")
    filtered_df = event_df[event_df['Energy'] < 0]
    print(filtered_df.to_string())


    print("---------------- Negative Latencies ------------------")
    filtered_df = event_df[event_df['Latency'] < 0]
    print(filtered_df.to_string())
    print("------------------------------------------------------\n\n")

# Print Statistics
moo = event_df[event_df["Event_Type"].isin(['in-out', "in-no_out"])]
woof = event_df[event_df["Event_Type"] == 'in-out']
meow = event_df[event_df["Event_Type"] == 'leak']

# Create log file that prints out intermediate statistics
stats_file = open(os.path.join(run_directory, "analyze_dataset_statistics.txt"), 'w')
print_and_log(stats_file, "Statistics")
print_and_log(stats_file, f"Number of Runs: {NUMBER_OF_RAW_RUNS}")
print_and_log(stats_file, f"CSV Number of Columns:{event_df.shape[1]}")
print_and_log(stats_file, f"Total Events: {event_df.shape[0]}")
print_and_log(stats_file, f"Number of Input Spike Events: {moo.shape[0]}")
print_and_log(stats_file, f"Number of inputs with Outputs: {woof.shape[0]} / {moo.shape[0]} ({woof.shape[0] / moo.shape[0] * 100:.2f}%)")
print_and_log(stats_file, f"Number of Timing Events: {meow.shape[0]}")
print_and_log(stats_file, "\n")
print_and_log(stats_file, "---------------- Timing Event Statistics ------------------")

# Statistics of Latency Events
leak_df = event_df[event_df['Event_Type'] == 'leak'].copy()  # .copy() to ensure we are working with a new dataframe
leak_df.loc[:, 'Timestep'] = (leak_df['Input_Total_Time'] / one_time_period).astype(int)
timestep_counts = leak_df['Timestep'].value_counts().sort_index()

# Calculate the percentage of each timestep
total_timesteps = len(leak_df)

# Print each timestep count in the desired format
for timestep, count in timestep_counts.items():
    percentage = (count / total_timesteps) * 100
    print_and_log(stats_file, f"Latency Number of Timesteps: ({timestep}) {count} / {total_timesteps} ({percentage:.2f}%)")
print_and_log(stats_file, "\n------------------------------------------------------\n")

print_and_log(stats_file, leak_df.sort_values(by='Timestep', ascending=True))

# Save away!
event_df.to_csv(DATASET_CREATION_FILENAME, index=False)
print_and_log(stats_file, "Saved!")

# At the end, show all the figures together
if PLOT_RUNS:
    plt.legend()
    plt.show(block=False)
    plt.pause(0.001) # Pause for interval seconds.
    input("hit[enter] to end.")
    plt.close('all') # all open plots are correctly closed after each run

