import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import shutil

# JH created helper files 
from create_spikes import *
from tools_helper import *

# Dynamically load configs :)
import argparse
from dynamic_config_load import inject_config
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Name of the config file (without .py)")
args = parser.parse_args()

inject_config(args.config, globals())

# ======================
'''
This file is solely used to run SPICE simulations to characterize an analog circuit implemented in SPICE. (File 1 of creation of the dataset)
Author: Jason Ho, SLAM Lab, UT Austin
'''

# ----
# Setup all the filepaths

# Simulator detection
sim_str = idiot_proof_sim_string(SIMULATOR)
assert(is_simulator_real(sim_str))

pwl_file_template = RUN_NAME + "_" + "pwl_file_{}_{}.txt"       # First one referes to run number, second is input net connection :)

# File IO to create directory structure for run
run_directory = os.path.join('../data', RUN_NAME)
libraries_directory = os.path.join(run_directory, 'libraries')
pwl_file_main_directory = os.path.join(run_directory, "pwl_files")
spice_run_directory = os.path.join(run_directory, 'spice_runs')

# Make necessary directories by using the most nested directory
if not os.path.exists(pwl_file_main_directory):
    os.makedirs(pwl_file_main_directory)
    print(f"Directory '{pwl_file_main_directory}' created successfully.")

# Copy model file into logging directory (such that runs occur in that directory as well)
filename = os.path.basename(MODEL_FILEPATH)
local_spice_model_filepath = os.path.join(spice_run_directory, filename)

# Copy over SPICE file
if not os.path.exists(spice_run_directory):
    os.makedirs(spice_run_directory)

shutil.copyfile(MODEL_FILEPATH, local_spice_model_filepath)

# Copy over other files into spice_run_directory
if OTHER_NECESSARY_FILES_IN_SPICE_RUN_DIRECTORY:
    for necessary_file in OTHER_NECESSARY_FILES_IN_SPICE_RUN_DIRECTORY:
        shutil.copyfile(necessary_file, os.path.join(spice_run_directory, os.path.basename(necessary_file)))

# Copy over Library File
if not os.path.exists(libraries_directory):
    os.makedirs(libraries_directory)

for library_filepath in LIBRARY_FILES:
    library_filename = os.path.basename(library_filepath)
    lib_destination_file = os.path.join(libraries_directory, library_filename)

    shutil.copyfile(library_filepath, lib_destination_file)

# ---------



# ======================
# Start of script
current_runs = 0
total_sim_time_s = TOTAL_TIME_NS * 10**(-9)
sampling_period = total_sim_time_s / NUM_INPUT_POINTS
sampling_frequency = 1 / sampling_period

# Start Tracking Randomized Values
# Dictionary where {RUN NUMBER (int): list of tuple of (name of value, value)}
randomized_values = defaultdict(list)

# Spike timeframe
time_vector = np.linspace(0, total_sim_time_s, NUM_INPUT_POINTS)

if SPIKING_INPUT:
    # Get spiking footprint that is used for input generation for spiking
    spike_footprint =  analyze_spike_file(SPICE_FOOTPRINT_FILE, OUTPUT_SPIKE_NAME, SPIKE_START, SPIKE_END, sampling_period, PLOT_SPIKE_BOUNDS, simulator=sim_str)
    footprint_charge = np.trapz(spike_footprint, time_vector[:spike_footprint.shape[0]])
    print("Total Charge of Spike, weight 1: {}".format(footprint_charge))
    print("Peak Amplitude of One Guy: {}".format(np.max(spike_footprint)))
    print("NUMBER OF SPIKE POINTS: {}".format(spike_footprint.shape[0]))

    if CUSTOM_SPIKE:
        spike_points = spike_footprint.shape[0]
        custom_time_vector = time_vector[:spike_points]

        # Note: Generates about 0.3, 0.4 amps in 16 time points, or 4.0 * 10^-10, or 0.4 nanoseconds
        total_charge = np.trapz(spike_footprint, custom_time_vector)

        total_spike_time = CUSTOM_SPIKE_NUM_TIME_STEP_LENGTH * sampling_period
        print("Total Spike Time: {} ns".format(total_spike_time * 10**9))
        height_of_pwm_wave = total_charge / total_spike_time

        spike_footprint = np.empty(CUSTOM_SPIKE_NUM_TIME_STEP_LENGTH)
        spike_footprint.fill(height_of_pwm_wave)


# Generate PWL files for all input runs

# Create Inputs File :)
if not SPIKING_INPUT:
    inputs_filepath_tracking = os.path.join(run_directory, 'input_tracking.csv')
    inputs_fd = open(inputs_filepath_tracking, 'w')
    header = 'Run_Number,Input_Net_Name,Digital_Timestep,Value\n'
    inputs_fd.write(header)

# Setup which runs will be all negative / all positive
run_input_same_sign = np.random.choice([0, 1], size=NUMBER_OF_RUNS, p=[1-SAME_SIGN_WEIGHTS_FRACTION, SAME_SIGN_WEIGHTS_FRACTION])

while (current_runs < NUMBER_OF_RUNS):
    # Go through and create PWL files for each of the inputs :)
    if SPIKING_INPUT:
        for (net_name, net) in zip(INPUT_NET_NAME, INPUT_NET):
            # Get random stuff
            circuit_fan_in = random.randint(CIRCUIT_FAN_IN_RANGE[1], CIRCUIT_FAN_IN_RANGE[2])
            circuit_num_input_spikes = random.randint(CIRCUIT_SIM_NUM_SPIKES[1], CIRCUIT_SIM_NUM_SPIKES[2])

            randomized_values[current_runs].append((CIRCUIT_SIM_NUM_SPIKES[0]+'_'+ net_name, circuit_num_input_spikes))
            randomized_values[current_runs].append((CIRCUIT_FAN_IN_RANGE[0]+'_'+ net_name, circuit_fan_in))

            # Generate input Spike Map and then add the spike footprint on top of that.
            try:
                if DIGITAL_INPUT_ENFORCEMENT:
                    spike_map = generate_spike_map_digital(circuit_fan_in, circuit_num_input_spikes, REFRACTORY_PERIOD, total_sim_time_s, sampling_frequency, DIGITAL_FREQUENCY, RUN_TIMEOUT, DELAY_RATIO)
                else:
                    spike_map = generate_spike_map(circuit_fan_in, circuit_num_input_spikes, REFRACTORY_PERIOD, total_sim_time_s, sampling_frequency, RUN_TIMEOUT, DELAY_RATIO)
            except Exception as e:
                # If error in generation, then continue again
                print("Could not generate spike map within timeout of {} seconds. Trying again".format(RUN_TIMEOUT))
                continue

            # Applies random weight to each of the spikes in the spike map
            low_weight = WEIGHT_LOW
            high_weight = WEIGHT_HIGH

            if run_input_same_sign[current_runs]:
                # All inputs have the same sign :)
                results_all_positive = np.random.choice([True, False])

                if results_all_positive:
                    # if all results positive :)
                    low_weight = 0
                else:
                    # Otherwise, clip high weight to 0 :)
                    high_weight = 0

            input_vector = generate_output_vector(spike_map, spike_footprint, low_weight, high_weight)
            
            if PLOT_SPIKE_BOUNDS:
                plt.figure(0)
                plt.plot(time_vector*10**9, input_vector)
                plt.title(f"Input Spike Vector for Run {current_runs}")

                if DIGITAL_INPUT_ENFORCEMENT:
                    # Plot digital timestep start and end
                    num_samples = int(total_sim_time_s * sampling_frequency)
                    num_digital_samples = int(total_sim_time_s * DIGITAL_FREQUENCY)
                    digital_period = total_sim_time_s / num_digital_samples


                    for a in range(num_digital_samples):
                        plt.axvline(a*digital_period*10**9, c='pink')
                        plt.text(a*digital_period*10**9, 0, f'{a}', rotation=90, verticalalignment='bottom', fontsize=8)
                
                plt.show()
                exit()

            # Create PWL File to store it
            pwl_file = os.path.join(pwl_file_main_directory, pwl_file_template.format(current_runs, net_name))
            create_pwl_file(pwl_file, time_vector, input_vector)
    else:
        # Determine inputs first :)
        time_period_s = 1 / DIGITAL_FREQUENCY
        num_periods_in_total_sim_time = int(total_sim_time_s / time_period_s) 

        # Determine transition time as percentage of the time of each guy (10% nominal for now)
        transition_time = time_period_s * 0.1

        # Number of nets to randomize
        num_nets = len(INPUT_NET)
        all_inputs = np.zeros((num_nets, num_periods_in_total_sim_time))

        # Check what should be the top and bottom of inputs possible :)
        low_voltage = VSS
        high_voltage = VDD

        try:
            if SET_INPUT_BOUND:
                low_voltage = LOW_INPUT_BOUND
                high_voltage = HIGH_INPUT_BOUND
        except:
            pass
        
        if run_input_same_sign[current_runs]:
            # Coin flip for all pos or neg
            results_all_positive = np.random.choice([True, False])

            if results_all_positive:
                low_voltage = 0
            else:
                # if everything should be negative
                high_voltage = 0

        for k in range(num_periods_in_total_sim_time):
            # Loop through and determine if it is funny or not.
            BINARY_INPUT = coin_flip(BINARY_INPUT_FRACTION)

            if k==0:
                # Base Case
                if BINARY_INPUT:
                    binary_input_vector = np.random.choice([-1, 1], size=num_nets)
                    theoretical_voltage_vector = np.where(binary_input_vector == -1, low_voltage, high_voltage)
                else:
                    theoretical_voltage_vector = np.random.uniform(low_voltage, high_voltage, size=num_nets)

                all_inputs[:,k] = theoretical_voltage_vector
            else:
                # Coin flip as to whether different or not :)
                same_input = coin_flip(DELAY_RATIO)

                if same_input:
                    all_inputs[:,k] = all_inputs[:,k-1]
                else:
                    # Base Case
                    if BINARY_INPUT:
                        binary_input_vector = np.random.choice([-1, 1], size=num_nets)
                        theoretical_voltage_vector = np.where(binary_input_vector == -1, low_voltage, high_voltage)
                    else:
                        theoretical_voltage_vector = np.random.uniform(low_voltage, high_voltage, size=num_nets)

                    all_inputs[:,k] = theoretical_voltage_vector

        for idx, (net_name, net) in enumerate(zip(INPUT_NET_NAME, INPUT_NET)):
            # Nonspiking input for PWL, more akin to generation of pulses :)
            # Digital frequency enforced
            if not DIGITAL_INPUT_ENFORCEMENT:
                print("ERROR in generate_dataset.py: Need to enforce digital input if regular pulse input 11/3/2024")
                exit(1)

            corresponding_timing_vector_with_input = np.arange(0, num_periods_in_total_sim_time)
            voltage_vector = all_inputs[idx, :]

            # Write all of this into a nice inputs file for easy parsing later on :)
            for i in range(corresponding_timing_vector_with_input.shape[0]):
                format = f"{current_runs},{net_name},{corresponding_timing_vector_with_input[i]},{voltage_vector[i]}\n"
                inputs_fd.write(format)

            # Generate PWL files :)
            pwl_file = os.path.join(pwl_file_main_directory, pwl_file_template.format(current_runs, net_name))
            create_pwl_from_input_pulses(pwl_file, time_period_s, transition_time, voltage_vector)

    # Increment current runs
    current_runs+=1

if not SPIKING_INPUT:
    inputs_fd.close()

current_sim_runs = 0
runs = []

# ------------------

# Create all the simulation files 
for i in range(NUMBER_OF_RUNS):
    # Get filepath for output file based on current_sim_runs
    fn, fn_ext = os.path.splitext(os.path.basename(local_spice_model_filepath))
    new_filename = f"{fn}_{current_sim_runs}{fn_ext}"
    output_fp = os.path.join(os.path.dirname(local_spice_model_filepath), new_filename)

    if not SPIKING_INPUT:
        randomized_values[current_sim_runs] = []
    
    with open(local_spice_model_filepath, 'r') as infile, open(output_fp, 'w') as outfile:

        if SPIKING_INPUT:
            # Copy over subcircuit
            for line in infile:
                outfile.write(line)

            # Write Vdd
            write_voltage(outfile, 'Vdd', 'vdd', 0, VDD)

            # Add constrained variables
            for constrained_tuple in KNOB_PARAMS:
                netlist_name = constrained_tuple[0]
                net_name = netlist_name.split('_', 1)[-1]
                netlist_voltage = random.uniform(constrained_tuple[1], constrained_tuple[2])

                # Log into the output
                randomized_values[current_sim_runs].append((netlist_name, round(netlist_voltage, 3)))

                write_voltage(outfile, netlist_name, net_name, VSS, netlist_voltage)

            # Write PWL File and input voltage source connected to it
            for (net_name, net) in zip(INPUT_NET_NAME, INPUT_NET):
                full_pwl_filename_spice = os.path.join("../pwl_files", pwl_file_template.format(current_sim_runs, net_name))
                write_input_spike_file(outfile, net_name, f'vdd_2_{net}', net, full_pwl_filename_spice, VDD, simulator=SIMULATOR, write_voltage_src=INPUT_SEPARATE_VDD)
        else:
            # Lines that need to change are net names that are not voltages tentatively
            v_params, other_params = separate_v_params(KNOB_PARAMS)
            weights = np.random.choice([-1, 0, 1], size=len(other_params), p=[1/3, 1/3, 1/3])
            weight_index = 0

            nets_to_change = {}
            zero_weight_nets = {}

            # Go through other params that require augmenting the SPICE model file
            for name, low_val, high_val, opt_flag in other_params:
                type_of_net, num_net = name.lower().split('_')

                # We only handle changing of weights right now for the guy
                if type_of_net == 'weight' or type_of_net == 'bias':
                    # Get weight nets to change
                    positive_weight, negative_weight = WEIGHT_NET_NAMES_TO_CHANGE[name]

                    if opt_flag == 'b':
                        pos_map, neg_map = map_weight_to_memristive_crossbar(weights[weight_index], low_val, high_val)

                        nets_to_change[positive_weight] = pos_map
                        nets_to_change[negative_weight] = neg_map

                        if weights[weight_index] == 0 and (name != 'bias_1'):
                            zero_weight_nets[positive_weight] = 1
                            zero_weight_nets[negative_weight] = 1

                        # Make sure to log everything
                        randomized_values[current_sim_runs].append((name, weights[weight_index]))

                        weight_index +=1
                    else:
                        print(f"ERROR in generating weights. Got Option: {opt_flag}, but should be binary 'b'")
                        exit(1)
                else:
                    print(f"Error: Encountered something other than weight as another parameter: {name}")
                    exit(1)

            # Write into outfile the new parameters that were changed :)
            change_SPICE_param_in_file(infile, outfile, nets_to_change, zero_weight_nets)

            # Write Vdd, VSS
            write_voltage(outfile, 'vdd', 'vdd', 0, VDD)
            write_voltage(outfile, 'vss', 'vss', 0, VSS)

            # Go through voltage parameters
            for name, low_range, high_range, opt_flag in v_params:
                net_name = name.split('_', 1)[-1]

                if opt_flag == 'c':
                    netlist_voltage = random.uniform(low_range, high_range)
                else:
                    netlist_voltage = np.random.choice([low_range, high_range], size=1, p=[1/2, 1/2])[0]

                # Log into the output
                randomized_values[current_sim_runs].append((name, round(netlist_voltage, 3)))

                write_voltage(outfile, name, net_name, VSS, netlist_voltage)

            # Load in the PWL files
            for (net_name, net) in zip(INPUT_NET_NAME, INPUT_NET):
                full_pwl_filename_spice = os.path.join("../pwl_files", pwl_file_template.format(current_sim_runs, net_name))
                write_input_spike_file(outfile, net_name, f'vdd_2_{net}', net, full_pwl_filename_spice, VDD, simulator=SIMULATOR, write_voltage_src=INPUT_SEPARATE_VDD, current_src=INPUT_CURRENT_SRC)


        # Add load capacitance
        if SPIKING_INPUT:
            circuit_fan_out = random.randint(CIRCUIT_FAN_OUT_RANGE[1], CIRCUIT_FAN_OUT_RANGE[2])
            randomized_values[current_sim_runs].append((CIRCUIT_FAN_OUT_RANGE[0], circuit_fan_out))
        else:
            circuit_fan_out = 1
        

        output_cap = circuit_fan_out * LOAD_CAPACITANCE * 10**15      # in femtofarads 
        write_capacitance(outfile, OUTPUT_CAPACITANCE_NAME, OUTPUT_LOAD_CAP_NET, 0, output_cap)

        # Add subcircuit definition
        write_line_with_newline(outfile, SUBCIRCUIT_DEFINITION)

        # Add Simulation
        analysis = f".tran {SIM_MIN_STEP_SIZE_NS}ns {TOTAL_TIME_NS}ns 0 1ns"
        write_line_with_newline(outfile, analysis)

        # Add sim specific things
        write_sim_specific_simulation_information(outfile, NUMBER_OF_INPUTS, simulator=SIMULATOR)

        # Add end
        write_line_with_newline(outfile, ".end")
    
    # Increment 
    runs.append(output_fp)
    current_sim_runs+=1
    
# --------
# Run Simulations
run_simulation(runs, num_processes=NUM_PROCESSES, simulator=SIMULATOR)

# --------
# Log everything from simulations and run generation

# Write circuit_fanout to log folder
randomized_output_parameters_fd = os.path.join(run_directory, "randomized_output_log.csv")

# Error check that fanout contains 100
if current_sim_runs != NUMBER_OF_RUNS:
    print("Error occurred. Number of fanout numbers not equal to number of runs!, {} / {}".format(current_sim_runs, NUMBER_OF_RUNS))

# Write input parameter logs
with open(randomized_output_parameters_fd, 'w') as f:
    # Get header
    header = "Run_Number"
    for tuples in randomized_values[0]:
        header += "," + tuples[0]
    header += "\n"

    f.write(header)

    for key, value in sorted(randomized_values.items()):
        row_to_write = str(key)
        for tuples in value:
            row_to_write += "," + str(tuples[1])
        row_to_write += "\n"

        f.write(row_to_write)

# Write hyperparameter log file to the run directory
hyperparameter_fd = os.path.join(run_directory, "hyperparameters.txt")

if SPIKING_INPUT:
    randomized_params_ranges = [CIRCUIT_SIM_NUM_SPIKES, CIRCUIT_FAN_IN_RANGE, CIRCUIT_FAN_OUT_RANGE] + KNOB_PARAMS
else:
    randomized_params_ranges = KNOB_PARAMS

with open(hyperparameter_fd, "w") as f:
    # Run Hyperparameters
    f.write(f"RUN_NAME: {RUN_NAME}\n")
    f.write(f"NUMBER_OF_RUNS: {NUMBER_OF_RUNS}\n")
    f.write(f"NUMBER_OF_PROCESSES: {NUM_PROCESSES}\n")
    f.write(f"SIMULATOR: {sim_str}\n")
    f.write(f"SPIKING_INPUT: {SPIKING_INPUT}\n")
    f.write(f"CIRCUIT_STATEFUL: {CIRCUIT_STATEFUL}\n")
    f.write(f"TOTAL_TIME_NS: {TOTAL_TIME_NS}\n")
    f.write(f"SIM_MIN_STEP_SIZE_NS: {SIM_MIN_STEP_SIZE_NS}\n")
    f.write(f"DIGITAL_INPUT_ENFORCEMENT: {DIGITAL_INPUT_ENFORCEMENT}\n")
    f.write(f"DIGITAL_FREQUENCY: {DIGITAL_FREQUENCY}\n")
    f.write(f"LOAD_CAPACITANCE: {LOAD_CAPACITANCE}\n")
    f.write(f"INPUT_SEPARATE_VDD: {INPUT_SEPARATE_VDD}\n")
    f.write(f"DELAY_RATIO: {DELAY_RATIO}\n")
    f.write(f"SAME_SIGN_WEIGHTS_FRACTION: {SAME_SIGN_WEIGHTS_FRACTION}\n")
    
    # Spiking input hyperparameters
    if SPIKING_INPUT:
        f.write(f"NUM_INPUT_POINTS: {NUM_INPUT_POINTS}\n")
        f.write(f"REFRACTORY_PERIOD: {REFRACTORY_PERIOD}\n")
        f.write(f"RUN_TIMEOUT: {RUN_TIMEOUT}\n")
        f.write(f"SPICE_FOOTPRINT_FILE: {SPICE_FOOTPRINT_FILE}\n")
        f.write(f"FOOTPRINT_CHARGE: {footprint_charge}\n")
        f.write(f"SPIKE_START: {SPIKE_START}\n")
        f.write(f"SPIKE_END: {SPIKE_END}\n")
        f.write(f"OUTPUT_SPIKE_NAME: {OUTPUT_SPIKE_NAME}\n")
        f.write(f"CUSTOM_SPIKE: {CUSTOM_SPIKE}\n")
        f.write(f"CUSTOM_SPIKE_NUM_TIME_STEP_LENGTH: {CUSTOM_SPIKE_NUM_TIME_STEP_LENGTH}\n")

    # Imported files for runs :)
    f.write(f"MODEL_FILEPATH: {MODEL_FILEPATH}\n")
    f.write(f"LIBRARY_FILEPATH: {LIBRARY_FILES}\n")
    f.write(f"OTHER_NECESSARY_FILES: {OTHER_NECESSARY_FILES_IN_SPICE_RUN_DIRECTORY}\n")

    # I/O
    f.write(f"INPUT_NAMES: {str(INPUT_NET_NAME)}\n")
    f.write(f"INPUT_NET: {INPUT_NET}\n")
    f.write(f"OUTPUT_CAPACITANCE_NAME: {OUTPUT_CAPACITANCE_NAME}\n")

    for label,a,b,opt_flag in randomized_params_ranges:
        f.write(f"{label}: ({a}, {b})\n")

print("Hyperparameters have been saved to:", hyperparameter_fd)
