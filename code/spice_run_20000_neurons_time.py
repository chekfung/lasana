from PyLTSpice import SimCommander, RawRead
import os
import numpy as np
from create_spikes import generate_spike_map, generate_output_vector, create_pwl_file, analyze_spike_file, generate_spike_map_digital
from collections import defaultdict
import shutil
import time
import pandas as pd
from tools_helper import *

# ======================
'''
This file is solely used to characterize an analog neuron implemented in SPICE. Firstly, it requires that we 
hand characterize one spike output of a SPICE simulation so that it can be used to generate the spike footprint for
all input spikes in the future. Then, we use this spike footprint to either create a PWM-based spike, similar to the
spikes found in Brainscales-2 that take digital spikes, convert them to analog spikes, and then compute in analog, while
sending back output spike information for a particular neuron digitally.

This is a research project funded under the SLAM lab at UT Austin
Author: Jason Ho

'''
# Hyperparameters
RUN_NAME = "20000_runs_spice"
PWL_FILE_RUNS = "spiking_20000_runs"           # Specifies where we want to extract the PWL files from 
NUM_PROCESSES = 1                                   # Number of maximum number of threads to run 

NUMBER_OF_NEURONS_PER_RUN = [10, 100, 1000, 3000, 5000, 20000]
NUMBER_OF_RUNS = len(NUMBER_OF_NEURONS_PER_RUN)
SIMULATOR = 'spectre'
print("Number of Runs to Execute:{}".format(NUMBER_OF_RUNS))

# Model SPICE Filepath
# Neuron Model File Locations
model_filepath = '../data/spiking_neuron_spice_files/analog_lif_neuron.sp'
library_filepath = '../data/spiking_neuron_spice_files/libraries/45nm_LP.pm'


# end Hyperparameters
# -----------------------------------
# Input Layer Randomized Input
pwl_file_template = PWL_FILE_RUNS + "_" + "pwl_file_{}_I_inj.txt"

# ---------
# File IO to create directory structure for run
run_directory = os.path.join('../data', RUN_NAME)
libraries_directory = os.path.join(run_directory, 'libraries')
spice_run_directory = os.path.join(run_directory, 'spice_runs')

# Change, Extract PWL files from other run :O
pwl_file_run_directory = os.path.join('../data', PWL_FILE_RUNS)
pwl_file_main_directory = os.path.join(pwl_file_run_directory, "pwl_files")

# Get hyperparameters for the guy that we are running from
pwl_run_randomized_params = os.path.join(pwl_file_run_directory, "randomized_output_log.csv")
df = pd.read_csv(pwl_run_randomized_params)
df = df.set_index("Run_Number", drop=False)

if not os.path.exists(pwl_file_run_directory):
    print(f"ERROR: PWL file run directory DOES NOT EXIST!")
    exit(1)

HYPERPARAMETERS_FILEPATH = os.path.join(pwl_file_run_directory, "hyperparameters.txt")

hyperparameters_dict = {}

with open(HYPERPARAMETERS_FILEPATH, 'r') as file:
    for line in file:
        key, value = line.strip().split(': ')
        # Convert value to appropriate type if needed
        if value.isdigit():
            value = int(value)
        elif '.' in value and value.replace('.', '').isdigit():
            value = float(value)
        elif value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        hyperparameters_dict[key] = value

# Sanity Check

if max(NUMBER_OF_NEURONS_PER_RUN) > hyperparameters_dict["NUMBER_OF_RUNS"]:
    print("Number of PWL Files available is less than number you want to simulate! 💀 Available: {} Used: {}".format(hyperparameters_dict["NUMBER_OF_RUNS"], max(NUMBER_OF_NEURONS_PER_RUN)))
    exit(1)


# ---------------- DON'T TOUCH ------------------------ 
TOTAL_TIME_NS = hyperparameters_dict["TOTAL_TIME_NS"]                             # Nanoseconds of runtime        # Need this to be the same as the PWL files that we extract from!
number_input_points = hyperparameters_dict["NUM_INPUT_POINTS"]                      # Get sampling frequency, out of this, but only used for spike generation fidelity as well as how big PWL file is :)
REFRACTORY_PERIOD = float(hyperparameters_dict["REFRACTORY_PERIOD"])               # seconds   ; Time before another spike can occur on the same neuron 
NEURON_CAPACITANCE = float(hyperparameters_dict["LOAD_CAPACITANCE"])            # Farads    ;

OTHER_RANDOM_VOLTAGE_VALUES = [("V_sf", 0.5, 1.5), ("V_adap", 0.5, 1.5), ("V_leak", 0.5, 1.5), ("V_rtr", 0.5, 1.5)]  # NOTE: Default values for all of these knobs are specified in the design. We are simply changing them here.

# ---------------- END DON'T TOUCH ------------------------ 

# Copy model file into logging directory (such that runs occur in that directory as well)
filename = os.path.basename(model_filepath)
local_spice_model_filepath = os.path.join(spice_run_directory, filename)

# Copy over SPICE file
if not os.path.exists(spice_run_directory):
    os.makedirs(spice_run_directory)

shutil.copyfile(model_filepath, local_spice_model_filepath)

# Copy over Library File
if not os.path.exists(libraries_directory):
    os.makedirs(libraries_directory)

library_filename = os.path.basename(library_filepath)
lib_destination_file = os.path.join(libraries_directory, library_filename)

shutil.copyfile(library_filepath, lib_destination_file)
# ---------

# Start of script
current_runs = 0

# Formatted strings for easily writing :)
voltage_knob_format = "{}{} {}{} 0 {}\n" 
voltage_power_rail_format = "Vdd{} vdd{} 0 1.5\n"
fanout_load_cap_format = "Cout{} spk{} 0 {}fF IC=0V\n"
model_format = "X{} {} leak{} sf{} rtr{} adap{} spk{} vdd{} 0 lif_neuron\n"
output_pwl_spice_format = "I_inj{} vdd_input{} spikes{} PWL PWLFILE={}\n"  

current_sim_runs = 0
current_sim_files = []

# Create files that we will run :)
while (current_sim_runs < NUMBER_OF_RUNS):
    # For each run, determine number of neurons to run
    NUMBER_OF_NEURONS = NUMBER_OF_NEURONS_PER_RUN[current_sim_runs]

    # Copy over the model file that we will use
    # Get filepath for output file based on current_sim_runs
    fn, fn_ext = os.path.splitext(os.path.basename(local_spice_model_filepath))
    new_filename = f"{fn}_{NUMBER_OF_NEURONS}{fn_ext}"
    output_fp = os.path.join(os.path.dirname(local_spice_model_filepath), new_filename)

    with open(local_spice_model_filepath, 'r') as infile, open(output_fp, 'w') as outfile:
        for line in infile:
            outfile.write(line)

        for neuron_id in range(NUMBER_OF_NEURONS):
            for constrained_tuple in OTHER_RANDOM_VOLTAGE_VALUES:
                netlist_name = constrained_tuple[0]
                netlist_voltage = df.iloc[neuron_id][netlist_name]

                # Write into the netlist
                string_to_write = voltage_knob_format.format(netlist_name, neuron_id, netlist_name.split('_', 1)[1], neuron_id, netlist_voltage)
                outfile.write(string_to_write)
            
            # Declare Power Rail for neuron
            neuron_power_rail = voltage_power_rail_format.format(neuron_id, neuron_id)
            outfile.write(neuron_power_rail)

            # Add PWL file for first input and power rail for input
            voltage_rail_input_name = "_input{}".format(neuron_id)
            input_power_rail = voltage_power_rail_format.format(voltage_rail_input_name, voltage_rail_input_name)
            outfile.write(input_power_rail)

            pwl_file_directory = os.path.join("../data/", PWL_FILE_RUNS, "pwl_files")
            full_pwl_filename_spice = os.path.join(pwl_file_directory, pwl_file_template.format(neuron_id))
            outfile.write(output_pwl_spice_format.format(neuron_id, neuron_id, neuron_id, full_pwl_filename_spice))

            # Declare Out Cap
            circuit_fan_out = df.iloc[neuron_id]["Circuit_Fan_Out"]
            output_cap = circuit_fan_out * NEURON_CAPACITANCE * 10**15      # in femtofarads
            output_cap_line = fanout_load_cap_format.format(neuron_id, neuron_id, output_cap)
            outfile.write(output_cap_line)

            # Declare Model
            model_string = model_format.format(neuron_id, "spikes{}".format(neuron_id), neuron_id, neuron_id, neuron_id, neuron_id, neuron_id, neuron_id)
            outfile.write(model_string)

        # Write simulator specific things
        # Add Other Instructions
        runtime = ".tran 0.01ns {}ns\n".format(TOTAL_TIME_NS)
        outfile.write(runtime)

        write_sim_specific_simulation_information(outfile, 1, simulator=SIMULATOR)

        write_line_with_newline(outfile, '.end')

    current_sim_files.append(output_fp) 
    current_sim_runs +=1


# Run all the files consecutively
actual_runs = 0
run_times = {}

for i in range(NUMBER_OF_RUNS):
    run_file = current_sim_files[i]
    print(run_file)
    # Theoretically should wait to finish
    start = time.time()
    run_simulation_one_file(run_file, simulator=SIMULATOR)
    end = time.time()

    total_run_time = end - start
    run_times[NUMBER_OF_NEURONS_PER_RUN[i]] = total_run_time
    print(f"Num Neurons: {NUMBER_OF_NEURONS_PER_RUN[i]}, Runtime: {total_run_time} s")

    # Delete logs
    base_name = os.path.splitext(os.path.basename(run_file))[0]
    log_folder = os.path.join(spice_run_directory, base_name + ".raw")

    try:
        shutil.rmtree(log_folder)  # Only works for empty directories
        print(f"Folder '{log_folder}' deleted successfully.")
    except OSError as e:
        print(f"Error: {e.strerror}")


print(run_times)