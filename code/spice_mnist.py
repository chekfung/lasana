import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import shutil
import sys

# JH created helper files for generating datasets
from create_spikes import *
from tools_helper import *
from SpikeData import *

# For analyzing and understanding SPICE output
from stat_helpers import min_max_normalization, normalize
from scipy import signal
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
figure_counter = 0

# Loading MNIST Dataset through PyTorch 
import torch
import dill
import torchvision
import torchvision.transforms as transforms
import time

# Make sure all seeds are the same
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

'''
spice_mnist.py is a file created to run through the entire MNIST test set as a comparison to the trained LASANA models.
Essentially, it takes the neuron subcircuit and the MNIST inputs, converts the MNIST-based rate-based encoded inputs, and
then converts them to SPICE files given a spike footprint (same one used to train the LASANA models), and then runs through
the entire SPICE simulation, layer by layer, while emulating the digital backend in python.

For instance, for a 3-layer fully connected, feed-forward SNN (same as the one in /LASANA_APPLICATIONS/run_mnist_lasana.py),
of [784, 128, 10], we would first generate the initial spike generation from the MNIST input by rate in the first layer,
construct the SPICE file to run that, analyze the output to get spiking and energy information, and take the weights of the network
and then create the next spike train to the next layer. The next layer would then be generated and run in SPICE, and then so on.
Once the last layer is reached we will take the spike outputs and generate the maxpool of spikes to determine what spiked, and
compare that against the label.

For simplicity sake, we will only support a batch of 2, due to RAM limitations

Also note, on my machine, (i7-13700, 32GB Ram) it took 2404 hours to run the entire script, so it would be better to 
run this on multiple machines, while splitting the number of images in parallel. One can change the image_start_offset and
then go from there. 

The logs for this are also impossiblly large (multiple TB), so note before running.
'''
# ----------
# START HYPERPARAMETERS
RUN_NAME = 'test_spiking_mnist_golden_results'
NUMBER_OF_IMAGES = 10000                          # Arbitrarily set the number of runs
IMAGE_START_OFFSET = 0
NUMBER_OF_TIMESTEPS_PER_INFERENCE = 100
DIGITAL_TIMESTEP = 5 * 10**-9
SIM_MIN_STEP_SIZE_NS = 0.01                     # Min Step Size nanoseconds of the simulation 
BATCH_SIZE = 2              # Seems like we are limited to 2 since RAM limited 
NUM_PROCESSES = min(BATCH_SIZE, 10)             # Max num processes at 10 , but otherwise set as batch_size
MAX_IMAGE_ID = NUMBER_OF_IMAGES + IMAGE_START_OFFSET -1 
RUN_SPICE_SIMULATION = True
PLOT_RUNS = False

# Pytorch MNIST information
PYTORCH_MODEL_FILE = '../data/spiking_mnist_model.pt'
WEIGHT_SCALING = 1                         # Scaling of weights (to account for leak) [NOTE: Does not include first layer (as there are no weights)]
INPUT_SCALING = 2                          # Scaling of inputs between layers (note that input scaling implicitly affects weight scaling since weight scaling is applied first, then input scaling)

# Calculated Hyperparams
TOTAL_TIME_NS = DIGITAL_TIMESTEP * NUMBER_OF_TIMESTEPS_PER_INFERENCE * 10**9

NUM_INPUT_POINTS = TOTAL_TIME_NS*10                         # Get sampling frequency, out of this, but only used for spike generation fidelity as well as how big PWL file is 
POINTS_PER_DIGITAL_TIMESTEP = int(NUM_INPUT_POINTS / NUMBER_OF_TIMESTEPS_PER_INFERENCE)
INPUT_SEPARATE_VDD = True
SIMULATOR = 'spectre'
VDD = 1.5          
VSS = 0
SPIKING_INPUT = True                            # Determines if there is spiking input 
NUMBER_OF_INPUTS = 1
NUMBER_OF_WEIGHTS = 1
LOAD_CAPACITANCE = 500 * 10**(-15)                                # Farads    

# I/O 
INPUT_NET_NAME = ['I_inj']                                        # Name of current input (s) (Requires I in front)
INPUT_NET = ['spikes']                                            # Name of interconnected net (s)
KNOB_PARAMS = {"V_sf": 0.5, "V_adap": 0.5, "V_leak": 0.57, "V_rtr": 0.5}

# For example, Cout, spk -> Cout spk VSS xF     (Cout is the name, spk is the pos connection, VSS is the neg connection, and xF is the capacitance)
OUTPUT_CAPACITANCE_NAME = 'Cout'                                  # Capacitance Name (Requires C in front) Corresponds to input nets 
OUTPUT_LOAD_CAP_NET = 'spk'                                       # Name of interconnected load net Corresponds to input nets 

# Circuit Definition to run
SUBCIRCUIT_FORMAT = 'X{} {} {} {} {} {} {} {} 0 lif_neuron'
# Model SPICE Filepath
# Neuron Model File Locations
MODEL_FILEPATH = '../data/spiking_neuron_spice_files/analog_lif_neuron.sp'
OTHER_NECESSARY_FILES_IN_SPICE_RUN_DIRECTORY = []
LIBRARY_FILES = ['../data/spiking_neuron_spice_files/libraries/45nm_LP.pm']
OUTPUT_SPIKE_NAME = 'i(C)'            # This is for the footprint 

PLOT_SPIKE_BOUNDS = False

if SIMULATOR == 'spectre':
    SPICE_FOOTPRINT_FILE = '../data/spiking_neuron_spice_files/analog_lif_neuron_footprint_run.sp'
    SPIKE_START = 306
    SPIKE_END = 361

# END OF HYPERPARAMETERS
# -----------------
# Create all necessary I/O filepath 

# Simulator detection
sim_str = idiot_proof_sim_string(SIMULATOR)
assert(is_simulator_real(sim_str))

PWL_FILE_TEMPLATE = RUN_NAME + "_" + "pwl_file_img{}_l{}_n{}.txt"       # First one referes to run number, second is input net connection 

# File IO to create directory structure for run
RUN_DIRECTORY = os.path.join('../data', RUN_NAME)
CSV_DIRECTORY = os.path.join(RUN_DIRECTORY, 'event_csvs')
LIBRARIES_DIRCTORY = os.path.join(RUN_DIRECTORY, 'libraries')
PWL_FILE_MAIN_DIRECTORY = os.path.join(RUN_DIRECTORY, "pwl_files")
SPICE_RUN_DIRECTORY = os.path.join(RUN_DIRECTORY, 'spice_runs')

# Make necessary directories by using the most nested directory
if not os.path.exists(PWL_FILE_MAIN_DIRECTORY):
    os.makedirs(PWL_FILE_MAIN_DIRECTORY)
    print(f"Directory '{PWL_FILE_MAIN_DIRECTORY}' created successfully.")

# Copy model file into logging directory (such that runs occur in that directory as well)
FILENAME = os.path.basename(MODEL_FILEPATH)
LOCAL_SPICE_MODEL_FILEPATH = os.path.join(SPICE_RUN_DIRECTORY, FILENAME)

# Make directories 
if not os.path.exists(CSV_DIRECTORY):
    os.makedirs(CSV_DIRECTORY)

# Copy over SPICE file
if not os.path.exists(SPICE_RUN_DIRECTORY):
    os.makedirs(SPICE_RUN_DIRECTORY)

shutil.copyfile(MODEL_FILEPATH, LOCAL_SPICE_MODEL_FILEPATH)

# Copy over other files into SPICE_RUN_DIRECTORY
if OTHER_NECESSARY_FILES_IN_SPICE_RUN_DIRECTORY:
    for necessary_file in OTHER_NECESSARY_FILES_IN_SPICE_RUN_DIRECTORY:
        shutil.copyfile(necessary_file, os.path.join(SPICE_RUN_DIRECTORY, os.path.basename(necessary_file)))

# Copy over Library File
if not os.path.exists(LIBRARIES_DIRCTORY):
    os.makedirs(LIBRARIES_DIRCTORY)

for library_filepath in LIBRARY_FILES:
    library_FILENAME = os.path.basename(library_filepath)
    lib_destination_file = os.path.join(LIBRARIES_DIRCTORY, library_FILENAME)

    shutil.copyfile(library_filepath, lib_destination_file)

# ----------------
# Get spike footprint 
total_sim_time_s = TOTAL_TIME_NS * 10**(-9)
sampling_period = total_sim_time_s / NUM_INPUT_POINTS
sampling_frequency = 1 / sampling_period
SPIKE_FOOTPRINT =  analyze_spike_file(SPICE_FOOTPRINT_FILE, OUTPUT_SPIKE_NAME, SPIKE_START, SPIKE_END, sampling_period, PLOT_SPIKE_BOUNDS, simulator=sim_str)
time_vector = np.linspace(0, total_sim_time_s, int(NUM_INPUT_POINTS))
FOOTPRINT_LEN = SPIKE_FOOTPRINT.shape[0]
FOOTPRINT_CHARGE = np.trapz(SPIKE_FOOTPRINT, time_vector[:SPIKE_FOOTPRINT.shape[0]])

print(f"Total Charge of Spike, weight 1: {FOOTPRINT_CHARGE}".format())
print("Peak Amplitude of One Guy: {}".format(np.max(SPIKE_FOOTPRINT)))
print("NUMBER OF SPIKE POINTS: {}".format(SPIKE_FOOTPRINT.shape[0]))

# Load MNIST Dictionary and Model Weights 
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
subset_testset = torch.utils.data.Subset(testset, range(IMAGE_START_OFFSET, min(len(testset), MAX_IMAGE_ID+1)))
testloader = torch.utils.data.DataLoader(subset_testset, batch_size=BATCH_SIZE, shuffle=False)
num_batches = len(testloader)

# Load in Model weights for Pytorch
mnist_model = torch.load(PYTORCH_MODEL_FILE,pickle_module=dill,map_location=torch.device("cpu"))
model_weights = {}      # keys are "fc1.weight", and fc2.weight
for param_name, param in mnist_model.named_parameters():
    detached = param.detach().numpy()
    model_weights[param_name] = detached
    print(f"Model Weight: {param_name} Shape: {detached.shape}")


# ------- END HYPERPARAMETERS -------------

# Important helper function 
def poisson_spike_train(image, timesteps):
    """
    Converts an image batch into Poisson spike trains.
    Args:
        image: Input image tensor (batch, 1, N, N), normalized
        timesteps: Number of timesteps for spiking simulation
    Returns:
        Poisson spike train (timesteps, batch, N**2)
    """
    batch_size = image.shape[0]
    image = image.view(image.shape[0], -1)  # Flatten (batch, 784)
    image = torch.clamp(image, 0, 1)  # Ensure values are in [0,1]
    spike_train = torch.rand(timesteps, *image.shape) < image  # Poisson spikes

    # Reshape to (timesteps * batch, 784)
    spike_train = spike_train.view(timesteps * batch_size, -1).float()

    return spike_train.float()

# Supporting code for layer generation given that we know what all the inputs are 
def create_layer_file(num_inputs, image_id, layer_id, spice_model_fd, PWL_FILE_TEMPLATE, subcircuit_template, KNOB_PARAMS, VDD, VSS, SIMULATOR='spectre'):
    # Get filepath for output file based on current_sim_runs
    fn, fn_ext = os.path.splitext(os.path.basename(spice_model_fd))
    new_FILENAME = f"{fn}_{image_id}_layer{layer_id}{fn_ext}"
    output_fp = os.path.join(os.path.dirname(spice_model_fd), new_FILENAME)
    
    with open(spice_model_fd, 'r') as infile, open(output_fp, 'w') as outfile:
        # Copy over subcircuit
        for line in infile:
            outfile.write(line)

        # Setup voltages and currents to probe 
        # NOTE: Voltage probes use NET NAME; current probes use COMPONENT NAME 
        # In the future, I should definitely make it easier to do this 
        voltage_probes = []
        current_probes = []

        for neuron_idx in range(num_inputs):
            # Create Separate Vdd
            # Note: actual net name is vdd_id so like vdd_1

            write_voltage(outfile, f'Vdd_{neuron_idx}', f'vdd_{neuron_idx}', 0, VDD)
            voltage_probes.append(f'v(vdd_{neuron_idx})')
            current_probes.append(f'i(Vdd_{neuron_idx})')

            # Create Knobs (to make sure every neuron is separate )
            # Note: actual net_name is like V_leak -> leak_1
            for k in KNOB_PARAMS:
                netlist_name = k + f"_{neuron_idx}"
                net_name = k.split('_', 1)[-1] + f"_{neuron_idx}"
                netlist_voltage = KNOB_PARAMS[k]
                write_voltage(outfile, netlist_name, net_name, VSS, netlist_voltage)

            # Create Input PWL 
            current_src_name = f'I_inj_{neuron_idx}'
            full_pwl_FILENAME_spice = os.path.join("../pwl_files", PWL_FILE_TEMPLATE.format(image_id, layer_id, neuron_idx))
            write_input_spike_file(outfile, current_src_name, f'vdd_2_{neuron_idx}', f'input_{neuron_idx}', full_pwl_FILENAME_spice, VDD, simulator=SIMULATOR, write_voltage_src=True)
            current_probes.append(f'i({current_src_name})')

            # Create Output Circuit Cap
            output_cap = 1 * LOAD_CAPACITANCE * 10**15      # in femtofarads 
            write_capacitance(outfile, f'Cout_{neuron_idx}', f'out_{neuron_idx}', 0, output_cap)
            current_probes.append(f'i(Cout_{neuron_idx})')

            # Create Subcircuit Definition
            subcircuit_str = subcircuit_template.format(neuron_idx, f'input_{neuron_idx}', f"leak_{neuron_idx}", 
                                                        f"sf_{neuron_idx}", f"rtr_{neuron_idx}", f'adap_{neuron_idx}', 
                                                        f'out_{neuron_idx}', f'vdd_{neuron_idx}')
            write_line_with_newline(outfile, subcircuit_str)
            voltage_probes.append(f"v(input_{neuron_idx})")

        # Add Simulation Transient Analysis
        analysis = f".tran {SIM_MIN_STEP_SIZE_NS}ns {TOTAL_TIME_NS}ns 0 1ns"
        write_line_with_newline(outfile, analysis)

        # Add sim Specific Information
        write_sim_specific_simulation_information(outfile, num_inputs, probe_all_nets=False, 
                                                  voltage_nets=voltage_probes,current_nets=current_probes,
                                                  simulator=SIMULATOR)

        # Add end
        write_line_with_newline(outfile, ".end")

    return output_fp

def generate_fully_connected_spike_train(layer_num_neurons, NUMBER_OF_TIMESTEPS_PER_INFERENCE, spike_data_df, weights, WEIGHT_SCALING, INPUT_SCALING):
    layer_spiketrain = np.zeros((NUMBER_OF_TIMESTEPS_PER_INFERENCE, layer_num_neurons))

    # Provides per timestep dict of which neurons spiked 
    spikes_per_timestep = spike_data_df.groupby('Digital_Time_Step')['Neuron_Num'].apply(list).to_dict()

    for t in spikes_per_timestep:
        # Go through each timestep and determine what maps to what
        neuron_ids = spikes_per_timestep[t]

        for id in neuron_ids:
            # Go through each neuron that spiked and add for next timestep 
            if t+1 < NUMBER_OF_TIMESTEPS_PER_INFERENCE:
                layer_spiketrain[t+1,:] += (weights[:,id] * WEIGHT_SCALING)
        
    # Apply Input Scaling 
    layer_spiketrain = layer_spiketrain * INPUT_SCALING
    return layer_spiketrain

# Currently supported only with the Indiveri 2003 Neuron 
def analyze_spike_information_mnist(image_num, layer_num, neuron_num, number_of_timesteps, one_time_period, FOOTPRINT_CHARGE, param_knobs, raw_time, raw_input_spikes, raw_circuit_state, raw_output_spikes, raw_vdd, raw_vdd_current, VERBOSE=False, PLOT_RUNS=False, INPUT_WELL_DEFINED=True, SIMULATOR='spectre'):
    OUTPUT_SPIKE_THRESHOLD = 1 * 10**-6      # use this value to determine if a trace has any spikes (i.e. if nothing above 1 uA, then no spike in whole thing)
    global figure_counter 
    # Events list for everything here
    events_list = []
    
    #print(f"Analyzing Image number: {image_num}, Layer: {layer_num}, Neuron: {neuron_num}")

    if SIMULATOR == 'ltspice':
        # Need to mask out negative time for some reason
        time_vec = mask_negative_time(raw_time, raw_time)
    else:
        time_vec = raw_time

    raw_instantaneous_power = np.abs(raw_vdd * raw_vdd_current)

    if SIMULATOR == 'ltspice':
        input_spikes = mask_negative_time(raw_time, raw_input_spikes)
        circuit_state = mask_negative_time(raw_time, raw_circuit_state)
        output_spikes = mask_negative_time(raw_time, raw_output_spikes)
        instantaneous_power = mask_negative_time(raw_time, raw_instantaneous_power)
    else:
        input_spikes = raw_input_spikes
        circuit_state = raw_circuit_state 
        output_spikes = raw_output_spikes
        instantaneous_power = raw_instantaneous_power

    # ----

    # Get gradients (first derivative)
    output_gradients = np.gradient(output_spikes, time_vec)

    out_mask = output_spikes > OUTPUT_SPIKE_THRESHOLD

    # Remove Drift 
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
    #print(f"Output Mask Filter Val: {normalized_filtered_output[0] * 3}")
    output_mask = (normalized_filtered_output > normalized_filtered_output[0] * 3)

    gradient_spike_mask = get_gradient_where_spike(output_spikes, output_gradients, time_vec)
    combine_all_masks = out_mask & gradient_spike_mask   

    #Plot everything right now with the timestep bounds 
    if PLOT_RUNS:
        plt.figure(figure_counter)
        figure_counter+=1
        plt.plot(time_vec*10**9,normalize(input_spikes), color='green', label='in')
        plt.plot(time_vec*10**9,normalize(output_spikes), color='red', label='out')
        plt.plot(time_vec*10**9,input_mask,color='purple', label='in_mask')
        plt.plot(time_vec*10**9, gradient_spike_mask, color='purple', label='gradient_mask', ls='--')
        plt.plot(time_vec*10**9, out_mask, color='blue', label='out_mask', ls='--')
        plt.plot(time_vec*10**9, combine_all_masks, color='orange', ls='--',label='out_spike')
        plt.plot(time_vec*10**9, output_mask, color='black', linewidth=2, label='out mask')
        plt.plot(time_vec*10**9, circuit_state, label='circuit_state', color='black', ls='--')
        plt.plot(time_vec*10**9,normalize(output_gradients), color='orange', label='output_grad')
        #plt.plot(time_vec*10**9, meow, color='black')
        plt.title(f"Run {i}")

    timesteps_without_events = 0

    # Ok, we are going to loop through each of the timesteps now 
    for j in range(number_of_timesteps):
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
        end_index = np.min([end_index, len(time_vec)-1])      # Join the end of the index with the start of the new guy 

        # Continue writing on that plot to show where these things are 
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

                timing_event["Image_Num"] = image_num
                timing_event["Layer_Num"] = layer_num
                timing_event["Neuron_Num"] = neuron_num
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
                for knob in param_knobs:
                    knob_value = param_knobs[knob]
                    timing_event[knob] = knob_value

                events_list.append(timing_event)

                if VERBOSE:
                    print(f"Timing Event {timing_event_start_index}:{timing_event_end_index}")
                
            # After processing the timing events, if it exists, now we can process the spike event.
            timesteps_without_events = 0

            # --------------- Process Spike Event Now ----------------
            if VERBOSE:
                print(f"Spike Event Timestep: {start_index}:{end_index}")

            # Get the peak index for the spike for latency calculations 
            spike_peak_index = np.argmax(np.abs(input_spikes[input_start_spike_index:input_end_spike_index+1])) + input_start_spike_index # Note: make sure to plus one to include input_end_spike_index in there

            plt.axvline(time_vec[input_start_spike_index]*10**9, color='blue')

            spike_event = {}

            # House Keeping
            spike_event["Image_Num"] = image_num
            spike_event["Layer_Num"] = layer_num
            spike_event["Neuron_Num"] = neuron_num
            spike_event["Event_Start_Index"] = start_index 
            spike_event["Event_End_Index"]  = end_index
            spike_event["Digital_Time_Step"] = j

            # Input Spike Details
            footprint_spike_charge = FOOTPRINT_CHARGE          # From generate_dataset.py
            spike_charge =  np.trapz(input_spikes[input_start_spike_index:input_end_spike_index+1], time_vec[input_start_spike_index:input_end_spike_index+1])
            spike_event["Input_Peak_Amplitude"] = input_spikes[spike_peak_index]
            spike_event["Input_Total_Charge"] = spike_charge
            spike_event["Input_Total_Time"] = (timesteps_without_events+1) * one_time_period   # Need to include all of 124 inside of 124, since no step 125 
            spike_event["Weight"] = spike_charge / footprint_spike_charge       

            # I/O
            spike_event["Cap_Voltage_At_Input_Start"] = circuit_state[start_index]
            spike_event["Cap_Voltage_At_Output_End"] = circuit_state[end_index]
            spike_event['Energy'] = np.trapz(instantaneous_power[start_index:end_index+1], time_vec[start_index:end_index+1])

            # Put in Knobs
            for knob in param_knobs:
                knob_value = param_knobs[knob]
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
                    event_type='in-no_out'
                    latency=0

                    print(f"WARNING: SPIKE SAME INDICES AT RUN: {i}, TIMESTEP: {j}")

                else:
                    # Anything else 
                    latency = time_vec[output_start_index] - time_vec[start_index]
                    event_type = 'in-out'

                plt.axvline(time_vec[output_start_index]*10**9, color='red')

            else:
                # No output spike, in-no_out
                latency = 0
                event_type = 'in-no_out'

            # These two are dependent on if we have a spike or not 
            spike_event["Event_Type"] = event_type
            spike_event['Latency'] = latency

            events_list.append(spike_event)

        else:
            # If spike not found, we need to check if this is the last timestep. If so, we need to make a timing event for this last timestep or more (significant leakage events)
            if j == number_of_timesteps - 1:
                # If last timestep, but no spike, we need to evaluate this as a timing event 
                timestep_since_last_spike = (j - timesteps_without_events) * one_time_period

                timing_event = {}
                timing_event_start_index = np.where(time_vec >= timestep_since_last_spike)[0][0]
                timing_event_end_index = end_index # Note: This is end index because there is no event, we process this timing event all the way to the end 
                if VERBOSE:
                    print(f"Timing Event Length: {len(time_vec)}, End Index: {end_index}")

                timing_event["Image_Num"] = image_num
                timing_event["Layer_Num"] = layer_num
                timing_event["Neuron_Num"] = neuron_num
                timing_event["Event_Start_Index"] = timing_event_start_index 
                timing_event["Event_End_Index"]  = timing_event_end_index
                timing_event["Event_Type"] = 'leak'
                timing_event["Digital_Time_Step"] = j
                timing_event["Input_Peak_Amplitude"] = 0
                timing_event["Input_Total_Charge"] = 0
                timing_event["Input_Total_Time"] = (timesteps_without_events+1) * one_time_period   # Need to include all of 124 inside of 124, since no step 125 
                timing_event["Weight"] = 0
                timing_event["Cap_Voltage_At_Input_Start"] = circuit_state[timing_event_start_index]
                timing_event["Cap_Voltage_At_Output_End"] = circuit_state[timing_event_end_index]
                timing_event['Latency'] = 0
                timing_event['Energy'] = np.trapz(instantaneous_power[timing_event_start_index:timing_event_end_index+1], time_vec[timing_event_start_index:timing_event_end_index+1])

                # Put in Knobs
                for knob in param_knobs:
                    knob_value = param_knobs[knob]
                    timing_event[knob] = knob_value

                events_list.append(timing_event)
            
            # if no spike, add one to the timesteps_without_events 
            timesteps_without_events+=1

    # At the end, show all the figures together
    if PLOT_RUNS:
        plt.legend()
        plt.show(block=False)
        plt.pause(0.001) # Pause for interval seconds.
        input("hit[enter] to end.")
        plt.close('all') # all open plots are correctly closed after each run

    # After the end of everything, put everything into a dataframe and export to csv
    event_columns = ["Image_Num", "Layer_Num", "Neuron_Num", "Event_Type", 'Event_Start_Index', 'Event_End_Index', 'Digital_Time_Step',"Input_Peak_Amplitude", "Input_Total_Charge", "Weight","Input_Total_Time", "Cap_Voltage_At_Input_Start", "Cap_Voltage_At_Output_End", "Energy", "Latency"]
    event_df = pd.DataFrame(events_list, columns=event_columns)

    # Some Stats if wanted for verbosity
    if VERBOSE:
        print(event_df.to_string())

        event_df = event_df.sort_values(by=['Neuron_Num', 'Digital_Time_Step', "Event_Type"], ascending=[True, True, False])

        # Print any crazies first 
        filtered_df = event_df[event_df.groupby(['Neuron_Num', 'Event_End_Index'])['Event_End_Index'].transform('count') > 1]
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

        print("Statistics")
        print(f"Number of Runs: {1}")
        print(f"CSV Number of Columns:{event_df.shape[1]}")
        print(f"Total Events: {event_df.shape[0]}")
        print(f"Number of Input Spike Events: {moo.shape[0]}")

        if moo.shape[0] == 0:
            # Get by division by zero error
            print(f"Number of inputs with Outputs: {woof.shape[0]} / {moo.shape[0]}")
        else:
            print(f"Number of inputs with Outputs: {woof.shape[0]} / {moo.shape[0]} ({woof.shape[0] / moo.shape[0] * 100:.2f}%)")
        print(f"Number of Timing Events: {meow.shape[0]}")

        print("\n")
        print("---------------- Timing Event Statistics ------------------")

        # Statistics of Latency Events
        leak_df = event_df[event_df['Event_Type'] == 'leak'].copy()  # .copy() to ensure we are working with a new dataframe
        leak_df.loc[:, 'Timestep'] = (leak_df['Input_Total_Time'] / one_time_period).astype(int)
        timestep_counts = leak_df['Timestep'].value_counts().sort_index()

        # Calculate the percentage of each timestep
        total_timesteps = len(leak_df)

        # Print each timestep count in the desired format
        for timestep, count in timestep_counts.items():
            percentage = (count / total_timesteps) * 100
            print(f"Latency Number of Timesteps: ({timestep}) {count} / {total_timesteps} ({percentage:.2f}%)")
        print("\n------------------------------------------------------\n")
    return event_df



def process_image(image_data):
    '''Processes an entire MNIST image in parallel for multiprocessing across batch '''
    try:
        iter_start = time.time()
        # First figure out what the inputs are
        image_id, image, label = image_data 

        # Creates Spike train of size (timesteps, neurons), in the default case (100, 784)
        spike_train = poisson_spike_train(image, NUMBER_OF_TIMESTEPS_PER_INFERENCE).numpy()

        # Convert from digital timesteps to real timesteps is 
        layer0_num_neurons = spike_train.shape[1]
        spike_train = spike_train[:, :layer0_num_neurons]
        spikemap = np.zeros((NUMBER_OF_TIMESTEPS_PER_INFERENCE * POINTS_PER_DIGITAL_TIMESTEP, layer0_num_neurons))
        spikemap[::POINTS_PER_DIGITAL_TIMESTEP, :] = spike_train

        # Create real vector spike train
        spike_indices = np.where(spikemap != 0)
        for i,j in zip(*spike_indices):
            spikemap[i:i+FOOTPRINT_LEN, j] = SPIKE_FOOTPRINT * INPUT_SCALING
        
        # Generate PWL Files for input image
        for i in range(layer0_num_neurons):
            pwl_file = os.path.join(PWL_FILE_MAIN_DIRECTORY, PWL_FILE_TEMPLATE.format(image_id, 0, i))
            create_pwl_file(pwl_file, time_vector, spikemap[:,i])

        print(f"Image: {image_id} / {MAX_IMAGE_ID}, Generated PWL Files for Input Layer")

        # Create first layer file
        layer0_file = create_layer_file(layer0_num_neurons, image_id, 0, LOCAL_SPICE_MODEL_FILEPATH, PWL_FILE_TEMPLATE, SUBCIRCUIT_FORMAT, KNOB_PARAMS, VDD, VSS)

        # Run First Layer Simulation
        start_time = time.time()
        if RUN_SPICE_SIMULATION:
            run_simulation_one_file(layer0_file, simulator=SIMULATOR)
        end_time = time.time()
        print(f"Image: {image_id}, Simulation Time of Layer0: {end_time - start_time}s") # Sim time was about 412 seconds for first layer

        # Analyze output for first layer
        start_time = time.time()

        raw_obj_layer0 = read_simulation_file(layer0_file, simulator=SIMULATOR)
        print(f"Image: {image_id}, Simulation File Read for Layer0")

        # Get time and I(inj), and I(C)
        time_vec = get_signal("time", raw_obj_layer0, simulator=SIMULATOR)

        layer0_neuron_datasets = []

        for i in range(layer0_num_neurons):
            # Get circuit raw information
            raw_input_current = get_signal(f"i(I_inj_{i})", raw_obj_layer0, simulator=SIMULATOR)
            raw_output_current = get_signal(f"i(Cout_{i})", raw_obj_layer0, simulator=SIMULATOR)
            raw_circuit_state = get_signal(f"v(input_{i})", raw_obj_layer0, simulator=SIMULATOR)
            raw_vdd = get_signal(f"v(vdd_{i})", raw_obj_layer0, simulator=SIMULATOR)
            raw_vdd_current = get_signal(f"i(Vdd_{i})", raw_obj_layer0, simulator=SIMULATOR)
            neuron_dataset = analyze_spike_information_mnist(image_id,0,i,NUMBER_OF_TIMESTEPS_PER_INFERENCE, 
                                                            DIGITAL_TIMESTEP, FOOTPRINT_CHARGE, KNOB_PARAMS, time_vec, raw_input_current,
                                                            raw_circuit_state, raw_output_current, raw_vdd, raw_vdd_current,
                                                            VERBOSE=False, PLOT_RUNS=False, SIMULATOR=SIMULATOR)
            layer0_neuron_datasets.append(neuron_dataset)
        
        # Concatenate Everything into one big pandas dataset
        layer0_events = pd.concat(layer0_neuron_datasets, axis=0, ignore_index=True)

        # Save events 
        layer0_events.to_csv(os.path.join(CSV_DIRECTORY,f"img{image_id}_layer{0}_events_dataset.csv"), index=False)

        end_time = time.time()
        print(f"Image: {image_id}, Simulation Time to read Layer0 Output: {end_time - start_time}")

        # Just get spike events
        spike_data_df = layer0_events[layer0_events['Event_Type'] == 'in-out']

        # ---------------- END OF LAYER 0 ---------------------

                        
        # Construct spike train for second layer yeah 
        layer1_num_neurons = 128
        layer1_spiketrain = generate_fully_connected_spike_train(layer1_num_neurons, NUMBER_OF_TIMESTEPS_PER_INFERENCE, spike_data_df, 
                                            model_weights['fc1.weight'], WEIGHT_SCALING, INPUT_SCALING)
        
        spikemap_layer1 = np.zeros((NUMBER_OF_TIMESTEPS_PER_INFERENCE * POINTS_PER_DIGITAL_TIMESTEP, layer1_num_neurons))
        spikemap_layer1[::POINTS_PER_DIGITAL_TIMESTEP, :] = layer1_spiketrain

        spike_indices = np.where(spikemap_layer1 != 0)
        for i,j in zip(*spike_indices):
            # Apply weight to spike footprint
            spikemap_layer1[i:i+FOOTPRINT_LEN, j] = SPIKE_FOOTPRINT * spikemap_layer1[i,j]

        # Generate PWL Files for spike trains
        for i in range(layer1_num_neurons):
            pwl_file = os.path.join(PWL_FILE_MAIN_DIRECTORY, PWL_FILE_TEMPLATE.format(image_id, 1, i))
            create_pwl_file(pwl_file, time_vector, spikemap_layer1[:,i])

        print(f"Image: {image_id} / {MAX_IMAGE_ID}, Generated PWL Files for Layer 1")

        # Create first layer file
        layer1_file = create_layer_file(layer1_num_neurons, image_id, 1, LOCAL_SPICE_MODEL_FILEPATH, PWL_FILE_TEMPLATE, SUBCIRCUIT_FORMAT, KNOB_PARAMS, VDD, VSS)

        # Run Simulation for second layer1
        start_time = time.time()
        if RUN_SPICE_SIMULATION:
            run_simulation_one_file(layer1_file, simulator=SIMULATOR)
        end_time = time.time()
        print(f"Image: {image_id}, Simulation Time of Layer1: {end_time - start_time}s") # Sim time was about 412 seconds for first layer

        # Read Simulation File for Layer1
        start_time = time.time()
        raw_obj_layer1 = read_simulation_file(layer1_file, simulator=SIMULATOR)
        print(f"Image: {image_id}, Simulation File Read for Layer1")

        # Get time and I(inj), and I(C)
        time_vec = get_signal("time", raw_obj_layer1, simulator=SIMULATOR)

        layer1_neuron_datasets = []

        for i in range(layer1_num_neurons):
            # Get circuit raw information
            raw_input_current = get_signal(f"i(I_inj_{i})", raw_obj_layer1, simulator=SIMULATOR)
            raw_output_current = get_signal(f"i(Cout_{i})", raw_obj_layer1, simulator=SIMULATOR)
            raw_circuit_state = get_signal(f"v(input_{i})", raw_obj_layer1, simulator=SIMULATOR)
            raw_vdd = get_signal(f"v(vdd_{i})", raw_obj_layer1, simulator=SIMULATOR)
            raw_vdd_current = get_signal(f"i(Vdd_{i})", raw_obj_layer1, simulator=SIMULATOR)
            neuron_dataset = analyze_spike_information_mnist(image_id,1,i,NUMBER_OF_TIMESTEPS_PER_INFERENCE, 
                                                            DIGITAL_TIMESTEP, FOOTPRINT_CHARGE, KNOB_PARAMS, time_vec, raw_input_current,
                                                            raw_circuit_state, raw_output_current, raw_vdd, raw_vdd_current,
                                                            VERBOSE=False, PLOT_RUNS=False, SIMULATOR=SIMULATOR)
            layer1_neuron_datasets.append(neuron_dataset)
        
        # Concatenate Everything into one big pandas dataset
        layer1_events = pd.concat(layer1_neuron_datasets, axis=0, ignore_index=True)
        layer1_events.to_csv(os.path.join(CSV_DIRECTORY,f"img{image_id}_layer{1}_events_dataset.csv"), index=False)

        end_time = time.time()
        print(f"Image: {image_id}, Simulation Time to read Layer1 Output: {end_time - start_time}")

        # Just get spike events
        spike_data_layer1_df = layer1_events[layer1_events['Event_Type'] == 'in-out']

        # ---------------- END OF LAYER 1 ---------------------
        # Construct spike train for third layer
        layer2_num_neurons = 10
        layer2_spiketrain = generate_fully_connected_spike_train(layer2_num_neurons, NUMBER_OF_TIMESTEPS_PER_INFERENCE, spike_data_layer1_df, 
                                            model_weights['fc2.weight'], WEIGHT_SCALING, INPUT_SCALING)
        
        spikemap_layer2 = np.zeros((NUMBER_OF_TIMESTEPS_PER_INFERENCE * POINTS_PER_DIGITAL_TIMESTEP, layer2_num_neurons))
        spikemap_layer2[::POINTS_PER_DIGITAL_TIMESTEP, :] = layer2_spiketrain

        spike_indices = np.where(spikemap_layer2 != 0)
        for i,j in zip(*spike_indices):
            # Apply weight to spike footprint
            spikemap_layer2[i:i+FOOTPRINT_LEN, j] = SPIKE_FOOTPRINT * spikemap_layer2[i,j]

        # Generate PWL Files for spike trains
        for i in range(layer2_num_neurons):
            pwl_file = os.path.join(PWL_FILE_MAIN_DIRECTORY, PWL_FILE_TEMPLATE.format(image_id, 2, i))
            create_pwl_file(pwl_file, time_vector, spikemap_layer2[:,i])

        print(f"Image: {image_id} / {MAX_IMAGE_ID}, Generated PWL Files for Layer 2")


        # Create Third Layer File
        layer2_file = create_layer_file(layer2_num_neurons, image_id, 2, LOCAL_SPICE_MODEL_FILEPATH, PWL_FILE_TEMPLATE, SUBCIRCUIT_FORMAT, KNOB_PARAMS, VDD, VSS)

        # Run Third Layer Simulation
        start_time = time.time()
        if RUN_SPICE_SIMULATION:
            run_simulation_one_file(layer2_file, simulator=SIMULATOR)
        end_time = time.time()
        print(f"Image: {image_id}, Simulation Time of Layer2: {end_time - start_time}s") # Sim time was about 412 seconds for first layer

        # Read Simulation File for Layer1
        start_time = time.time()
        raw_obj_layer2 = read_simulation_file(layer2_file, simulator=SIMULATOR)
        print(f"Image: {image_id}, Simulation File Read for Layer2")
        # Analyze Third Layer File, Max Pool and find which is right 
        time_vec = get_signal("time", raw_obj_layer2, simulator=SIMULATOR)

        layer2_neuron_datasets = []

        for i in range(layer2_num_neurons):
            # Get circuit raw information
            raw_input_current = get_signal(f"i(I_inj_{i})", raw_obj_layer2, simulator=SIMULATOR)
            raw_output_current = get_signal(f"i(Cout_{i})", raw_obj_layer2, simulator=SIMULATOR)
            raw_circuit_state = get_signal(f"v(input_{i})", raw_obj_layer2, simulator=SIMULATOR)
            raw_vdd = get_signal(f"v(vdd_{i})", raw_obj_layer2, simulator=SIMULATOR)
            raw_vdd_current = get_signal(f"i(Vdd_{i})", raw_obj_layer2, simulator=SIMULATOR)
            neuron_dataset = analyze_spike_information_mnist(image_id,2,i,NUMBER_OF_TIMESTEPS_PER_INFERENCE, 
                                                            DIGITAL_TIMESTEP,FOOTPRINT_CHARGE, KNOB_PARAMS, time_vec, raw_input_current,
                                                            raw_circuit_state, raw_output_current, raw_vdd, raw_vdd_current,
                                                            VERBOSE=False, PLOT_RUNS=False, SIMULATOR=SIMULATOR)
            layer2_neuron_datasets.append(neuron_dataset)
        
        # Concatenate Everything into one big pandas dataset
        layer2_events = pd.concat(layer2_neuron_datasets, axis=0, ignore_index=True)
        layer2_events.to_csv(os.path.join(CSV_DIRECTORY,f"img{image_id}_layer{2}_events_dataset.csv"), index=False)

        end_time = time.time()
        print(f"Image: {image_id}, Simulation Time to read Layer2 Output: {end_time - start_time}")

        # Just get spike events
        spike_data_layer2_df = layer2_events[layer2_events['Event_Type'] == 'in-out']

        # Check Which one spiked most
        spike_counts = spike_data_layer2_df.groupby('Neuron_Num').size()
        spike_counts_dict = spike_counts.to_dict()
        max_neuron = max(spike_counts_dict, key=spike_counts_dict.get)

        # Return max neuron, label and the dictionary 
        iter_end = time.time()
        total_time_run = iter_end-iter_start
        print(f"Image: {image_id}, Total Time: {total_time_run}")
        #print(f"Image: {image_id}, Predicted: {max_neuron}, Label: {label}, Dict: {spike_counts_dict}, Total Time: {total_time_run:.2f}s")   
        return (image_id, max_neuron, label, spike_counts_dict, total_time_run)
    
    except Exception as e:
        print(f"ERR on {os.getpid()}, MSG: {e}")
        e.terminate()


# ======================================
# START SCRIPT #

def test_multiprocessing(image_data):
    try:
        image_id, image, label = image_data 
        time.sleep(0.5)
        #meow = 1/0
        print(f"Running Image: {image_id}, Process ID: {os.getpid()}")
        return (image_id, 0, 0, {}, 10)
    except Exception as e:
        print(f"ERR on {os.getpid()}, MSG: {e}")
        #e.terminate()

start = time.time()

total_correct = 0
total_img_processed = 0
try:
    # Multiprocess across batch
    for batch_id, (images, labels) in enumerate(testloader):
        image_data_list = [(batch_id * BATCH_SIZE + i + IMAGE_START_OFFSET, img, lbl) for i, (img, lbl) in enumerate(zip(images, labels))]
        batch_start = time.time()

        with multiprocessing.Pool(NUM_PROCESSES) as pool:
            results = pool.map(process_image, image_data_list)
            #results = pool.map(test_multiprocessing, image_data_list)

        print(f"Batch ID: {batch_id}, Finished Parallel Batch Call")
        sys.stdout.flush() 

        batch_img_processed = len(results)
        batch_total_correct = 0

        for (image_id, pred, lbl, spike_dict, total_time_run) in results:
            print(f"Image: {image_id}, Predicted: {pred}, Label: {lbl}, Dict: {spike_dict}, Total Time: {total_time_run:.2f}s")            
            sys.stdout.flush() 

            if pred == lbl:
                batch_total_correct +=1

        total_img_processed += batch_img_processed
        total_correct += batch_total_correct
        
        print(f"Batch: {batch_id}, Batch Accuracy: ({batch_total_correct} / {batch_img_processed}) {batch_total_correct / batch_img_processed * 100:.2f}, Total Accuracy: ({total_correct} / {total_img_processed}) {total_correct / total_img_processed * 100:.2f}")
        sys.stdout.flush() 
        batch_end = time.time()
        print(f"Batch {batch_id}, Run Time: {batch_end-batch_start:.2f}s")
        sys.stdout.flush() 
        # Get rid of PWL files and spice_run logs  (Otherwise we will run out of space hehe)
        shutil.rmtree(PWL_FILE_MAIN_DIRECTORY)
        os.makedirs(PWL_FILE_MAIN_DIRECTORY)

        shutil.rmtree(SPICE_RUN_DIRECTORY)
        os.makedirs(SPICE_RUN_DIRECTORY)

        shutil.copyfile(MODEL_FILEPATH, LOCAL_SPICE_MODEL_FILEPATH)

        # Copy over other files into SPICE_RUN_DIRECTORY
        if OTHER_NECESSARY_FILES_IN_SPICE_RUN_DIRECTORY:
            for necessary_file in OTHER_NECESSARY_FILES_IN_SPICE_RUN_DIRECTORY:
                shutil.copyfile(necessary_file, os.path.join(SPICE_RUN_DIRECTORY, os.path.basename(necessary_file)))

        if total_img_processed >= NUMBER_OF_IMAGES:
            print(f"Finishing Early! Ran {total_img_processed} / {NUMBER_OF_IMAGES}")
            sys.stdout.flush() 
            break
except Exception as e:
    print(f"ERR on batch {batch_id}: {e}")

end = time.time()

print(f"Finished Running {NUMBER_OF_IMAGES} inferences! Exiting ")
print(f"Total Accuracy: ({total_correct} / {total_img_processed}) {total_correct / total_img_processed * 100:.2f}")
print(f"Ran in {end-start}s")

# Save file :)
with open('../results/spiking_neuron_mnist_results.txt', 'w') as f:
    f.write(f"Finished Running {NUMBER_OF_IMAGES} inferences! Exiting \n")
    f.write(f"Total Accuracy: ({total_correct} / {total_img_processed}) {total_correct / total_img_processed * 100:.2f}\n")
    f.write(f"Ran in {end-start}s\n")

sys.stdout.flush() 

if PLOT_RUNS:
    plt.show()
