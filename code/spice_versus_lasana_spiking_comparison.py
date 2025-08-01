import pandas as pd
import os
import numpy as np 
from sklearn.metrics import mean_absolute_percentage_error
figure_counter = 0

'''
LASANA File for comparing Spectre / HSPICE golden results against the LASANA equivalents for Spiking MNIST.
Specifically, this file provides high-level total energy, and latency comparisons per inference. Additionally, 
since these networks consist of many strings of LASANA models that are strung together to represent a multitude
of crossbars, we provide the latency and energy MAPEs as well, which show how each of the sub-modules inside of the
network perform.

For each of the equivalent runs, each of the inferences have a separate file for each of the layers, labeled
baed on the LASANA_STR and SPICE_STR, dictated below.

This script then spits out two results into the "results_folder." The results folder consists of per-inference
results of average total energy, average total latency, energy MAPE, and latency MAPE. The second file is the same
results that were printed to the console, but saved under metrics_summary.txt

Note that following the scripts, these log files are created automatically, by both the SPICE MNIST scripts
and LASANA scripts provided. 

Author: Jason Ho
'''

# --------------------------
# Start of the Hyperparameters
RUN_FOLDER = "../data"                                                      # Run Folder where all the results are stored
LASANA_RUNS = os.path.join(RUN_FOLDER, 'spiking_mnist_lasana_results')      # Specific run folder for the LASANA, layer-specific logs
SPICE_RUNS = os.path.join(RUN_FOLDER, 'spiking_mnist_golden_results')       # Specific run folder for the SPICE, layer-specific logs
RESULTS_FOLDER = '../results/spiking_mnist_lasana_spice_comparison'         # results folder, which stores the results of this script

NUM_IMAGES = 500                                                            # Number of images to evaluate
IMAGE_OFFSET = 0                                                            # How many inferences to offset from the test dataset

NUM_LAYERS = 3                                                              # Number of layers in the spiking neural network
LAYER_SIZES = [784, 128, 10]                                                # Layer sizes of each of the fully connected spiking layers
NUM_TIMESTEPS = 100                                                         # Number of timesteps in each of the inferences

LASANA_STR = '{}_spike_info_layer{}.csv'                                    # First {} refers to image num, second {} refers to layer number
SPICE_STR = 'img{}_layer{}_events_dataset.csv'                              # First {} refers to image num, second {} refers to layer number
COLUMNS_OF_INTEREST = ["Neuron_Num","Digital_Time_Step","Event_Type", 'Weight', "Output_Spike", 'Energy', "Latency", "Cap_Voltage_At_Input_Start","Cap_Voltage_At_Output_End"]

# --------------------------------------

# Create Results folder if it does not exist
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

def calculate_percentage_error(spice_spikes, lasana_spikes):
    if lasana_spikes == 0:
        return 0  # If lasana_spikes is zero, we assume 100% error (to avoid division by zero)
    return abs(spice_spikes - lasana_spikes) / lasana_spikes * 100

img_ids = []
energy_errs = []
latency_errs = []
latency_mape = []
dynamic_energy_mape = []

for k in range(NUM_IMAGES):
    image_id = k + IMAGE_OFFSET
    if image_id % 50 == 0:
        print(f"Analyzing Image {image_id}")
    img_ids.append(image_id)

    lasana_dfs = {}
    spice_dfs = {}

    for i in range(NUM_LAYERS):
        # Load and scale SPICE data
        raw_spice_df = pd.read_csv(os.path.join(SPICE_RUNS, SPICE_STR.format(image_id, i)))
        raw_spice_df['Energy'] = raw_spice_df['Energy'] * 1e12  # to pJ
        raw_spice_df['Latency'] = raw_spice_df['Latency'] * 1e9  # to ns
        raw_spice_df['Output_Spike'] = (raw_spice_df['Event_Type'] == 'in-out').astype(int)
        spice_dfs[i] = raw_spice_df[COLUMNS_OF_INTEREST]
        
        # Load and scale Lasana data
        raw_lasana_df = pd.read_csv(os.path.join(LASANA_RUNS, LASANA_STR.format(image_id, i)))
        raw_lasana_df['Energy'] = raw_lasana_df['Energy'] * 1e12  # to pJ
        raw_lasana_df['Latency'] = raw_lasana_df['Latency'] * 1e9  # to ns
        lasana_dfs[i] = raw_lasana_df[COLUMNS_OF_INTEREST]

    total_energy_lasana = 0
    total_energy_spice = 0

    # ENERGY ERROR
    # Loop over each layer
    for i in range(NUM_LAYERS):
        spice_layer_sum = spice_dfs[i]["Energy"].sum()
        lasana_layer_sum = lasana_dfs[i]["Energy"].sum()
        
        # Get the energy for the current layer in both spice and lasana
        total_energy_spice += spice_layer_sum
        total_energy_lasana += lasana_layer_sum

    # Print Output
    total_energy_err = calculate_percentage_error(total_energy_spice, total_energy_lasana)
    #print(f"Total Energy Percentage Error: {total_energy_err}")
    energy_errs.append(total_energy_err)

    total_latency_lasana = 0
    total_latency_spice = 0

    # LATENCY ERROR
    # Loop over each layer
    for i in range(NUM_LAYERS):
        # Initialize counters for the current layer, filtered to 'in-out' events
        spice_layer_sum = spice_dfs[i]["Latency"].sum()
        lasana_layer_sum = lasana_dfs[i]["Latency"].sum()

        total_latency_spice += spice_layer_sum
        total_latency_lasana += lasana_layer_sum

    # Print Output
    total_latency_err = calculate_percentage_error(total_latency_spice, total_latency_lasana)
    #print(f"Total Latency Percentage Error: {total_latency_err}")
    latency_errs.append(total_latency_err)

    # Latency MAPE
    matched_spice_latencies = []
    matched_lasana_latencies = []

    for i in range(NUM_LAYERS):
        # Filter SPICE for 'in-out' events and Output_Spike == 1
        spice_filtered = spice_dfs[i][
            (spice_dfs[i]['Event_Type'] == 'in-out') & (spice_dfs[i]['Output_Spike'] == 1)
        ]

        # Filter Lasana for Output_Spike == 1
        lasana_filtered = lasana_dfs[i][
            lasana_dfs[i]['Output_Spike'] == 1
        ]

        # Rename latency columns to distinguish them after merge
        spice_filtered = spice_filtered.rename(columns={'Latency': 'Latency_spice'})
        lasana_filtered = lasana_filtered.rename(columns={'Latency': 'Latency_lasana'})
        
        # Merge on Neuron_Num and Timestep
        merged = pd.merge(
            spice_filtered[['Neuron_Num', "Digital_Time_Step", 'Latency_spice']],
            lasana_filtered[['Neuron_Num', "Digital_Time_Step", 'Latency_lasana']],
            on=['Neuron_Num', "Digital_Time_Step"],
            how='inner'
        )

        spice_arr = np.array(merged['Latency_spice'].tolist())
        lasana_arr = np.array(merged['Latency_lasana'].tolist())
        merged['Error'] = np.abs((merged['Latency_lasana'] - merged['Latency_spice']) / merged['Latency_spice'])

        # Append to master lists
        matched_spice_latencies.extend(merged['Latency_spice'].tolist())
        matched_lasana_latencies.extend(merged['Latency_lasana'].tolist())

    spice_arr = np.array(matched_spice_latencies)
    lasana_arr = np.array(matched_lasana_latencies)

    nonzero_mask = spice_arr != 0
    mape = mean_absolute_percentage_error(lasana_arr[nonzero_mask ], spice_arr[nonzero_mask]) * 100
    #print(f"Latency MAPE: {mape}%")
    latency_mape.append(mape)

    # Dynamic Energy MAPE
    matched_spice_energies = []
    matched_lasana_energies = []

    for i in range(NUM_LAYERS):
        # Filter SPICE for 'in-out' events and Output_Spike == 1
        spice_filtered = spice_dfs[i][
            (spice_dfs[i]['Event_Type'] == 'in-out')
        ]

        # Rename latency columns to distinguish them after merge
        spice_filtered = spice_filtered.rename(columns={'Energy': 'Energy_spice'})
        lasana_filtered = lasana_filtered.rename(columns={'Energy': 'Energy_lasana'})
        
        # Merge on Neuron_Num and Timestep
        merged = pd.merge(
            spice_filtered[['Neuron_Num', "Digital_Time_Step", 'Energy_spice']],
            lasana_filtered[['Neuron_Num', "Digital_Time_Step", 'Energy_lasana']],
            on=['Neuron_Num', "Digital_Time_Step"],
            how='inner'
        )

        spice_arr = np.array(merged['Energy_spice'].tolist())
        lasana_arr = np.array(merged['Energy_lasana'].tolist())
        merged['Error'] = np.abs((merged['Energy_lasana'] - merged['Energy_spice']) / merged['Energy_spice'])

        # Append to master lists
        matched_spice_latencies.extend(merged['Energy_spice'].tolist())
        matched_lasana_latencies.extend(merged['Energy_lasana'].tolist())

    spice_arr = np.array(matched_spice_latencies)
    lasana_arr = np.array(matched_lasana_latencies)

    nonzero_mask = spice_arr != 0
    mape = mean_absolute_percentage_error(lasana_arr[nonzero_mask ], spice_arr[nonzero_mask]) * 100
    #print(f"Dynamic Energy MAPE: {mape}%")
    dynamic_energy_mape.append(mape)

print(f"Num Images: {NUM_IMAGES}")
print(f"Average Energy Error: {sum(energy_errs) / len(energy_errs)}")
print(f"Average Latency Error: {sum(latency_errs) / len(latency_errs)} ")
print(f"Average MAPE Latency Error / Image Inference: {sum(latency_mape) / len(latency_mape)}")
print(f"Average MAPE Dynamic Energy Error / Image Inference: {sum(dynamic_energy_mape) / len(dynamic_energy_mape)}")

# Create Pandas DF to save everything to
data = list(zip(img_ids, energy_errs, latency_errs, dynamic_energy_mape, latency_mape))
columns = ['image_id', 'total_energy_percentage_err', 'total_latency_percentage_err',  "dynamic_energy_MAPE", 'latency_MAPE']

df = pd.DataFrame(data, columns=columns)
df.to_csv(os.path.join(RESULTS_FOLDER, 'per_inference_statistics.csv'), index=False)

# Save Average Energy, Latency, Latency MAPE, and Dynamic Energy MAPE
summary_file = os.path.join(RESULTS_FOLDER, "metrics_summary.txt")

with open(summary_file, "a") as f:
    f.write(f"Num Images: {NUM_IMAGES}\n")
    f.write(f"Average Energy Error: {sum(energy_errs) / len(energy_errs)}\n")
    f.write(f"Average Latency Error: {sum(latency_errs) / len(latency_errs)}\n")
    f.write(f"Average MAPE Latency Error / Image Inference: {sum(latency_mape) / len(latency_mape)}\n")
    f.write(f"Average MAPE Dynamic Energy Error / Image Inference: {sum(dynamic_energy_mape) / len(dynamic_energy_mape)}\n")

