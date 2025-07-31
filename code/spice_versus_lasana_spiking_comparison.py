import pandas as pd
import os
import numpy as np 
from sklearn.metrics import mean_absolute_percentage_error
figure_counter = 0

RUN_FOLDER = "../data"
lasana_runs = os.path.join(RUN_FOLDER, 'spiking_mnist_lasana_results')
spice_runs = os.path.join(RUN_FOLDER, 'spiking_mnist_golden_results')
results_folder = '../results/spiking_mnist_lasana_spice_comparison'

num_layers = 3
layer_sizes = [784, 128, 10]
num_timesteps = 100
num_images = 500
image_offset = 0
lasana_str = '{}_spike_info_layer{}.csv'
spice_str = 'img{}_layer{}_events_dataset.csv'
columns_of_interest = ["Neuron_Num","Digital_Time_Step","Event_Type", 'Weight', "Output_Spike", 'Energy', "Latency", "Cap_Voltage_At_Input_Start","Cap_Voltage_At_Output_End"]

# --------------------------------------
# Create Results folder if it does not exist
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

def calculate_percentage_error(spice_spikes, lasana_spikes):
    if lasana_spikes == 0:
        return 0  # If lasana_spikes is zero, we assume 100% error (to avoid division by zero)
    return abs(spice_spikes - lasana_spikes) / lasana_spikes * 100

img_ids = []
energy_errs = []
latency_errs = []
latency_mape = []
dynamic_energy_mape = []

for k in range(num_images):
    print(f"image {k}")
    image_id = k + image_offset
    img_ids.append(image_id)

    lasana_dfs = {}
    spice_dfs = {}

    for i in range(num_layers):
        # Load and scale SPICE data
        raw_spice_df = pd.read_csv(os.path.join(spice_runs, spice_str.format(image_id, i)))
        raw_spice_df['Energy'] = raw_spice_df['Energy'] * 1e12  # to pJ
        raw_spice_df['Latency'] = raw_spice_df['Latency'] * 1e9  # to ns
        raw_spice_df['Output_Spike'] = (raw_spice_df['Event_Type'] == 'in-out').astype(int)
        spice_dfs[i] = raw_spice_df[columns_of_interest]
        
        # Load and scale Lasana data
        raw_lasana_df = pd.read_csv(os.path.join(lasana_runs, lasana_str.format(image_id, i)))
        raw_lasana_df['Energy'] = raw_lasana_df['Energy'] * 1e12  # to pJ
        raw_lasana_df['Latency'] = raw_lasana_df['Latency'] * 1e9  # to ns
        lasana_dfs[i] = raw_lasana_df[columns_of_interest]

    total_energy_lasana = 0
    total_energy_spice = 0

    # ENERGY ERROR
    # Loop over each layer
    for i in range(num_layers):
        spice_layer_sum = spice_dfs[i]["Energy"].sum()
        lasana_layer_sum = lasana_dfs[i]["Energy"].sum()
        
        # Get the energy for the current layer in both spice and lasana
        total_energy_spice += spice_layer_sum
        total_energy_lasana += lasana_layer_sum

    # Print Output
    total_energy_err = calculate_percentage_error(total_energy_spice, total_energy_lasana)
    print(f"Total Energy Percentage Error: {total_energy_err}")
    energy_errs.append(total_energy_err)

    total_latency_lasana = 0
    total_latency_spice = 0

    # LATENCY ERROR
    # Loop over each layer
    for i in range(num_layers):
        # Initialize counters for the current layer, filtered to 'in-out' events
        spice_layer_sum = spice_dfs[i]["Latency"].sum()
        lasana_layer_sum = lasana_dfs[i]["Latency"].sum()

        total_latency_spice += spice_layer_sum
        total_latency_lasana += lasana_layer_sum

    # Print Output
    total_latency_err = calculate_percentage_error(total_latency_spice, total_latency_lasana)
    print(f"Total Latency Percentage Error: {total_latency_err}")
    latency_errs.append(total_latency_err)

    # Latency MAPE
    matched_spice_latencies = []
    matched_lasana_latencies = []

    for i in range(num_layers):
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
    print(f"Latency MAPE: {mape}%")
    latency_mape.append(mape)

    # Dynamic Energy MAPE
    matched_spice_energies = []
    matched_lasana_energies = []

    for i in range(num_layers):
        # Filter SPICE for 'in-out' events and Output_Spike == 1
        spice_filtered = spice_dfs[i][
            (spice_dfs[i]['Event_Type'] == 'in-out')
        ]

        lasana_filtered = lasana_dfs[i]

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
    print(f"Dynamic Energy MAPE: {mape}%")
    dynamic_energy_mape.append(mape)

print(f"Num Images: {num_images}")
print(f"Average Energy Error: {sum(energy_errs) / len(energy_errs)}")
print(f"Average Latency Error: {sum(latency_errs) / len(latency_errs)} ")
print(f"Average MAPE Latency Error / Image Inference: {sum(latency_mape) / len(latency_mape)}")
print(f"Average MAPE Dynamic Energy Error / Image Inference: {sum(dynamic_energy_mape) / len(dynamic_energy_mape)}")

# Create Pandas DF to save everything to
data = list(zip(img_ids, energy_errs, latency_errs, dynamic_energy_mape, latency_mape))
columns = ['image_id', 'total_energy_percentage_err', 'total_latency_percentage_err',  "dynamic_energy_MAPE", 'latency_MAPE']

df = pd.DataFrame(data, columns=columns)
df.to_csv(os.path.join(results_folder, 'per_inference_statistics.csv'), index=False)

# Save Average Energy, Latency, Latency MAPE, and Dynamic Energy MAPE
summary_file = os.path.join(results_folder, "metrics_summary.txt")

with open(summary_file, "w") as f:
    # TODO: Would be really cool if we could get accuracy here :)
    f.write(f"Num Images: {num_images}\n")
    f.write(f"Average Energy Error: {sum(energy_errs) / len(energy_errs)}\n")
    f.write(f"Average Latency Error: {sum(latency_errs) / len(latency_errs)}\n")
    f.write(f"Average MAPE Latency Error / Image Inference: {sum(latency_mape) / len(latency_mape)}\n")
    f.write(f"Average MAPE Dynamic Energy Error / Image Inference: {sum(dynamic_energy_mape) / len(dynamic_energy_mape)}\n")

