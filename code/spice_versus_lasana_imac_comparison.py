import pandas as pd
import os
figure_counter = 0

'''
LASANA File for comparing Spectre / HSPICE golden results against the LASANA equivalents for MNIST.
Specifically, this file provides high-level total energy, and latency comparisons per inference. Additionally, 
since these networks consist of many strings of LASANA models that are strung together to represent a multitude
of crossbars, we provide the latency and energy MAPEs as well, which show how each of the sub-modules inside of the
network perform.

For each of the equivalent runs, there is a top file that contains top-level information such as accuracy,
full energy, etc. of each of the inferences. For each inference, there is also a separate file, labeled
"image_X_inference", where X is the inference ID in the MNIST dataset.

This script then spits out two results into the "results_folder." The results folder consists of per-inference
results of average total energy, average total latency, energy MAPE, and latency MAPE. The second file is the same
results that were printed to the console, but saved under metrics_summary.txt

Note that following the scripts, these log files are created automatically, by both the SPICE MNIST scripts
and LASANA scripts provided. 

Author: Jason Ho
'''

# --------------------------
# Start of the Hyperparameters

RUN_FOLDER = "../data"                                                      # Where results are stored
LASANA_RUNS = os.path.join(RUN_FOLDER, 'crossbar_mnist_lasana_results')     # LASANA specific log files where each .csv represents one image inference
SPICE_RUNS = os.path.join(RUN_FOLDER, "crossbar_mnist_golden_results")      # SPICE specific log files where each .csv represents one image inference

# Full CSV Runs
FULL_IMAC = os.path.join(RUN_FOLDER,'crossbar_mnist_golden_acc_data.csv')   # LASANA specific top-log file which contains accuracy and total energy information
FULL_LASANA = os.path.join(RUN_FOLDER,'crossbar_mnist_lasana_acc_data.csv') # SPICE specific top-log file which contains accuracy and total energy information

# Output Name
results_folder = '../results/crossbar_mnist_lasana_spice_comparison'        # Output folder where results will be dumped out

NUM_IMAGES = 500                                                            # Number of images to be compared (Note: No error correction so will throw errors if there is a problem)
IMAGE_OFFSET = 0                                                            # Image Offset of where to start. 
NUM_LAYERS = 4                                                              # Number of layers in the neural network (not necessary, but an artifact from the other comparison script)
LAYER_SIZES = [400, 128, 84, 10]                                            # Layer sizes (again, artifact from other comparison scripts)

FILE_STR = "image_{}_inference.csv"                                         # File format for the logs. This is default, but unless someone changes it, this does not need to be changed

# --------------------------

def parse_circuit(name):
    parts = name.split('_')
    if name.startswith('Xsig'):
        return {
            'type': 'activation',
            'layer': int(parts[2]),
            'neuron_id': int(parts[3]),
            'x_partition_id': None,
            'y_partition_id': None,
            'row': None
        }
    elif name.startswith('layer'):
        return {
            'type': 'crossbar',
            'layer': int(parts[1]),
            'x_partition_id': int(parts[2]),
            'y_partition_id': int(parts[3]),
            'row': int(parts[4]),
            'neuron_id': None
        }
    else:
        return {
            'type': 'unknown',
            'layer': None,
            'x_partition_id': None,
            'y_partition_id': None,
            'row': None,
            'neuron_id': None
        }
    
def get_metrics(df_merged, custom_string="", supress_metrics=True):
    # Remove things are zero as this will cause errors :)
    df_merged = df_merged[(df_merged['energy_true'] != 0) & (df_merged['latency_true'] != 0)].copy()

    # Get total latency error
    total_latency_pred = df_merged['latency_pred'].sum()
    total_latency_true = df_merged['latency_true'].sum()
    total_latency_err = (abs(total_latency_pred - total_latency_true) / total_latency_true) * 100

    df_merged['mape_energy'] = abs(df_merged['energy_pred'] - df_merged['energy_true']) / df_merged['energy_true']
    mape_energy = df_merged['mape_energy'].mean() * 100

    df_merged['mape_latency'] = abs(df_merged['latency_pred'] - df_merged['latency_true']) / df_merged['latency_true']
    mape_latency = df_merged['mape_latency'].mean() * 100

    if not supress_metrics:
        print(f"{custom_string} Energy MAPE: {mape_energy:.2f}%")
        print(f"{custom_string} Latency MAPE: {mape_latency:.2f}%")

    return total_latency_err, mape_energy, mape_latency

# --------------------------

# Create Results folder if it does not exist
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

latency_errs = []
energy_errs = []

all_latency_mapes = []
all_energy_mapes = []
all_output_mses = []

results = []

# Look at full energies :)
full_imac_spice = pd.read_csv(FULL_IMAC)
full_imac_lasana = pd.read_csv(FULL_LASANA)
df_merged_energy = pd.merge(full_imac_lasana, full_imac_spice, on=['image_num'], suffixes=('_pred', '_true'))
df_merged_energy['percent_error_energy'] = (abs(df_merged_energy['energy_pred'] - df_merged_energy['energy_true']) / df_merged_energy['energy_true']) * 100

for k in range(NUM_IMAGES):
    image_id = k + IMAGE_OFFSET
    if image_id % 50 == 0:
        print(f"Analyzing Image {image_id}")

    # Read file
    lasana_run_df = pd.read_csv(os.path.join(LASANA_RUNS, FILE_STR.format(image_id)))
    spice_run_df = pd.read_csv(os.path.join(SPICE_RUNS, FILE_STR.format(image_id)))

    # Parse Circuit 
    parsed_df = lasana_run_df['circuit_name'].apply(parse_circuit).apply(pd.Series)
    lasana_run_df = pd.concat([lasana_run_df, parsed_df], axis=1)

    parsed_df = spice_run_df['circuit_name'].apply(parse_circuit).apply(pd.Series)
    spice_run_df = pd.concat([spice_run_df, parsed_df], axis=1)

    spice_run_df['latency'] = spice_run_df['latency'] * 10**9 # convert to ns
    spice_run_df['energy'] = spice_run_df['energy'] * 10**12 # convert to ps
    
    df_merged = pd.merge(lasana_run_df,spice_run_df,
                    on=['layer', 'x_partition_id', 'y_partition_id', 'row', 'neuron_id'],
                    suffixes=('_pred', '_true')
                )
    
    # All
    total_latency_err, all_energy_mape, all_latency_mape = get_metrics(df_merged, "ALL")
    all_latency_mapes.append(all_latency_mape)
    all_energy_mapes.append(all_energy_mape)
    latency_errs.append(total_latency_err)
    total_energy_error = df_merged_energy.loc[df_merged_energy['image_num'] == image_id, 'percent_error_energy'].values

    row = {
        "image_id": image_id,
        'total_energy_percentage_err': total_energy_error[0],
        'total_latency_percentage_err': total_latency_err,
        "dynamic_energy_MAPE": all_energy_mape,
        "latency_MAPE": all_latency_mape
    }
    results.append(row)

print(f"Num Images: {NUM_IMAGES}")
print(f"Average Energy Error: {df_merged_energy['percent_error_energy'].mean()}")
print(f"Average Latency Error: {sum(latency_errs) / len(latency_errs)}")
print(f"Average MAPE Energy Error / Image Inference: {sum(all_energy_mapes) / len(all_energy_mapes)}")
print(f"Average MAPE Latency Error / Image Inference: {sum(all_latency_mapes) / len(all_latency_mapes)}")


# Save all results to CSV so we do not have to recalculate :)
output_csv_name = os.path.join(results_folder, "per_inference_statistics.csv")
df_metrics = pd.DataFrame(results)
df_metrics.to_csv(output_csv_name, index=False)

# Save Average Energy, Latency, Latency MAPE, and Dynamic Energy MAPE
summary_file = os.path.join(results_folder, "metrics_summary.txt")

with open(summary_file, "a") as f:
    f.write(f"Num Images: {NUM_IMAGES}\n")
    f.write(f"Average Energy Error: {df_merged_energy['percent_error_energy'].mean()}\n")
    f.write(f"Average Latency Error: {sum(latency_errs) / len(latency_errs)}\n")
    f.write(f"Average MAPE Energy Error / Image Inference: {sum(all_energy_mapes) / len(all_energy_mapes)}\n")
    f.write(f"Average MAPE Latency Error / Image Inference: {sum(all_latency_mapes) / len(all_latency_mapes)}\n")
    
