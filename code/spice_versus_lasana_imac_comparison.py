import pandas as pd
import os
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
figure_counter = 0
pd.set_option('display.max_columns', None)

# FIXME: Get rid of MSE here because it is no longer needed.

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

    mse_energy = ((df_merged['energy_pred'] - df_merged['energy_true']) ** 2).mean()
    mse_latency = ((df_merged['latency_pred'] - df_merged['latency_true']) ** 2).mean()
    mse_output = ((df_merged['output_value_pred'] - df_merged['output_value_true']) ** 2).mean()

    if not supress_metrics:
        print(f"{custom_string} Energy MAPE: {mape_energy:.2f}%")
        print(f"{custom_string} Latency MAPE: {mape_latency:.2f}%")
        print(f"{custom_string} MSE (energy): {mse_energy:.4f} pJ^2")
        print(f"{custom_string} MSE (latency): {mse_latency:.4f} ns^2")

    return total_latency_err, mape_energy, mape_latency, mse_energy, mse_latency, mse_output

# --------------------------
RUN_FOLDER = "../data"
lasana_runs = os.path.join(RUN_FOLDER, 'crossbar_mnist_lasana_results')
spice_runs = os.path.join(RUN_FOLDER, "crossbar_mnist_golden_results")

# Full CSV Runs
full_imac = os.path.join(RUN_FOLDER,'crossbar_mnist_golden_acc_data.csv')
full_lasana = os.path.join(RUN_FOLDER,'crossbar_mnist_lasana_acc_data.csv')

# Output Name
results_folder = '../results/crossbar_mnist_lasana_spice_comparison'


# This is just
num_images = 50     
num_layers = 4
layer_sizes = [400, 128, 84, 10]
image_offset = 0
file_str = "image_{}_inference.csv"
columns_of_interest = ["circuit_name", "latency", "energy", 'output_value']

# Create Results folder if it does not exist
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

latency_errs = []
energy_errs = []

all_latency_mapes = []
all_energy_mapes = []
all_output_mses = []

results = []

for k in range(num_images):
    print(f"image {k}")
    image_id = k + image_offset

    # Read file
    lasana_run_df = pd.read_csv(os.path.join(lasana_runs, file_str.format(image_id)))
    spice_run_df = pd.read_csv(os.path.join(spice_runs, file_str.format(image_id)))

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
    
    print(df_merged)
    
    # All
    total_latency_err, all_energy_mape, all_latency_mape, all_energy_mse, all_latency_mse, all_output_mse = get_metrics(df_merged, "ALL")
    all_latency_mapes.append(all_latency_mape)
    all_energy_mapes.append(all_energy_mape)
    all_output_mses.append(all_output_mse)
    latency_errs.append(total_latency_err)

    row = {
        "image_id": k,

        # Overall
        "all_energy_mape": all_energy_mape,
        "all_latency_mape": all_latency_mape,
        "all_energy_mse": all_energy_mse,
        "all_latency_mse": all_latency_mse,
        "all_output_mse": all_output_mse,
    }

    results.append(row)

# Look at full energies :)
full_imac_spice = pd.read_csv(full_imac)
full_imac_lasana = pd.read_csv(full_lasana)
df_merged = pd.merge(full_imac_lasana, full_imac_spice, on=['image_num'], suffixes=('_pred', '_true'))
df_merged['percent_error_energy'] = (abs(df_merged['energy_pred'] - df_merged['energy_true']) / df_merged['energy_true']) * 100

# TODO: Maybe save this output in a different way. Literally could just send to a text file in the results so that they have it.
print(f"Num Images: {num_images}")
print(f"Average Energy Error: {df_merged['percent_error_energy'].mean()}")
print(f"Average Latency Error: {sum(latency_errs) / len(latency_errs)}")
print(f"Average MAPE All Latency Error / Image Inference: {sum(all_latency_mapes) / len(all_latency_mapes)}")
print(f"Average MAPE All Energy Error / Image Inference: {sum(all_energy_mapes) / len(all_energy_mapes)}")
#print(f"Average Output MSE / Image Inference:{sum(all_output_mses) / len(all_output_mses)}")



# Save all results to CSV so we do not have to recalculate :)
# FIXME: Fix this such that it gives similar results to the other one :)p
output_csv_name = os.path.join(results_folder, "per_inference_statistics.csv")
df_metrics = pd.DataFrame(results)
df_metrics.to_csv(output_csv_name, index=False)


# Save Average Energy, Latency, Latency MAPE, and Dynamic Energy MAPE
summary_file = os.path.join(results_folder, "metrics_summary.txt")

with open(summary_file, "w") as f:
    f.write(f"Num Images: {num_images}\n")
    f.write(f"Average Energy Error: {df_merged['percent_error_energy'].mean()}\n")
    f.write(f"Average Latency Error: {sum(latency_errs) / len(latency_errs)}\n")
    f.write(f"Average MAPE All Latency Error / Image Inference: {sum(all_latency_mapes) / len(all_latency_mapes)}\n")
    f.write(f"Average MAPE All Energy Error / Image Inference: {sum(all_energy_mapes) / len(all_energy_mapes)}\n")
