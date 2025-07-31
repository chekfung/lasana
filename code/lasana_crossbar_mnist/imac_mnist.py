import torch
import csv
import numpy as np
import os
from catboost import CatBoostRegressor
import math
import random
from crossbar import CrossBar
from neuron import DigitalNeuron
import pandas as pd

# Make sure all seeds are the same
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ------ BEGIN Hyperparameters -------
testnum=500                                                                 # Number of input test cases to run
testnum_per_batch=10                                                        # Number of test cases in a single batch, testnum should be divisible by this number
firstimage=0                                                                # start the test inputs from this image\
csv_name = '../../data/crossbar_mnist_lasana_acc_data.csv'                  # Refers to top CSV for high level acc and energy (where to save files)
csv_folder = '../../data/crossbar_mnist_lasana_results'                     # Per analog block energy, latency, etc. logs (where to save files)
SAVE_ACC_PATH = '../../results/crossbar_mnist_lasana_spice_comparison'      # Where to save the 

USE_QUANTIZATION = True
DAC_BITS = 8                                
MODEL_RUN_NAME_DIFF_10 = "pcm_crossbar_diff_10_run"                         # Where the LASANA models are located for diff 10 PCM Crossbar 32x1
MODEL_RUN_NAME_DIFF_30 = "pcm_crossbar_diff_30_run"                         # Where the LASASNA models are located for diff 30 PCM Crossbar 32x1

#list of inputs start
data_dir=os.path.join('../../data', 'imac_mnist_model') #The directory where data files are located (both the weights, biases, and dataset inputs and labels)
dataset_file='test_data.csv' #Name of the dataset file
label_file='test_labels.csv' #Name of the label file

# Crossbar Parameters
vdd=0.8                     # The positive supply voltage
vss=-0.8                    # The negative supply voltage
tsampling=4                 # The sampling time in nanosecond  
nodes=[400,120,84,10]       # Network Topology, an array which defines the DNN model size
xbar=[32,32]                # The crossbar tile size
gain=[30,30,10]             # Array for the differential amplifier gains of all hidden layers

# Running PCM
rlow=78000
rhigh=202000

# ------ END HYPERPARAMETERS -------
# Calculate partitions
hpar=[math.ceil((x+1)/xbar[0]) for x in nodes] 
hpar.pop() 
vpar=[math.ceil(x/xbar[1]) for x in nodes] 
vpar.pop(0) 

if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)
    print(f"Created folder: {csv_folder}")
else:
    print(f"Folder already exists: {csv_folder}")

print('Rlow=%f'%rlow)
print('Rhigh=%f'%rhigh)
print('Horizontal partitions = '+str(hpar))
print('Vertical partitions = '+str(vpar))

def adc(voltage, v_min=-0.8, v_max=0.8, bits=8):
    """
    Converts an analog voltage to a digital ADC value with saturation.
    """
    levels = 2 ** bits
    step_size = (v_max - v_min) / (levels - 1)

    # Saturate voltage
    voltage = np.clip(voltage, v_min, v_max)


    return np.round((voltage - v_min) / step_size).astype(int)


def dac(adc_value, v_min=-0.8, v_max=0.8, bits=8):
    """
    Converts digital ADC values (scalar or array) back to analog voltages.
    """
    levels = 2 ** bits
    step_size = (v_max - v_min) / (levels - 1)

    adc_value = np.clip(adc_value, 0, levels - 1)  # element-wise clipping

    return v_min + adc_value * step_size


def layer_partition(layer1,layer2, xbar_length, LayerNUM,hpar,vpar, data_dir):
    # updating the resistivity for specific technology node
    l0 = 39e-9 # Mean free path of electrons in Cu
    R=0.3 # probability for electron to reflect at the grain boundary
    layer1_wb = layer1+1 # number of bitcell in a row including weights and bias

    # Determine where vertical and horizontal partitions are (from IMAC Sim)
    # NOTE: if [1,30, 58], the first partition is 1,29, and the second is 30, 57 :) as the top end is noninclusive.
    horizontal_cuts = [1]

    n_hpar=1 # horizontal partition number
    c=1 # column number
    r=1 # row number
    posw_r=open(data_dir+'/'+'posweight'+str(LayerNUM)+".txt", "r") # read the positive line conductances
    for line in posw_r:
        if (float(line)!=0):
            if (r < layer2+1):
                r+=1
            else:
                c+=1
                r=1
                if (c == int(layer1_wb*n_hpar/hpar+min((layer1_wb%hpar)/n_hpar,1)+1)):
                    #print("positive increase horizontal partition")
                    #print(f"row: {r}, col: {c}")
                    n_hpar+=1

                    # Horizontal Partition Here
                    horizontal_cuts.append(c)
                r+=1
        else:
            r+=1
    horizontal_cuts.append(layer1_wb+1)
    posw_r.close()

    # Formally Print horizontal cuts :)
    output = f"Horizontal partitions Layer: {LayerNUM}: "
    output += " | ".join(f"[{horizontal_cuts[i]}:{horizontal_cuts[i+1]-1}]" for i in range(len(horizontal_cuts) - 1))
    print(output)
    print(f"Difference: {np.diff(horizontal_cuts)}")
    print(f"Note: Last Horizontal Partition gets the bias line")

    # Determine Vertical Partitions :)

    vertical_cuts = [1]
    
    # writing the circuit for vertical line parasitic resistances
    for i in range(layer1_wb):
        n_vpar=1 # vertical partition number
        c=i+1 # column number
        for j in range(layer2):
            r=j+1 # row number
            if (i == layer1): # only for the bias line (last line to write btw)
                if (j == 0):
                    temp=1
                elif (j == int(layer2*n_vpar/vpar+min((layer2%vpar)/n_vpar,1))):
                    temp=1
                    n_vpar+=1
                    vertical_cuts.append(r)
                else:
                    temp=1
    
    vertical_cuts.append(layer2+1)
    output = f"Vertical partitions Layer: {LayerNUM}: "
    output += " | ".join(f"[{vertical_cuts[i]}:{vertical_cuts[i+1]-1}]" for i in range(len(vertical_cuts) - 1))
    print(output)
    print(f"Difference: {np.diff(vertical_cuts)}")
    print(f"Note: Each Vertical Guy gets own bias. Each Partition's initial wire connects to vdd")

    # Open file descriptors for all possible things that we need to open (Each guy is xbar_length x 1) :)
    # File descriptor named: partioned_layer_{layer_num}_{hpar}_{vpar}_{split_vpar}.sp
    # We split vpar since we have 32x1 based mac units.
    open_fd = []    # Index into this using hpar, vpar, split_vpar index

    for x_id in range(hpar):
        for y_id in range(vpar):
            # Vertical refers to having to split up input into multiple :)
            # In our case, even though we will have the xbar_length x xbar_length, we will split to 32x1 for parallezability
            # Get vertical cuts length :)
            new_range_low = vertical_cuts[y_id]
            new_range_high = vertical_cuts[y_id+1]
            new_range = (new_range_high - new_range_low)

            for split_vpar in range(new_range):
                open_fd.append((x_id+1, y_id+1, split_vpar+1))

    return (horizontal_cuts, vertical_cuts, open_fd) 


# Learn how to import IMAC sim weights, biases, and partitioning :)
def determine_layer_partitions(nodes, xbar_length, hpar, vpar):
    layers = {}
    layer_cuts = {}

    for i in range(len(nodes)-1):
        hor_cut, vert_cut, layer_keys = layer_partition(nodes[i], nodes[i+1], xbar_length, i+1, hpar[i], vpar[i], data_dir)

        layers[i+1] = layer_keys
        layer_cuts[i+1] = (hor_cut, vert_cut)
    
    return layers, layer_cuts


def load_weights_and_bias(layer_num, layers, layer_cuts, xbar_length):
    # Loads in weights and biases (in np array) based on the layer partition dictionary so easy to add in everything 
    #       when running :)
    # Get weights, (layer2, by layer 1, so each row is output
    weights = pd.read_csv(os.path.join(data_dir, f"W{layer_num}.csv"), header=None).to_numpy()
    transposed_weights = weights.transpose()

    # Read Bias file
    biases = pd.read_csv(os.path.join(data_dir, f"B{layer_num}.csv"), header=None).to_numpy()

    partitioned_weights_and_biases = np.zeros((len(layers[layer_num]), 33))
    index = 0
    for (x_id, y_id, split_r) in layers[layer_num]:
        # Get how things are cut
        hor_cut, vert_cut = layer_cuts[layer_num]
        # Start and end
        # Calculate input ranges :)
        low_range_x = hor_cut[x_id-1]
        high_range_x = hor_cut[x_id]

        if x_id == len(hor_cut)-1:
            high_range_x -= 1

        # Calculate y
        low_range_y = vert_cut[y_id-1]
        global_y_value = (low_range_y - 1) + split_r
        
        #print(f"Y: {global_y_value-1}, X: [{low_range_x-1}:{high_range_x-1}]")
        partitioned_weights_and_biases[index, :high_range_x-low_range_x] = transposed_weights[global_y_value-1, low_range_x-1:high_range_x-1]
        #print(x_id, y_id, split_r, cool_weights.shape)

        # Find biases
        if x_id == len(hor_cut)-1:
            partitioned_weights_and_biases[index, 32] = biases[global_y_value-1, 0]
            
        index+=1

    # for i in range(len(layers[layer_num])):
    #     x_id, y_id, vpar = layers[layer_num][i]
    #     print(f"x_id:{x_id}, y_id: {y_id}, vpar: {vpar}, {partitioned_weights_and_biases[i, :]}")

    return partitioned_weights_and_biases


def load_data(input_array, layer_num, layer, layer_cuts, xbar_length):
    # Load in data from whatever IMAC SIM creates into partitions :)
    hor_cut, vert_cut = layer_cuts[layer_num]
    partitions = layer[layer_num]

    lasana_input_array = np.zeros((len(partitions), xbar_length))
    index = 0
    for (x_id, y_id, split_r) in partitions:
        # Calculate input ranges :)
        low_range_x = hor_cut[x_id-1]
        high_range_x = hor_cut[x_id]

        if x_id == len(hor_cut)-1:
            high_range_x -= 1

        # print(x_id, y_id, split_r)
        # print(low_range_x, high_range_x)
        lasana_input_array[index, :high_range_x-low_range_x] = input_array[low_range_x-1:high_range_x-1]

        index +=1

    return lasana_input_array


def combine_horizontal_partition_outputs(outputs, layer_num, next_layer_neurons, layer, layer_cuts):
    layer_keys = layer[layer_num]
    hor_cut, vert_cut = layer_cuts[layer_num]
    index = 0

    coalesced_output = np.zeros((next_layer_neurons, 1))

    for (x_id, y_id, split_r) in layer_keys:
        # Calculate y
        low_range_y = vert_cut[y_id-1]
        global_y_value = (low_range_y - 1) + split_r
        coalesced_output[global_y_value-1,0] += outputs[index] 
        
        index +=1
    
    return coalesced_output


# -----------
# Load in 10 diff ML Models
model_filepath = os.path.join('../../data', MODEL_RUN_NAME_DIFF_10, 'ml_models')
e_static_model_10 = CatBoostRegressor()
e_static_model_10.load_model(os.path.join(model_filepath, 'pcm_crossbar_catboost_static_energy.cbm'))

e_model_10 = CatBoostRegressor()
e_model_10.load_model(os.path.join(model_filepath, 'pcm_crossbar_catboost_dynamic_energy.cbm'))

l_model_10 = CatBoostRegressor()
l_model_10.load_model(os.path.join(model_filepath, 'pcm_crossbar_catboost_latency.cbm'))

behavior_model_10 = CatBoostRegressor()
behavior_model_10.load_model(os.path.join(model_filepath, 'pcm_crossbar_catboost_output.cbm'))

# Load in 30 diff ML Models
model_filepath = os.path.join('../../data', MODEL_RUN_NAME_DIFF_30, 'ml_models')
e_static_model_30 = CatBoostRegressor()
e_static_model_30.load_model(os.path.join(model_filepath, 'pcm_crossbar_catboost_static_energy.cbm'))

e_model_30 = CatBoostRegressor()
e_model_30.load_model(os.path.join(model_filepath, 'pcm_crossbar_catboost_dynamic_energy.cbm'))

l_model_30 = CatBoostRegressor()
l_model_30.load_model(os.path.join(model_filepath, 'pcm_crossbar_catboost_latency.cbm'))

behavior_model_30 = CatBoostRegressor()
behavior_model_30.load_model(os.path.join(model_filepath, 'pcm_crossbar_catboost_output.cbm'))

# Open Dataset labels and inputs :)
raw_input_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'), header=None).to_numpy()
input_data = np.sign(raw_input_data) * vdd
labels = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'), header=None).to_numpy()

# Calculate Batches :)
num_batches = testnum//testnum_per_batch #calculates the number of batch for the simulation
image_num=0 #number of image in the simulation
testimage=firstimage

# Determine Partitioning for this crossbar array :)
print("Creating Partitions and Mapping Weights and Biases!")
layers, layer_cuts = determine_layer_partitions(nodes, xbar[0], hpar, vpar)
layer_weights = {}

for i in range(len(nodes)-1):
    partitioned_weights = load_weights_and_bias(i+1, layers, layer_cuts, xbar[0])
    print(f"Layer: {i+1}, Weights and Biases Shape: {partitioned_weights.shape}")
    layer_weights[i+1] = partitioned_weights

print("Finished Creating Partitions and Mapping Weights and Biases!")

headers = ['image_num', 'golden_label', 'predicted_label', 'energy'] + \
          [f'output{j}' for j in range(10)]

file = open(csv_name, mode='w', newline='')
writer = csv.writer(file)
writer.writerow(headers)

per_circuit_header = ['circuit_name', 'latency', 'energy', 'output_value', 'previous_output_value']
correct = 0
images_processed = 0

for i in range(num_batches):
    # In SPICE a batch is technically in one single simulation, so we still need to simulate each batch
    # Setup all batch things to maintain inputs and outputs :)
    layer1_crossbar = CrossBar(len(layers[1]), layer_weights[1], e_model_30, l_model_30, e_static_model_30, behavior_model_30, tsampling*10**-9)
    layer2_crossbar = CrossBar(len(layers[2]), layer_weights[2], e_model_30, l_model_30, e_static_model_30, behavior_model_30, tsampling*10**-9)
    layer3_crossbar = CrossBar(len(layers[3]), layer_weights[3], e_model_10, l_model_10, e_static_model_10, behavior_model_10, tsampling*10**-9)

    # Setup all the neurons :)
    layer1_neurons = DigitalNeuron(nodes[1], None, None, None, None, tsampling*10**-9)
    layer2_neurons = DigitalNeuron(nodes[2], None, None, None, None, tsampling*10**-9)
    layer3_neurons = DigitalNeuron(nodes[3], None, None, None, None, tsampling*10**-9)
    
    for j in range(testnum_per_batch):
        real_image_id = (image_num + j) + firstimage
        energy_consumed = 0 

        # Create CSV Header
        filename = os.path.join(csv_folder, f"image_{real_image_id}_inference.csv")
        per_inference_fd = open(filename, mode='w', newline='')
        image_writer = csv.writer(per_inference_fd)
        image_writer.writerow(per_circuit_header)

        #print(f"Image: {real_image_id}")
        is_final_input_of_batch = (j == (testnum_per_batch-1))

        # Run Layer 1
        layer1_input = load_data(input_data[real_image_id, :], 1, layers, layer_cuts, xbar[0])
        #print(layer1_input)

        # Call CrossBar Layer
        layer1_crossbar_output, layer1_crossbar_latency, layer1_crossbar_energy, layer1_crossbar_last_outputs = layer1_crossbar.step(j, layer1_input, is_final_input_of_batch)
        energy_consumed += layer1_crossbar_energy.sum()

        # Save Outputs
        for p, (x_id, y_id, split_r) in enumerate(layers[1]):
            row = [
                f"layer_{1}_{x_id}_{y_id}_{split_r}",
                layer1_crossbar_latency[p],
                layer1_crossbar_energy[p],
                layer1_crossbar_output[p],
                layer1_crossbar_last_outputs[p][0]
            ]
            image_writer.writerow(row)

        # Combine Horizontal Outputs Together
        layer1_crossbar_output_coalesced = combine_horizontal_partition_outputs(layer1_crossbar_output, 1, nodes[1], layers, layer_cuts)
        #print(layer1_crossbar_output_coalesced)

        if USE_QUANTIZATION:
            layer1_crossbar_output_coalesced = dac(adc(layer1_crossbar_output_coalesced, bits=DAC_BITS), bits=DAC_BITS)

        # Run Activation Layer 1
        #print("Layer1 Neuron Output")
        layer1_neuron_output, layer1_neuron_latency, layer1_neuron_energy, layer1_neuron_last_outputs = layer1_neurons.step(j, layer1_crossbar_output_coalesced, is_final_input_of_batch)
        energy_consumed += layer1_neuron_energy.sum()

        if USE_QUANTIZATION:
            layer1_neuron_output = dac(adc(layer1_neuron_output, bits=DAC_BITS), bits=DAC_BITS)

        # Save Outputs
        for k in range(nodes[1]):
            neuron_name = f"Xsig_layer_{1}_{k+1}"
            row = [neuron_name, layer1_neuron_latency[k], layer1_neuron_energy[k], layer1_neuron_output[k], layer1_neuron_last_outputs[k][0]]
            image_writer.writerow(row)

        # Run Layer 2pp
        layer2_input = load_data(layer1_neuron_output, 2, layers, layer_cuts, xbar[0])

        # Call CrossBar Layer
        layer2_crossbar_output, layer2_crossbar_latency, layer2_crossbar_energy, layer2_crossbar_last_outputs = layer2_crossbar.step(j, layer2_input, is_final_input_of_batch)
        energy_consumed += layer2_crossbar_energy.sum()

        # Save Outputs
        for p, (x_id, y_id, split_r) in enumerate(layers[2]):
            row = [
                f"layer_{2}_{x_id}_{y_id}_{split_r}",
                layer2_crossbar_latency[p],
                layer2_crossbar_energy[p],
                layer2_crossbar_output[p],
                layer2_crossbar_last_outputs[p][0]
            ]
            image_writer.writerow(row)


        # Combine Horizontal Outputs Together
        layer2_crossbar_output_coalesced = combine_horizontal_partition_outputs(layer2_crossbar_output, 2, nodes[2], layers, layer_cuts)

        if USE_QUANTIZATION:
            layer2_crossbar_output_coalesced = dac(adc(layer2_crossbar_output_coalesced, bits=DAC_BITS), bits=DAC_BITS)

        # Run Activation Layer 2
        layer2_neuron_output, layer2_neuron_latency, layer2_neuron_energy, layer2_neuron_last_outputs = layer2_neurons.step(j, layer2_crossbar_output_coalesced, is_final_input_of_batch)
        energy_consumed += layer2_neuron_energy.sum()

        if USE_QUANTIZATION:
            layer2_neuron_output = dac(adc(layer2_neuron_output, bits=DAC_BITS), bits=DAC_BITS)

        # Save Outputs
        for k in range(nodes[2]):
            neuron_name = f"Xsig_layer_{2}_{k+1}"
            row = [neuron_name, layer2_neuron_latency[k], layer2_neuron_energy[k], layer2_neuron_output[k], layer2_neuron_last_outputs[k][0]]
            image_writer.writerow(row)

        # Run Layer 3
        layer3_input = load_data(layer2_neuron_output, 3, layers, layer_cuts, xbar[0])

        # Call CrossBar Lyer
        layer3_crossbar_output, layer3_crossbar_latency, layer3_crossbar_energy, layer3_crossbar_last_outputs = layer3_crossbar.step(j, layer3_input, is_final_input_of_batch)
        energy_consumed += layer3_crossbar_energy.sum()

        # Save Outputs
        for p, (x_id, y_id, split_r) in enumerate(layers[3]):
            row = [
                f"layer_{3}_{x_id}_{y_id}_{split_r}",
                layer3_crossbar_latency[p],
                layer3_crossbar_energy[p],
                layer3_crossbar_output[p],
                layer3_crossbar_last_outputs[p][0]
            ]
            image_writer.writerow(row)

        # Combine Horizontal Outputs Together
        layer3_crossbar_output_coalesced = combine_horizontal_partition_outputs(layer3_crossbar_output, 3, nodes[3], layers, layer_cuts)

        if USE_QUANTIZATION:
            layer3_crossbar_output_coalesced = dac(adc(layer3_crossbar_output_coalesced, bits=DAC_BITS), bits=DAC_BITS)

        # Run Activation Layer 3
        layer3_neuron_output, layer3_neuron_latency, layer3_neuron_energy, layer3_neuron_last_outputs = layer3_neurons.step(j, layer3_crossbar_output_coalesced, is_final_input_of_batch)
        energy_consumed += layer3_neuron_energy.sum()

        if USE_QUANTIZATION:
            layer3_neuron_output = dac(adc(layer3_neuron_output, bits=DAC_BITS), bits=DAC_BITS)

        # Save Outputs
        for k in range(nodes[3]):
            neuron_name = f"Xsig_layer_{3}_{k+1}"
            row = [neuron_name, layer3_neuron_latency[k], layer3_neuron_energy[k], layer3_neuron_output[k], layer3_neuron_last_outputs[k][0]]
            image_writer.writerow(row)

        #print(layer3_neuron_output)
        predicted_label = np.argmax(layer3_neuron_output)
        real_label = np.argmax(labels[real_image_id, :])


        row = [real_image_id] + [real_label] + [predicted_label] + [energy_consumed* 10**-12] + list(layer3_neuron_output)
        writer.writerow(row)

        if predicted_label==real_label:
            correct+=1
        images_processed+=1

        per_inference_fd.close()

    # Each batch Print 
    print(f"Batch: {i}, Images Correct: {correct} / {images_processed}, Acc: {correct / images_processed * 100}")

    # End of batch :)
    image_num = image_num + testnum_per_batch
    testimage = testimage + testnum_per_batch

file.close()
print(f"Finished Infernce on {testnum} Images!")
print(f"Top Log File Written to: {csv_name}")

# Print this into the results folder for the headless run
if SAVE_ACC_PATH and not os.path.exists(SAVE_ACC_PATH):
    os.makedirs(SAVE_ACC_PATH, exist_ok=True)

with open(os.path.join(SAVE_ACC_PATH, 'metrics_summary.txt'), 'w') as f:
    f.write(f"LASANA Total Accuracy: {correct} / {images_processed}, {correct / images_processed * 100:.2f}%\n")