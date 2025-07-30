import torch
import dill
import time
import numpy as np
import os
from catboost import CatBoostRegressor, CatBoostClassifier
import joblib
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg') # set the backend before importing pyplot
import matplotlib.pyplot as plt
from NeuronLayer import *
import random
# Make sure all seeds are the same
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -----
## Hyperparameters 
BATCH_SIZE = 1                             # Number of inferences to run per simulation before restarting simulation (ONLY SUPPORT BATCH OF 1)
NUMBER_OF_TIMESTEPS_PER_INFERENCE = 100     # Number of timesteps for each input image
PYTORCH_MODEL_FILE = '../../data/spiking_mnist_model.pt'
PLOT_THINGS = False
NUM_INFERENCES = 100
SAVE_LOGS = True
WEIGHT_SCALING = 1                         # Scaling of weights (to account for leak) 
INPUT_SCALING = 2                          # Scaling of inputs between layers (note that input scaling implicitly affects weight scaling since weight scaling is applied first, then input scaling)

# Neuron Hyperparameters
LASANA_RUN_NAME = 'spiking_neuron_run'#'explicit_edge_case_increase_spk_smaller_knob_range_3_10_25'#'explicit_edge_case_review_0.1_3_10_2025'#'larger_width_reset_2_6_2025'#'leak_works_1_29'
LASANA_MODELS_FD = os.path.join("../../data", LASANA_RUN_NAME, 'ml_models')
LOAD_MLP_MODELS = True                     # Alternative is to load in CatBoost models :)
DT = 5*10**-9       # Digital backend timestep

NEURON_PARAMETERS = [("V_sf", 0.5), ("V_adap", 0.5), ("V_leak", 0.57), ("V_rtr", 0.5)]      # Neuron parameters :)
STARTING_VOLTAGE_STATE = 0

# --------------------------------------

## Important Helper Functions
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

# --------------------------------------
## - Start of Script -
# Load in MNIST Dataset from 0 to 1

# Make logs folder if does not exist
lasana_log_folder = os.path.join('../../data', "lasana_spiking_mnist_logs")

if SAVE_LOGS:
    # Create ML Library
    if not os.path.exists(lasana_log_folder):
        os.makedirs(lasana_log_folder)

transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.MNIST(root='../../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
num_batches = len(testloader)

# Load in Model weights for Pytorch
mnist_model = torch.load(PYTORCH_MODEL_FILE,pickle_module=dill,map_location=torch.device("cpu"))
model_weights = {}      # keys are "fc1.weight", and fc2.weight
for param_name, param in mnist_model.named_parameters():
    model_weights[param_name] = param.detach().numpy()

if LOAD_MLP_MODELS:
    print("Load MLP Models")
    # Load in the std scaler
    random_seed = 42
    std_scaler_name = 'ml_standard_scalar_random_seed_' + str(random_seed) + ".joblib"
    full_path = os.path.join(LASANA_MODELS_FD, std_scaler_name)

    # Check if the path exists
    if os.path.exists(full_path):
        std_scaler = joblib.load(full_path)
    else:
        print(f"ERROR: Standard Scaler Path at {full_path} does not exist!")
        exit(1)

    # Note 8_5 is the fast one and accurate one: "mlp_spike_or_not_8_5.joblib"
    e_static_model = joblib.load(os.path.join(LASANA_MODELS_FD, 'spiking_neuron_mlp_static_energy.joblib'))
    e_model = joblib.load(os.path.join(LASANA_MODELS_FD, 'spiking_neuron_mlp_dynamic_energy.joblib'))
    l_model = joblib.load(os.path.join(LASANA_MODELS_FD, 'spiking_neuron_mlp_latency.joblib'))
    neuron_state_model = joblib.load(os.path.join(LASANA_MODELS_FD, "spiking_neuron_mlp_neuron_state.joblib"))
    spike_or_not_model = joblib.load(os.path.join(LASANA_MODELS_FD, "spiking_neuron_mlp_spike_or_not.joblib"))
else:
    # Load in CatBoost Models from LASANA
    e_static_model = CatBoostRegressor()
    e_static_model.load_model(os.path.join(LASANA_MODELS_FD, 'spiking_neuron_catboost_static_energy.cbm'))

    e_model = CatBoostRegressor()
    e_model.load_model(os.path.join(LASANA_MODELS_FD, 'spiking_neuron_catboost_dynamic_energy.cbm'))

    l_model = CatBoostRegressor()
    l_model.load_model(os.path.join(LASANA_MODELS_FD, 'spiking_neuron_catboost_latency.cbm'))

    neuron_state_model = CatBoostRegressor()
    neuron_state_model.load_model(os.path.join(LASANA_MODELS_FD, 'spiking_neuron_catboost_neuron_state.cbm'))

    spike_or_not_model = CatBoostClassifier()
    spike_or_not_model.load_model(os.path.join(LASANA_MODELS_FD, 'spiking_neuron_catboost_spike_or_not.cbm'))
    std_scaler = None

print("ML Models Loaded In")

# --------------------------------------
# Get Housekeeping setup so that we can keep track of everything
neuron_params_runtime = np.array([param[1] for param in NEURON_PARAMETERS])
current_batch = 0
total_img = 0
total_correct = 0
start_time = time.time()

# Run for each batch :)
for images, labels in testloader:
    # Keep track of time
    iter_start = time.time()  # Start time for this iteration

    # Convert to Poisson Spike Train
    img = images[0].squeeze(0)  # Remove channel dimension, shape becomes (28, 28)
    real_batch_size = images.shape[0]
    spike_train = poisson_spike_train(images, NUMBER_OF_TIMESTEPS_PER_INFERENCE).numpy()
    batch_num_timesteps = spike_train.shape[0]

    # Create Neural Network
    layer_0 = NeuronLayer("layer0",784, batch_num_timesteps, DT, neuron_params_runtime, model_weights['fc1.weight'],
                          128, WEIGHT_SCALING, std_scaler, 
                          neuron_state_model, e_static_model, spike_or_not_model, e_model, l_model, LOAD_MLP_MODELS)
    layer_1 = NeuronLayer("layer1",128, batch_num_timesteps, DT, neuron_params_runtime, model_weights['fc2.weight'],
                          10, WEIGHT_SCALING, std_scaler, 
                          neuron_state_model, e_static_model, spike_or_not_model, e_model, l_model, LOAD_MLP_MODELS)
    layer_2 = NeuronLayer("layer2",10, batch_num_timesteps, DT, neuron_params_runtime, None,
                          0, WEIGHT_SCALING, std_scaler, 
                          neuron_state_model, e_static_model, spike_or_not_model, e_model, l_model, LOAD_MLP_MODELS)

    if PLOT_THINGS:
        # Plot the image
        plt.figure(0)
        plt.imshow(img, cmap="gray")
        plt.title(f"Label: {labels[0].item()}")  # Show label as title
        plt.axis("off")  # Hide axis

        # Plot the Poisson Trains
        plt.figure(1)
        for i in range(784):
            plt.plot(np.arange(BATCH_SIZE * NUMBER_OF_TIMESTEPS_PER_INFERENCE), spike_train[:,i], label=f'{i}')
        plt.xlabel("Number of Timesteps")
        plt.ylabel("Spike train amplitude")
        plt.show()
        exit()
        break  # Only show the first image

    
    spiking_event_layer_1 = np.zeros(128)
    spiking_event_layer_2 = np.zeros(10)
    batch_output = np.zeros((batch_num_timesteps, 10))

    # Network Forward Pass
    for t in range(batch_num_timesteps):
        #print(f"Batch: {current_batch} / {num_batches}, Timestep: {t} / {batch_num_timesteps}")
        spiking_event_layer_0 = spike_train[t, :] * INPUT_SCALING
        # Loop through each timestep

        # Step through the network
        next_spiking_layer1 = layer_0.step(spiking_event_layer_0)
        next_spiking_layer2 = layer_1.step(spiking_event_layer_1)
        out = layer_2.step(spiking_event_layer_2)

        # Save Output?
        batch_output[t, :] = out

        # Update spiking inputs
        spiking_event_layer_1 = next_spiking_layer1 * INPUT_SCALING 
        spiking_event_layer_2 = next_spiking_layer2 * INPUT_SCALING

    # Housekeeping after running inference :)
    if SAVE_LOGS:
        # NOTE: This only works with batch size of 1 :)
        layer_0.save_layer_information(current_batch)
        layer_1.save_layer_information(current_batch)
        layer_2.save_layer_information(current_batch)

    # Now figure out if the labels make sense
    batch_correct = 0

    for i in range(real_batch_size):
        output_timesteps = batch_output[i*NUMBER_OF_TIMESTEPS_PER_INFERENCE:(i+1)*NUMBER_OF_TIMESTEPS_PER_INFERENCE, :]
        summed_guys = np.sum(output_timesteps, axis=0)
        print(summed_guys)
        max_index = np.argmax(summed_guys)

        print(f"Batch: {current_batch}, Image: {i}, Predicted: {max_index}, Label: {labels[i]}")
        #print(summed_guys)
        if max_index == labels[i]:
            batch_correct+=1

    batch_acc = batch_correct / real_batch_size * 100
    print(f"Batch: {current_batch}, Correct: {batch_correct} / {real_batch_size}, Accuracy (%): {batch_acc:.2f}%")

    # Add total correct
    total_correct += batch_correct
    total_img += real_batch_size

    # Total Number of Batch Accuracy
    total_acc_curr = total_correct / total_img * 100
    print(f"Cumulative Total Correct: {total_correct} / {total_img}, Accuracy (%): {total_acc_curr:.2f}%")

    # Keep track of time :)
    iter_end = time.time()  # End time for this iteration
    elapsed_time = iter_end - start_time  # Time since the loop started
    avg_time_per_iter = elapsed_time / (current_batch +1) # Average time per iteration
    estimated_total_time = avg_time_per_iter * num_batches  # Projected total time
    remaining_time = estimated_total_time - elapsed_time  # Estimated time left

    print(f"Iteration {current_batch+1}/{num_batches} - Elapsed: {elapsed_time:.2f}s, "
    f"Estimated Total: {estimated_total_time:.2f}s, Remaining: {remaining_time:.2f}s")

    current_batch +=1

    # Short circuit if we heave reached the right number of inferences
    if total_img >= NUM_INFERENCES:
        break

# Print full accuracies :)
total_acc = total_correct / total_img * 100
print(f"Total Accuracy: {total_correct} / {total_img}, {total_acc:.2f}%")
