# MAC UNIT, gain 30 hyperparameters
RUN_NAME = "mac_unit_diff_30_pcm_mnist_realistic_op_amp_add_binary"
NUMBER_OF_RUNS = 1000                           # Arbitrarily set the number of runs
NUM_PROCESSES = 20                               # Number of maximum number of processes to spin :)
TOTAL_TIME_NS = 500                             # Nanoseconds of runtime
SIM_MIN_STEP_SIZE_NS = 0.05                     # Min Step Size nanoseconds of the simulation 
NUM_INPUT_POINTS = TOTAL_TIME_NS*10             # Get sampling frequency, out of this, but only used for spike generation fidelity as well as how big PWL file is :)
DIGITAL_INPUT_ENFORCEMENT = True
DIGITAL_FREQUENCY = 0.25 * 10**9       # Right now, 250 MHz
SPIKING_INPUT = False                            # Determines if there is spiking input :)
CIRCUIT_STATEFUL = False
INPUT_SEPARATE_VDD = False
SIMULATOR = 'hspice'
SAME_SIGN_WEIGHTS_FRACTION = 0      # Not used
BINARY_INPUT_FRACTION = 0.1
DELAY_RATIO = 0.2   # Forces input to have approximately 20% leak events

# Model SPICE Filepath
# Neuron Model File Locations
MODEL_FILEPATH = 'IMAC_Sim/analog_MAC_32_model_1_diff_30.sp'
OTHER_NECESSARY_FILES_IN_SPICE_RUN_DIRECTORY = ['IMAC_Sim/models', "IMAC_Sim/param.inc"]
LIBRARY_FILES = ['IMAC_Sim/libraries/14nfet.pm', 'IMAC_Sim/libraries/14pfet.pm']

# Circuit Parameters
NUMBER_OF_INPUTS = 32
NUMBER_OF_WEIGHTS = NUMBER_OF_INPUTS
VDD = 0.8
VSS = -0.8
LOAD_CAPACITANCE = 500 * 10**(-15)                                # Farads    ;

# I/O for Logging and Run Creation
INPUT_NET_NAME = []                                               # Name of current input (s) (Requires I in front)
INPUT_NET = []                                                    # Name of interconnected net (s)
INPUT_CURRENT_SRC = False

for i in range(NUMBER_OF_INPUTS):
    INPUT_NET_NAME.append(f"v{i}")
    INPUT_NET.append(f"in{i}")

# For example, Cout, spk -> Cout spk VSS xF     (Cout is the name, spk is the pos connection, VSS is the neg connection, and xF is the capacitance)
OUTPUT_CAPACITANCE_NAME = 'Cout'                                  # Capacitance Name (Requires C in front) Corresponds to input nets :)
OUTPUT_LOAD_CAP_NET = 'out1'                                       # Name of load net


# Additional Knobs (Voltage and Weights that can be turned)
# NOTE: For the rest of the randomly constrained values, we take a four variable tuple containing as follows ("string of net name", minimum, maximum, opt_flag)
# where the opt_flag specify binary knobs here (c for continuous, b for binary)

# If memristors present, weighted like this :)
weight_net_p = "Rwpos{}_1"
weight_net_n = 'Rwneg{}_1'
#r_low = 8500#2500    # Ohms
#r_high = 25500# 100000 # Ohms
r_low = 78000
r_high = 202000

WEIGHT_NET_NAMES_TO_CHANGE = {}
KNOB_PARAMS = []

for i in range(1, NUMBER_OF_INPUTS+1):
    weight_name = f"weight_{i}"
    KNOB_PARAMS.append((weight_name, r_low, r_high, 'b'))
    WEIGHT_NET_NAMES_TO_CHANGE[weight_name] = (weight_net_p.format(i), weight_net_n.format(i))

# Assign bias
KNOB_PARAMS.append(("bias_1", r_low, r_high, 'b'))
WEIGHT_NET_NAMES_TO_CHANGE["bias_1"] = ("Rbpos1", "Rbneg1")

# ---------



# ----------------------------- END Generate Dataset Hyperparameters ------------------------------ #


# ----------------------------- BEGIN Analyze Dataset Hyperparameters ------------------------------ #

# Hyperparameters (MAC Unit)
DF_FILENAME = "test_11_2_2024.csv"              # Output filename for dataset

# Columns in output dataset
columns = ["Run_Name", "Run_Number", "Event_Type", 'Event_Start_Index', 'Event_End_Index', 'Digital_Time_Step', "Input_Total_Time", "Energy", "Latency", "Output_Value", "Last_Output_Value"]

KNOB_NAMES = []
for i in range(NUMBER_OF_INPUTS):
    KNOB_NAMES.append(f"weight_{i+1}")

KNOB_NAMES.append("bias_1")

columns += KNOB_NAMES

PLOT_RUNS = False
VERBOSE= False

# DATASET PARAMETERS
INPUT_WELL_DEFINED = True

# Circuit Parameters
INPUT_NAME = [] 

for i in range(NUMBER_OF_INPUTS):
    INPUT_NAME.append(f'v{i}')

OUTPUT_NAME = 'out1'

# Power Calculations (Since we have VDD, VSS we just use both)
VDD_VOLTAGE = 'vdd'
VDD_CURRENT = 'i(vdd)'
VSS_VOLTAGE = 'vss'
VSS_CURRENT = 'i(vss)'

# ----------------------------- END Analyze Dataset Hyperparameters ------------------------------ #

# ----------------------------- BEGIN MAC Model Training Hyperparameters ------------------------------ #
PLOT_MATPLOTLIB_FIGS = False
SAVE_FIGS = False
SAVE_CATBOOST_MODEL = True
SAVE_CATBOOST_CPP = False
SAVE_MLP_MODEL = True
SAVE_PYTORCH_MLP_MODEL = False
VALIDATION_SPLIT = 0.15
TRAIN_TEST_SPLIT = 0.15      # Percentage of dataset dedicated to test set

LIST_OF_COLUMNS_X_MAC = ["Input_Total_Time", "Last_Output_Value"]

# Input Voltages
for i in range(NUMBER_OF_INPUTS):
    LIST_OF_COLUMNS_X_MAC.append(f"v{i}")

# Input Weights
for i in range(NUMBER_OF_WEIGHTS):
    LIST_OF_COLUMNS_X_MAC.append(f"weight_{i+1}")
LIST_OF_COLUMNS_X_MAC.append("bias_1")
# ----------------------------- END MAC Model Training Hyperparameters ------------------------------ #