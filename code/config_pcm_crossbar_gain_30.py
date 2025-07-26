# ----------------------------- BEGIN Generate Dataset Hyperparameters ------------------------------ #
# Spiking Circuit Run Hyperparameters
RUN_NAME = 'mac_unit_diff_30_run'                           # Name of the run that will be created in data/
NUMBER_OF_RUNS = 1000                                       # Number of testbenches / randomized SPICE runs
NUM_PROCESSES = 20                                          # Maximum number of processes that can be spun for SPICE simulations (NOTE: Each process uses a CAD license)
TOTAL_TIME_NS = 500                                         # Nanoseconds of runtime / SPICE simulation run
SIM_MIN_STEP_SIZE_NS = 0.05                                 # Minimum step size in nanoseconds of the simulation 
NUM_INPUT_POINTS = TOTAL_TIME_NS*10                         # Determines fidelity of input that is used for generation of .PWL files (as well as how big they are)
DIGITAL_INPUT_ENFORCEMENT = True                            # Enforce that inputs can only occur at digital timesteps (If False, deprecated)
DIGITAL_FREQUENCY = 0.25 * 10**9                            # Digital Backend Frequency 
SPIKING_INPUT = False                                       # Whether the input is a spiking signal or a more traditional steady state signal
INPUT_SEPARATE_VDD = False                                  # Input voltage / current sources are connected to their own VDD rail (makes it easier to decompose energies later on))
SIMULATOR = 'hspice'                                        # Simulator: Current Synopsys HSPICE and Cadence Spectre are supported. LTSpice is deprecated.

# Circuit-Specific Parameters
VDD = 0.8                                                   # VDD Rail
VSS = -0.8                                                  # VSS Rail
NUMBER_OF_INPUTS = 32                                       # Number of input nets
LOAD_CAPACITANCE = 500 * 10**(-15)                          # Farads    ;

# I/O for Logging and Run Creation
INPUT_NET_NAME = []                                               # Name of current input (s) (Requires I in front)
INPUT_NET = []                                                    # Name of interconnected net (s)

for i in range(NUMBER_OF_INPUTS):
    INPUT_NET_NAME.append(f"v{i}")
    INPUT_NET.append(f"in{i}")

MODEL_FILEPATH = '../data/pcm_crossbar_diff30_spice_files/analog_MAC_32_model_1_diff30.sp'
OTHER_NECESSARY_FILES_IN_SPICE_RUN_DIRECTORY = ['../data/pcm_crossbar_diff30_spice_files/models', "../data/pcm_crossbar_diff30_spice_files/param.inc"]
LIBRARY_FILES = ['../data/pcm_crossbar_diff30_spice_files/libraries/14nfet.pm', '../data/pcm_crossbar_diff30_spice_files/libraries/14pfet.pm']

# For example, Cout, spk -> Cout spk VSS xF     (Cout is the name, spk is the pos connection, VSS is the neg connection, and xF is the capacitance)
OUTPUT_CAPACITANCE_NAME = 'Cout'               # Capacitance Name (Requires C in front) 
CIRCUIT_STATEFUL = False                       # Enable if circuit is Stateful
OUTPUT_LOAD_CAP_NET = 'out1'                   # Name of load net

SUBCIRCUIT_DEFINITION = "Xlayer vdd vss 0 "
for i in range(NUMBER_OF_INPUTS):
    SUBCIRCUIT_DEFINITION += f"in{i} "
SUBCIRCUIT_DEFINITION += f"{OUTPUT_LOAD_CAP_NET} layer"

# Additional circuit knobs that can be changed that we want to randomize and treat as parameters that can change in the circuit
r_low = 78000
r_high = 202000

weight_net_p = "Rwpos{}_1"
weight_net_n = 'Rwneg{}_1'
WEIGHT_NET_NAMES_TO_CHANGE = {}
KNOB_PARAMS = []

for i in range(1, NUMBER_OF_INPUTS+1):
    weight_name = f"weight_{i}"
    KNOB_PARAMS.append((weight_name, r_low, r_high, 'b'))
    WEIGHT_NET_NAMES_TO_CHANGE[weight_name] = (weight_net_p.format(i), weight_net_n.format(i))

# Assign bias
KNOB_PARAMS.append(("bias_1", r_low, r_high, 'b'))
WEIGHT_NET_NAMES_TO_CHANGE["bias_1"] = ("Rbpos1", "Rbneg1")

# ---
# Input Parameters
INPUT_CURRENT_SRC = False

NUMBER_OF_WEIGHTS = 32                                           # In the spiking input case, the weight is tied directly to a spike. 
# WEIGHT_LOW = -1                                                 # If custom spike input, relative to input weight ; Otherwise, 
# WEIGHT_HIGH = 1                                                 # If custom spiek input, relative to input weight; Otherwise, 

SAME_SIGN_WEIGHTS_FRACTION = 0      # Not used
BINARY_INPUT_FRACTION = 0.1
DELAY_RATIO = 0.2   # Forces input to have approximately 20% leak events

# ----------------------------- END Generate Dataset Hyperparameters ------------------------------ #


# ----------------------------- BEGIN Analyze Dataset Hyperparameters ------------------------------ #
# Hyperparameters (MAC Unit)
DF_FILENAME = "pcm_crossbar_gain_30_dataset.csv"              # Output filename for dataset

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

# ----------------------------- BEGIN Model Training Hyperparameters ------------------------------ #
DETERMINISTIC = True                    # Sets the random seeds for weight initialization to RANDOM_SEED below.  
RANDOM_SEED = 42                        # Only if DETERMINISTIC is True. Makes the training the same every time for MLP and CatBoost
PLOT_MATPLOTLIB_FIGS = False            # Show matplotlib figures
SAVE_FIGS = True                        # Save correlation plots
SAVE_CATBOOST_MODEL = True              # Self Explanatory. Saves all of the catboost models
SAVE_CATBOOST_CPP = False               # Saves Catboost but in C++ 
SAVE_MLP_MODEL = True                   # Save MLP, sklearn models
SAVE_PYTORCH_MLP_MODEL = False          # Save MLP, PyTorch models (not supported right now)
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

# ----------------------------- END Spike Model Training Hyperparameters ------------------------------ #