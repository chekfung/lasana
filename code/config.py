# ----------------------------- BEGIN Generate Dataset Hyperparameters ------------------------------ #
# Spiking Circuit Hyperparameters
# RUN_NAME = 'spiking_dataset_generation_speed' #'explicit_edge_case_increase_spk_smaller_knob_range_3_10_25'             #"add_weights_1000_runs_vary_0.5_1.5"#"20000_training_set_with_weights"#"PLAYGROUND_V3_VARY_0.5_1.5_1000_RUNS_3_28_fix_capacitance"
# NUMBER_OF_RUNS = 2000                           # Arbitrarily set the number of runs
# NUM_PROCESSES = 20                               # Number of maximum number of processes to spin :)
# TOTAL_TIME_NS = 500                             # Nanoseconds of runtime
# SIM_MIN_STEP_SIZE_NS = 0.01                     # Min Step Size nanoseconds of the simulation 
# NUM_INPUT_POINTS = TOTAL_TIME_NS*10                         # Get sampling frequency, out of this, but only used for spike generation fidelity as well as how big PWL file is :)
# DIGITAL_INPUT_ENFORCEMENT = True
# DIGITAL_FREQUENCY = 0.2 * 10**9                 # Right now, 200 MHz    (250Mhz for DAC submission)
# CIRCUIT_STATEFUL = True
# INPUT_SEPARATE_VDD = True
# SIMULATOR = 'spectre'
# VDD = 1.5           # This should be 1.5V for DAC 2025 submission
# VSS = 0
# SPIKING_INPUT = True                            # Determines if there is spiking input :)
# NUMBER_OF_INPUTS = 1
# NUMBER_OF_WEIGHTS = 1
# LOAD_CAPACITANCE = 500 * 10**(-15)                                # Farads    ; #TODO: DAC2025 is 500 * 10**(-15)
# WEIGHT_LOW = -2
# WEIGHT_HIGH = 2
# SAME_SIGN_WEIGHTS_FRACTION = 0.1                                 # Fraction of runs that will have all the same signed weights / inputs (either negative or positive with 50/50 chance)

# # I/O 
# INPUT_NET_NAME = ['I_inj']                                        # Name of current input (s) (Requires I in front)
# INPUT_NET = ['spikes']                                            # Name of interconnected net (s)
# INPUT_CURRENT_SRC = True


# # For example, Cout, spk -> Cout spk VSS xF     (Cout is the name, spk is the pos connection, VSS is the neg connection, and xF is the capacitance)
# OUTPUT_CAPACITANCE_NAME = 'Cout'                                  # Capacitance Name (Requires C in front) Corresponds to input nets :)
# OUTPUT_LOAD_CAP_NET = 'spk'                                       # Name of interconnected load net Corresponds to input nets :)

# # Circuit Definition to run
# SUBCIRCUIT_DEFINITION = f'X1 {INPUT_NET[0]} leak sf rtr adap {OUTPUT_LOAD_CAP_NET} vdd 0 lif_neuron'

# # Model SPICE Filepath
# # Neuron Model File Locations
# MODEL_FILEPATH = '../spiking_neuron_models/spice/analog_lif_neuron_2_6_2025.sp'
# OTHER_NECESSARY_FILES_IN_SPICE_RUN_DIRECTORY = []
# LIBRARY_FILES = ['../spiking_neuron_models/libraries/45nm_LP.pm']

# # NOTE: For the rest of the randomly constrained values, we take a four variable tuple containing as follows ("string of net name", minimum, maximum, opt_flag)
# # where the opt_flag specify binary knobs here (c for continuous, b for binary)
# # (0.5, 1.5) for the dac submission :)
# KNOB_PARAMS = [("V_sf", 0.5, 0.8, 'c'), ("V_adap", 0.5, 0.8, 'c'), ("V_leak", 0.4, 0.8, 'c'), ("V_rtr", 0.5, 0.8, 'c')]  # NOTE: Default values for all of these knobs are specified in the design. We are simply changing them here.

# # ---
# # Spiking Neuron Input Parameters
# if SPIKING_INPUT:
#     DELAY_RATIO = 0.3   # Forces input to have approximately 30% leak events
#     REFRACTORY_PERIOD = 5 * 10**(-10)               # seconds   ; Time before another spike can occur on the same neuron 
#     RUN_TIMEOUT = 10                                # seconds   ; If spike mapping solution not found in x seconds, throw error.
#     CIRCUIT_SIM_NUM_SPIKES = ("Num_Input_Spikes", 0, 300, "c")          # Number of simulated input spikes in one simulation (minimum, maximum)
#     CIRCUIT_FAN_IN_RANGE = ("Circuit_Fan_In", 4, 32, "c")                # Neuron FAN IN range (minimum, maximum)
#     CIRCUIT_FAN_OUT_RANGE = ("Circuit_Fan_Out", 1, 1, "c")              # Neuron FAN OUT range (minimum, maximum)
#     PLOT_SPIKE_BOUNDS = False
#     OUTPUT_SPIKE_NAME = 'i(C)'            # This is for the footprint :)

#     # Block Spike
#     # Note: We denote block spike as a PWM of a set voltage where the height of the block is determined by the raw spike
#     #       total charge. The total charge is matched, but implemented as a PWM spike. Changing the time step length will
#     #       maintain the same charge as the original spike footprint above, meaning the height will be lower. We use this
#     #       to better imitate hybrid spiking systems, such as BrainScales-2.
#     CUSTOM_SPIKE = False
#     CUSTOM_SPIKE_NUM_TIME_STEP_LENGTH = 10

#     if SIMULATOR == 'spectre':
#         # DAC 2025 Submission
#         #SPICE_FOOTPRINT_FILE = '../spiking_neuron_models/spice/analog_lif_neuron_playground_spectre.sp'
#         #SPIKE_START = 292
#         #SPIKE_END = 354
#         SPICE_FOOTPRINT_FILE = '../spiking_neuron_models/spice/analog_lif_neuron_playground_spectre_larger_width_reset.sp'
#         SPIKE_START = 306
#         SPIKE_END = 361
        
#     else:
#         # TODO: get rid of this bloat where we are keeping the old DAC2025 submission stuff
#         # DAC2025 Submission
#         SPICE_FOOTPRINT_FILE = '../spiking_neuron_models/spice/analog_lif_neuron_playground.sp'
#         SPIKE_START = 220
#         SPIKE_END = 253

# else: 
#     BINARY_INPUT = True                                           # Either VDD or VSS (binary), or some uniform disribution between the two otherwise


# ---------------------------

# MAC UNIT hyperparameters

RUN_NAME = "mac_unti_diff_30_pcm_mnist_realistic_op_amp_add_binary"#"mac_unit_bias_diff_10_pcm_mnist_FINAL_FINAL_FINER_FIDELITY_0_8"#"mac_unit_11_14_20ua"#"test_run_11_11_2024_static_0.2"                     #"add_weights_1000_runs_vary_0.5_1.5"#"20000_training_set_with_weights"#"PLAYGROUND_V3_VARY_0.5_1.5_1000_RUNS_3_28_fix_capacitance"
NUMBER_OF_RUNS = 1000                           # Arbitrarily set the number of runs
NUM_PROCESSES = 20                               # Number of maximum number of processes to spin :)
TOTAL_TIME_NS = 500                             # Nanoseconds of runtime
SIM_MIN_STEP_SIZE_NS = 0.05                     # Min Step Size nanoseconds of the simulation 
NUM_INPUT_POINTS = 5000                      # Get sampling frequency, out of this, but only used for spike generation fidelity as well as how big PWL file is :)
DIGITAL_INPUT_ENFORCEMENT = True
DIGITAL_FREQUENCY = 0.25 * 10**9       # Right now, 200 MHz
SPIKING_INPUT = False                            # Determines if there is spiking input :)
CIRCUIT_STATEFUL = False
INPUT_SEPARATE_VDD = False
SIMULATOR = 'hspice'
SAME_SIGN_WEIGHTS_FRACTION = 0      # Not used

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
OUTPUT_LOAD_CAP_NET = 'out1'                                       # Name of interconnected load net Corresponds to input nets :)
SUBCIRCUIT_DEFINITION = "Xlayer vdd vss 0 "
for i in range(NUMBER_OF_INPUTS):
    SUBCIRCUIT_DEFINITION += f"in{i} "
SUBCIRCUIT_DEFINITION += f"{OUTPUT_LOAD_CAP_NET} layer"

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

if not SPIKING_INPUT:
    #BINARY_INPUT = False     # Forces nonspiking input to either be vss or vdd
    BINARY_INPUT_FRACTION = 0.1
    DELAY_RATIO = 0.2   # Forces input to have approximately 20% leak events

# ---------------------





# MAC ARRAY PARAMS
# RUN_NAME = "mac_array_test"#"mac_unit_11_14_20ua"#"test_run_11_11_2024_static_0.2"                     #"add_weights_1000_runs_vary_0.5_1.5"#"20000_training_set_with_weights"#"PLAYGROUND_V3_VARY_0.5_1.5_1000_RUNS_3_28_fix_capacitance"
# NUMBER_OF_RUNS = 1000                              # Arbitrarily set the number of runs
# NUM_PROCESSES = 20                               # Number of maximum number of processes to spin :)
# TOTAL_TIME_NS = 500                             # Nanoseconds of runtime
# SIM_MIN_STEP_SIZE_NS = 0.05                     # Min Step Size nanoseconds of the simulation 
# NUM_INPUT_POINTS = 5000                      # Get sampling frequency, out of this, but only used for spike generation fidelity as well as how big PWL file is :)
# DIGITAL_INPUT_ENFORCEMENT = True
# DIGITAL_FREQUENCY = 0.25 * 10**9       # Right now, 200 MHz
# SPIKING_INPUT = False                            # Determines if there is spiking input :)
# CIRCUIT_STATEFUL = False
# INPUT_SEPARATE_VDD = False
# SIMULATOR = 'hspice'

# # Model SPICE Filepath
# # Neuron Model File Locations
# MODEL_FILEPATH = 'IMAC_Sim/layer3_no_partition_32_length.sp'
# OTHER_NECESSARY_FILES_IN_SPICE_RUN_DIRECTORY = ['IMAC_Sim/models', "IMAC_Sim/param.inc"]
# LIBRARY_FILES = ['IMAC_Sim/libraries/14nfet.pm', 'IMAC_Sim/libraries/14pfet.pm']

# # Circuit Parameters
# NUMBER_OF_INPUTS = 32
# NUMBER_OF_WEIGHTS = 320
# VDD = 0.8
# VSS = -0.8
# LOAD_CAPACITANCE = 50000 * 10**(-15)                                # Farads    ;

# # I/O for Logging and Run Creation
# INPUT_NET_NAME = []                                               # Name of current input (s) (Requires I in front)
# INPUT_NET = []                                                    # Name of interconnected net (s)
# for i in range(NUMBER_OF_INPUTS):
#     INPUT_NET_NAME.append(f"v{i}")
#     INPUT_NET.append(f"in{i}")

# # For example, Cout, spk -> Cout spk VSS xF     (Cout is the name, spk is the pos connection, VSS is the neg connection, and xF is the capacitance)
# OUTPUT_CAPACITANCE_NAME = 'Cout'                                  # Capacitance Name (Requires C in front) Corresponds to input nets :)
# OUTPUT_LOAD_CAP_NET = 'out1'                                       # Name of interconnected load net Corresponds to input nets :)
# SUBCIRCUIT_DEFINITION = "Xlayer vdd vss 0 "
# for i in range(NUMBER_OF_INPUTS):
#     SUBCIRCUIT_DEFINITION += f"in{i} "
# SUBCIRCUIT_DEFINITION += f"{OUTPUT_LOAD_CAP_NET} layer"

# # Additional Knobs (Voltage and Weights that can be turned)
# # NOTE: For the rest of the randomly constrained values, we take a four variable tuple containing as follows ("string of net name", minimum, maximum, opt_flag)
# # where the opt_flag specify binary knobs here (c for continuous, b for binary)

# # If memristors present, weighted like this :)
# weight_net_p = "Rwpos{}_{}"
# weight_net_n = 'Rwneg{}_{}'
# r_low = 2500    # Ohms
# r_high = 100000 # Ohms

# WEIGHT_NET_NAMES_TO_CHANGE = {}
# KNOB_PARAMS = []

# for i in range(NUMBER_OF_WEIGHTS):
#     weight_name = f"weight_{i+1}"

#     bot = int(i / NUMBER_OF_INPUTS) + 1
#     top = i % NUMBER_OF_INPUTS + 1

#     KNOB_PARAMS.append((weight_name, r_low, r_high, 'b'))
#     WEIGHT_NET_NAMES_TO_CHANGE[weight_name] = (weight_net_p.format(top, bot), weight_net_n.format(top, bot))

# # ---

# if not SPIKING_INPUT:
#     BINARY_INPUT = False     # Forces nonspiking input to either be vss or vdd
#     DELAY_RATIO = 0.2   # Forces input to have approximately 20% leak events



# END HYPERPARAMETERS

# ----------------------------- END Generate Dataset Hyperparameters ------------------------------ #


# ----------------------------- BEGIN Analyze Dataset Hyperparameters ------------------------------ #

# ---------------------------------------------------------------------
# # Hyperparameters (Spiking Circuit)
# #RUN PARAMETERS
# DF_FILENAME = "test1.csv"#"test.csv"#"mlcad_final_fix_6_3_weights_included.csv" #"dataset_all_energies_annotated.csv"

# # Columns in output dataset
# columns = ["Run_Name", "Run_Number", "Event_Type", 'Event_Start_Index', 'Event_End_Index', 'Digital_Time_Step',"Input_Peak_Amplitude", "Input_Total_Charge", "Weight","Input_Total_Time", "Cap_Voltage_At_Input_Start", "Cap_Voltage_At_Output_End", "Energy", "Latency"]
# KNOB_NAMES = ["Circuit_Fan_In_I_inj", "Circuit_Fan_Out", "V_sf", "V_adap", "V_leak", "V_rtr"]
# columns += KNOB_NAMES

# PLOT_RUNS = False
# VERBOSE= False

# # DATASET PARAMETERS
# INPUT_WELL_DEFINED = True

# # Neuron Calculations
# INPUT_NAME = ['i(I_inj)']
# STATE_NET = 'v(spikes)'
# OUTPUT_NAME = 'i(Cout)'

# # Power Calculations
# VDD_VOLTAGE = 'v(vdd)'
# VDD_CURRENT = 'i(Vdd)'
# VSS_VOLTAGE = None
# VSS_CURRENT = None

# # ------------


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

#-----------


# Hyperparameters (MAC Array)
# DF_FILENAME = "test_11_2_2024.csv"              # Output filename for dataset

# # Columns in output dataset
# columns = ["Run_Name", "Run_Number", "Event_Type", 'Event_Start_Index', 'Event_End_Index', 'Digital_Time_Step', "Input_Total_Time", "Energy", "Latency", "Output_Value", "Last_Output_Value"]

# KNOB_NAMES = []
# NUMBER_OF_WEIGHTS = 320
# for i in range(NUMBER_OF_WEIGHTS):
#     KNOB_NAMES.append(f"weight_{i+1}")

# columns += KNOB_NAMES

# PLOT_RUNS = False
# VERBOSE= False

# # DATASET PARAMETERS
# INPUT_WELL_DEFINED = True

# # Circuit Parameters
# INPUT_NAME = [] 
# NUMBER_OF_INPUTS = 32

# for i in range(NUMBER_OF_INPUTS):
#     INPUT_NAME.append(f'v{i}')

# OUTPUT_NAME = 'out1'

# # Power Calculations (Since we have VDD, VSS we just use both)
# VDD_VOLTAGE = 'vdd'
# VDD_CURRENT = 'i(vdd)'
# VSS_VOLTAGE = 'vss'
# VSS_CURRENT = 'i(vss)'

# ----------------------------- END Analyze Dataset Hyperparameters ------------------------------ #

# ----------------------------- BEGIN Spike Model Training Hyperparameters ------------------------------ #
PLOT_MATPLOTLIB_FIGS = False
SAVE_FIGS = False
SAVE_CATBOOST_MODEL = True
SAVE_CATBOOST_CPP = False
SAVE_MLP_MODEL = True
SAVE_PYTORCH_MLP_MODEL = False
VALIDATION_SPLIT = 0.15
TRAIN_TEST_SPLIT = 0.15      # Percentage of dataset dedicated to test set
LIST_OF_COLUMNS_X = ["Cap_Voltage_At_Input_Start", "Weight", "Input_Total_Time", "V_sf", "V_adap", "V_leak", "V_rtr"]
# ----------------------------- END Spike Model Training Hyperparameters ------------------------------ #

# ----------------------------- BEGIN MAC Model Training Hyperparameters ------------------------------ #
# TODO: We have not support the new test validation train test split for mac units :)
LIST_OF_COLUMNS_X_MAC = ["Input_Total_Time", "Last_Output_Value"]

# Input Voltages
for i in range(NUMBER_OF_INPUTS):
    LIST_OF_COLUMNS_X_MAC.append(f"v{i}")

# Input Weights
for i in range(NUMBER_OF_WEIGHTS):
    LIST_OF_COLUMNS_X_MAC.append(f"weight_{i+1}")
LIST_OF_COLUMNS_X_MAC.append("bias_1")
# ----------------------------- END MAC Model Training Hyperparameters ------------------------------ #