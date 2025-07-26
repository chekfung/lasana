# ----------------------------- BEGIN Generate Dataset Hyperparameters ------------------------------ #
# Spiking Circuit Run Hyperparameters
RUN_NAME = 'spiking_neuron_run'                             # Name of the run that will be created in data/
NUMBER_OF_RUNS = 2000                                       # Number of testbenches / randomized SPICE runs
NUM_PROCESSES = 20                                          # Maximum number of processes that can be spun for SPICE simulations (NOTE: Each process uses a CAD license)
TOTAL_TIME_NS = 500                                         # Nanoseconds of runtime / SPICE simulation run
SIM_MIN_STEP_SIZE_NS = 0.01                                 # Minimum step size in nanoseconds of the simulation 
NUM_INPUT_POINTS = TOTAL_TIME_NS*10                         # Determines fidelity of input that is used for generation of .PWL files (as well as how big they are)
DIGITAL_INPUT_ENFORCEMENT = True                            # Enforce that inputs can only occur at digital timesteps (If False, deprecated)
DIGITAL_FREQUENCY = 0.2 * 10**9                             # Digital Backend Frequency 
SPIKING_INPUT = True                                        # Whether the input is a spiking signal or a more traditional steady state signal
INPUT_SEPARATE_VDD = True                                   # Input voltage / current sources are connected to their own VDD rail (makes it easier to decompose energies later on))
SIMULATOR = 'spectre'                                       # Simulator: Current Synopsys HSPICE and Cadence Spectre are supported. LTSpice is deprecated.

# Circuit-Specific Parameters
VDD = 1.5                                                   # VDD Rail
VSS = 0                                                     # VSS Rail
NUMBER_OF_INPUTS = 1                                        # Number of input nets
INPUT_NET_NAME = ['I_inj']                                  # Name of current input (s) (Requires I in front)
INPUT_NET = ['spikes']                                      # Name of interconnected net (s)
LOAD_CAPACITANCE = 500 * 10**(-15)                          # Farads    ;


MODEL_FILEPATH = '../data/spiking_neuron_spice_files/analog_lif_neuron.sp'         # Top SPICE file that JUST CONTAINS the subcircuit of the analog circuit that will be run.
OTHER_NECESSARY_FILES_IN_SPICE_RUN_DIRECTORY = []                                                 # Additional SPICE files that might be necessary to be linked with the model
LIBRARY_FILES = ['../data/spiking_neuron_spice_files/libraries/45nm_LP.pm']                 # Library files that might need to be linked such as technology files, etc.

# For example, Cout, spk -> Cout spk VSS xF     (Cout is the name, spk is the pos connection, VSS is the neg connection, and xF is the capacitance)
OUTPUT_CAPACITANCE_NAME = 'Cout'                                                        # Capacitance Name (Requires C in front) 
OUTPUT_LOAD_CAP_NET = 'spk'                                                             # Name of output load cap net
CIRCUIT_STATEFUL = True                                                                 # Enable if circuit is Stateful


SUBCIRCUIT_DEFINITION = f'X1 {INPUT_NET[0]} leak sf rtr adap {OUTPUT_LOAD_CAP_NET} vdd 0 lif_neuron'

# Additional circuit knobs that can be changed that we want to randomize and treat as parameters that can change in the circuit
# NOTE: For the rest of the randomly constrained values, we take a four variable tuple containing as follows ("string of net name", minimum, maximum, opt_flag)
# where the opt_flag specify binary knobs here (c for continuous, b for binary)
KNOB_PARAMS = [("V_sf", 0.5, 0.8, 'c'), ("V_adap", 0.5, 0.8, 'c'), ("V_leak", 0.4, 0.8, 'c'), ("V_rtr", 0.5, 0.8, 'c')]  # NOTE: Default values for all of these knobs are specified in the design. We are simply changing them here.


# ---
# Spiking Neuron Input Parameters
INPUT_CURRENT_SRC = True

NUMBER_OF_WEIGHTS = 1                                           # In the spiking input case, the weight is tied directly to a spike. 
WEIGHT_LOW = -2                                                 # If custom spike input, relative to input weight ; Otherwise, 
WEIGHT_HIGH = 2                                                 # If custom spiek input, relative to input weight; Otherwise, 

if SPIKING_INPUT:
    # Input Spike Footprint Generation
    '''
    When generating an input spike train, a natural question is to ask, what does a spike from this circuit even look like?

    Rather than try and emulate what a spike might look like, we take in a circuit file that is partially completed, run it, and
    grab the spike from there. This is then used for input generation.
    '''
    DELAY_RATIO = 0.3                                           # Forces input to have approximately 20% leak events
    SAME_SIGN_WEIGHTS_FRACTION = 0.1                            # Fraction of runs that will have all the same signed weights / inputs (either negative or positive with 50/50 chance)
    REFRACTORY_PERIOD = 5 * 10**(-10)                           # seconds   ; Time before another spike can occur on the same neuron 
    RUN_TIMEOUT = 10                                            # seconds   ; If spike mapping solution not found in x seconds, throw error.
    CIRCUIT_SIM_NUM_SPIKES = ("Num_Input_Spikes", 0, 300, "c")  # Number of simulated input spikes in one simulation (minimum, maximum)
    CIRCUIT_FAN_IN_RANGE = ("Circuit_Fan_In", 4, 32, "c")       # Neuron FAN IN range (minimum, maximum)
    CIRCUIT_FAN_OUT_RANGE = ("Circuit_Fan_Out", 1, 1, "c")      # Neuron FAN OUT range (minimum, maximum)


    PLOT_SPIKE_BOUNDS = False                                   # Plot spike footprint after running to see whether or not we correctly capture the spike
    OUTPUT_SPIKE_NAME = 'i(C)'            # This is for the footprint :)
    SPICE_FOOTPRINT_FILE = '../data/spiking_neuron_spice_files/analog_lif_neuron_footprint_run.sp'
    SPIKE_START = 306
    SPIKE_END = 361

    # Block Spike
    # Note: We denote block spike as a PWM of a set voltage where the height of the block is determined by the raw spike
    #       total charge. The total charge is matched, but implemented as a PWM spike. Changing the time step length will
    #       maintain the same charge as the original spike footprint above, meaning the height will be lower. We use this
    #       to better imitate hybrid spiking systems, such as BrainScales-2.
    CUSTOM_SPIKE = False
    CUSTOM_SPIKE_NUM_TIME_STEP_LENGTH = 10

# ----------------------------- END Generate Dataset Hyperparameters ------------------------------ #


# ----------------------------- BEGIN Analyze Dataset Hyperparameters ------------------------------ #
# Hyperparameters (Spiking Circuit)
DF_FILENAME = "spiking_neuron_dataset.csv"   # Output file to save the dataset that will be used for ML training

# Columns in output dataset
columns = ["Run_Name", "Run_Number", "Event_Type", 'Event_Start_Index', 'Event_End_Index', 'Digital_Time_Step',"Input_Peak_Amplitude", "Input_Total_Charge", "Weight","Input_Total_Time", "Cap_Voltage_At_Input_Start", "Cap_Voltage_At_Output_End", "Energy", "Latency"]
KNOB_NAMES = ["Circuit_Fan_In_I_inj", "Circuit_Fan_Out", "V_sf", "V_adap", "V_leak", "V_rtr"]
columns += KNOB_NAMES

PLOT_RUNS = False              # Plot each of the runs to try and debug what is going on
VERBOSE= False                 # Increase verbosity for debug sake

# DATASET PARAMETERS    
INPUT_WELL_DEFINED = True      # A parameter that makes edge detection a little bit easier if the input is well defined (not noisy)

# For attributing which input corresponds to what
INPUT_NAME = ['i(I_inj)']    
STATE_NET = 'v(spikes)'
OUTPUT_NAME = 'i(Cout)'

# Power Calculations
VDD_VOLTAGE = 'v(vdd)'
VDD_CURRENT = 'i(Vdd)'
VSS_VOLTAGE = None
VSS_CURRENT = None

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
LIST_OF_COLUMNS_X = ["Cap_Voltage_At_Input_Start", "Weight", "Input_Total_Time", "V_sf", "V_adap", "V_leak", "V_rtr"]

# ----------------------------- END Spike Model Training Hyperparameters ------------------------------ #