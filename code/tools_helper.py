from PyLTSpice import SimCommander, RawRead
from psf_utils import PSF
import numpy as np
import threading
import multiprocessing
import os 
import time

'''
This file is used as a wrapper around tools (LTSpice, Cadence Virtuoso).

This is a research project funded under the SLAM lab at UT Austin
Author: Jason Ho

NOTE: LTSpice is deprecated and no longer supported in this current version of the scripts.

'''

def write_line_with_newline(fp, line):
    fp.write(line+'\n')

def idiot_proof_sim_string(simulator_str):
    """
    Converts various simulator names to a standardized lowercase format and maps them to specific simulators (e.g., "cadence" to "spectre").

    Parameters:
        simulator_str (str): The name of the simulator.

    Returns:
        str: The normalized simulator name.

    Raises:
        TypeError: If `simulator_str` is not a string.
        ValueError: If `simulator_str` is not recognized as a supported simulator.
    """
    
    if not isinstance(simulator_str, str):
        raise TypeError(f"Input must be a string, but received {type(simulator_str).__name__}.")

    # Normalize by lowering case and stripping whitespace
    normalized_str = simulator_str.strip().lower()
    
    # Define the mapping for replacements
    simulator_mapping = {
        "cadence": "spectre",
        "synopsys": "hspice",
        "ltspice": "ltspice",
        "spectre": "spectre",
        "hspice": "hspice"
    }
    
    # Check if the normalized string is in the mapping
    if normalized_str not in simulator_mapping:
        raise ValueError(f"IDIOT_PROOF_SIM_STRING: Unrecognized simulator '{simulator_str}'. Expected one of: {', '.join(simulator_mapping.keys())}")
    
    # Return the mapped value
    return simulator_mapping[normalized_str]

def is_simulator_real(simulator):
    """
    Verifies if a simulator is recognized and currently supported in this version.

    Parameters:
        simulator (str): The name of the simulator.

    Returns:
        bool: True if the simulator is recognized and supported, False otherwise.

    Raises:
        ValueError: If the simulator is not supported.
    """
    
    # Sanity check to make sure simulator exists
    sim_str = idiot_proof_sim_string(simulator)

    if sim_str == 'spectre':
        return True
    elif sim_str == 'ltspice':
        return True
    elif sim_str == 'hspice':
        #raise ValueError("IS_SIMULATOR_REAL: Tentatively HSPICE is not supported in this version of LASGNA. (10/28/2024)")
        return True
    else:
        return False

def build_filepath_to_sim_file(fullpath_spice_file, simulator='spectre'): 
    """
    Constructs the file path to the simulation result file based on the simulator.

    Parameters:
        fullpath_spice_file (str): The full path to the SPICE file.
        simulator (str): The simulator name, defaults to 'spectre'.

    Returns:
        str: The file path to the simulation's raw data file.

    Raises:
        FileNotFoundError: If `fullpath_spice_file` does not exist.
        ValueError: If the simulator is not supported.
    """
    
    sim_str = idiot_proof_sim_string(simulator)
    assert(is_simulator_real(sim_str))

    # Check if SPICE file exists
    if not os.path.exists(fullpath_spice_file):
        raise FileNotFoundError(f"BUILD_FILEPATH_TO_SIM_FILE: The specified path '{fullpath_spice_file}' does not exist. Please check the path and try again.")
    
    # If Exists, split up into basename and directory structure
    sp_fname = os.path.basename(fullpath_spice_file)
    sp_dname = os.path.dirname(fullpath_spice_file)

    if sim_str == 'spectre':
        # Spectre Sim Filepath to Data Raw Data
        spectre_timesweep_name = 'timeSweep.tran.tran'
        spectre_run_folder = sp_fname.rsplit('.', 1)[0] + '.raw'
        raw_filepath = os.path.join(sp_dname, spectre_run_folder, spectre_timesweep_name)

    elif sim_str == 'ltspice':
        # LTSpice Sim .RAW file
        raw_filepath = os.path.join(sp_dname, sp_fname.rsplit('.', 1)[0] + '.raw')

    elif sim_str == 'hspice':
        # Note: We convert hspice tr0 files into PSF files using the Cadence PSF util
        raw_filepath = os.path.join(sp_dname, sp_fname.rsplit('.', 1)[0] + '.psf') 
    else: 
        raise ValueError("BUILD_FILEPATH_TO_SIM_FILE: Tentatively HSPICE is not supported in this version of LASGNA. (10/28/2024)")
    
    return raw_filepath

def read_simulation_file(sp_filepath, simulator='spectre', max_retries=3, initial_wait=5):
    """
    Reads the simulation output file with retry logic for memory errors.

    Parameters:
        sp_filepath (str): The file path to the SPICE file.
        simulator (str): The simulator name, defaults to 'spectre'.
        max_retries (int): Number of times to retry in case of failure.
        initial_wait (int): Initial wait time before retrying (exponential backoff).

    Returns:
        object: A simulation file object (PSF for Spectre or RawRead for LTSpice), or None if it fails.

    Raises:
        ValueError: If the simulator is not supported.
    """
    #from tools_helper import PSF, RawRead, build_filepath_to_sim_file, idiot_proof_sim_string, is_simulator_real

    sim_str = idiot_proof_sim_string(simulator)
    assert is_simulator_real(sim_str)

    raw_fp = build_filepath_to_sim_file(sp_filepath, simulator=simulator)
    print(f"Decoding: {raw_fp}")

    for attempt in range(max_retries):
        try:
            if sim_str == 'spectre':
                return PSF(raw_fp)

            elif sim_str == 'ltspice':
                return RawRead(raw_fp)

            elif sim_str == 'hspice':
                assert os.path.basename(raw_fp).rsplit('.', 1)[1] == 'psf'
                return PSF(raw_fp)

            else:
                raise ValueError(f"Unsupported simulator: {simulator}")

        except MemoryError:
            print(f"[ERROR] Out of memory while processing {sp_filepath}. Retrying {attempt + 1}/{max_retries}...")
            time.sleep(initial_wait * (2 ** attempt))  # Exponential backoff

        except Exception as e:
            print(f"[ERROR] Failed to process {sp_filepath}: {e}")
            break  # Stop retrying for non-memory errors

    print(f"[ERROR] Skipping {sp_filepath} after multiple failures.")
    return None  # Return None to indicate failure



def print_signal_names(sim_file_obj, simulator='spectre'):
    """
    Prints the names of the signals in the simulation output file.

    Parameters:
        sim_file_obj (object): The simulation file object.
        simulator (str): The simulator name, defaults to 'spectre'.

    Raises:
        ValueError: If the simulator is not supported.
    """
    
    # Make sure simulator string is not something crazy
    sim_str = idiot_proof_sim_string(simulator)
    assert(is_simulator_real(sim_str))
    all_signals_names = []

    print("======== Print Signal Names ========")
    if sim_str == 'spectre' or sim_str == 'hspice':
        # Spectre
        
        kinds = {
        'float double': 'real',
        'float complex': 'complex',
        }

        sweep = sim_file_obj.get_sweep()
        print(f'{sweep.name:<15}  {sweep.units:<12}  real')

        for signal in sim_file_obj.all_signals():
            kind = signal.type.kind
            kind = kinds.get(kind, kind)
            print(f'{signal.name:<15}  {signal.units:<12}  {kind}')
            all_signals_names.append(signal.name)
    
    elif sim_str == 'ltspice':
        # LTSpice Sim .RAW file
        print(sim_file_obj.get_trace_names())            # Get and print a list of all the traces
        print(sim_file_obj.get_raw_property())           # Print all the properties found in the Header section

    else:
        # HSPICE
        raise ValueError(f"PRINT_SIGNAL_NAMES: Tentatively SIM: {simulator} is not supported in this version of LASGNA. (10/28/2024)")
    
    print("======== End Print Signal Names ========\n")
    
    return all_signals_names


def get_signal(signal_name, sim_file_obj, simulator='spectre'):
    """
    Retrieves a specific signal's data from the simulation file.

    Parameters:
        signal_name (str): The name of the signal to retrieve.
        sim_file_obj (object): The simulation file object.
        simulator (str): The simulator name, defaults to 'spectre'.

    Returns:
        np.array: Array of the signal values.

    Raises:
        ValueError: If the simulator is not supported.
    """
    # With HSPICE, no voltage v() in front of names and refers to the net name. With spectre, they have the v connected to the net
    
    # Make sure simulator string is not something crazy
    sim_str = idiot_proof_sim_string(simulator)
    assert(is_simulator_real(sim_str))

    if signal_name == None:
        return None

    if sim_str == 'spectre' or sim_str == 'hspice':
        # Spectre (Using PSF Utils) and hspice after converting to ascii psf using util :)
        if signal_name == 'time':
            sweep = sim_file_obj.get_sweep()
            return np.array(sweep.abscissa)
        else:
            signal = sim_file_obj.get_signal(signal_name)
            return np.array(signal.ordinate)

    elif sim_str == 'ltspice':
        # LTSpice Sim .RAW file
        return np.array(sim_file_obj.get_trace(signal_name))

    else:
        raise ValueError(f"GET_SIGNAL: Tentatively sim: {simulator} is not supported in this version of LASGNA. (10/28/2024)")
    
def mask_negative_time(time, signal):
    """
    Masks negative time values for the LTSpice simulator.

    Parameters:
        time (np.array): The time vector.
        signal (np.array): The signal vector.

    Returns:
        np.array: Masked signal with only non-negative time values.
    """
    
    # Used for LTSpice because it is absolutely funny and weird
    time_vec_mask = time >= 0

    # Apply mask to all to keep only positive time values (Not sure why this happens lol)
    mask_time_vec = time[time_vec_mask]
    mask_signal = signal[time_vec_mask]

    return mask_signal

# --------------------------------------- #

def run_synopsys_hspice(args):
    """
    Runs a Synopsys HSPICE simulation and logs results. Note: requires hspice (AND cadence virtuoso for the PSF util ahhaa

    Parameters:
        args (tuple): Contains SPICE file path and simulation index.

    Returns:
        str: Simulation success or error message.
    """
    
    spice_filepath, sim_index = args
    process_id = os.getpid()
    thread_id = threading.get_ident()

    # Get filepaths and everything
    dirname = os.path.dirname(spice_filepath)
    basename = os.path.basename(spice_filepath)
    splited = basename.rsplit('.', 1)[0] 
    spectre_run_folder = splited + '.psf'
    output_raw_path = os.path.join(dirname, splited+'.tr0')
    psf_raw_path = os.path.join(dirname, spectre_run_folder)

    # Print the simulation number, process, and thread information
    print(f"Running FILE: {basename}, simulation #{sim_index} on process ID: {process_id}, thread ID: {thread_id}")

    # Run the SPICE simulation
    log_file = os.path.join(dirname, splited+'_sim_log.txt')
    cmd = f"hspice {spice_filepath} -o {dirname} > {log_file}"
    exit_code = os.system(cmd)
    if exit_code != 0:
        error_message = f"Error running simulation #{sim_index} for {spice_filepath}, exit code: {exit_code}"
        print(error_message)
        return error_message
    
    # Use PSF Util to convert to .psf ASCII such that we can use it :)
    cmd2 = f'psf -i {output_raw_path} -o {psf_raw_path}'
    exit_code = os.system(cmd2)

    if exit_code != 0:
        error_message = f"Error converting to PSF simulation #{sim_index} for {spice_filepath}, exit code: {exit_code}"
        print(error_message)
        return error_message

    return f"Simulation FILE: {basename} Index #:{sim_index} completed successfully"


# Run Simulation Files :)
def run_cadence_spectre(args):
    """
    Runs a Cadence Spectre simulation and logs results. Note: requires spectre (module load virtuoso)

    Parameters:
        args (tuple): Contains SPICE file path and simulation index.

    Returns:
        str: Simulation success or error message.
    """
    
    spice_filepath, sim_index = args
    process_id = os.getpid()
    thread_id = threading.get_ident()

    # Get filepaths and everything
    dirname = os.path.dirname(spice_filepath)
    basename = os.path.basename(spice_filepath)
    splited = basename.rsplit('.', 1)[0] 
    spectre_run_folder = splited + '.raw'
    total_raw_path = os.path.join(dirname, spectre_run_folder)

    # Print the simulation number, process, and thread information
    print(f"Running FILE: {basename}, simulation #{sim_index} on process ID: {process_id}, thread ID: {thread_id}")

    # Run the SPICE simulation
    log_file = os.path.join(dirname, splited+'_sim_log.txt')
    cmd = f"spectre +aps {spice_filepath} -format psfascii -r {total_raw_path} +multithread =l {log_file}"
    exit_code = os.system(cmd)
    if exit_code != 0:
        error_message = f"Error running simulation #{sim_index} for {spice_filepath}, exit code: {exit_code}"
        print(error_message)
        return error_message

    return f"Simulation FILE: {basename} Index #:{sim_index} completed successfully"

def run_ltspice(args):
    """
    Runs an LTSpice simulation using `SimCommander`.

    Parameters:
        args (tuple): Contains SPICE file path and simulation index.

    Returns:
        str: Simulation success or error message.
    """
    
    spice_filepath, sim_index = args
    process_id = os.getpid()
    thread_id = threading.get_ident()

    # Print the simulation number, process, and thread information
    print(f"Running simulation #{sim_index} on process ID: {process_id}, thread ID: {thread_id}")

    LTC = SimCommander(spice_filepath)
    LTC.run()
    LTC.wait_completion()

    if LTC.okSim == 1:
        basename = spice_filepath.rsplit('.', 1)[0]
        if os.path.exists(basename+"_1.sp"):
            os.remove(basename+"_1.sp")
        if os.path.exists(basename+"_1.log"):
            os.rename(basename+"_1.log", basename+".log")
        if os.path.exists(basename+"_1.raw"):
            os.rename(basename+"_1.raw", basename+".raw")
        if os.path.exists(basename+"_1.op.raw"):
            os.rename(basename+"_1.op.raw", basename+".op.raw")
        return f"Simulation #{sim_index} completed successfully"
    else:
        error_message = f"Error running simulation #{sim_index} for {spice_filepath}, exit code: {LTC.okSim}"
        print(error_message)
        return error_message
    
def run_simulation_one_file(one_spice_file, simulator='spectre'):
    '''
    Similar to the function below, but without multiprocessing spins so that it can be easier to multiprocess longer
    parallel processes beyond run_simulation
    '''
    sim_str = idiot_proof_sim_string(simulator)
    assert(is_simulator_real(sim_str))

    # Verify all SPICE file paths exist
    if not os.path.exists(one_spice_file):
        raise FileNotFoundError(f"RUN_SIMULATION ERR: The specified path '{one_spice_file}' does not exist. Please check the path and try again.")
    
    # Prepare arguments with file paths and simulation indices
    simulation_args = (one_spice_file, 0)

    # Select appropriate function for the simulator
    if sim_str == 'spectre':
        results = run_cadence_spectre(simulation_args)
    elif sim_str == 'ltspice':
        results = run_ltspice(simulation_args)
    elif sim_str == 'hspice':
        results = run_synopsys_hspice(simulation_args)
    else:
        raise ValueError(f"RUN_SIMULATION: Simulator Error {simulator}. (10/28/2024)")

    print(results)


def run_simulation(spice_files_list, num_processes=1, simulator='spectre'):
    """
    Runs SPICE simulations concurrently across multiple processes using the specified simulator.

    Parameters:
        spice_files_list (list): List of file paths to SPICE files for simulation.
        num_processes (int, optional): Number of processes to use for parallel simulations. Defaults to 1.
        simulator (str, optional): Simulator to use for running the simulations. 
            Supported values: 'spectre' (Cadence Spectre) or 'ltspice' (LTSpice). Defaults to 'spectre'.

    Raises:
        FileNotFoundError: If any file path in `spice_files_list` does not exist.
        ValueError: If the specified simulator is unsupported.

    Notes:
        - For each file in `spice_files_list`, the simulation index and file path are paired for simulation.
        - Each simulator has a specific function, selected based on `simulator`.
        - HSPICE is currently unsupported.

    Example:
        >>> run_simulation(['path/to/file1.sp', 'path/to/file2.sp'], num_processes=4, simulator='ltspice')

    """

    # First Thing
    ARTIFICIAL_PROCESS_LIMIT = 20
    
    sim_str = idiot_proof_sim_string(simulator)
    assert(is_simulator_real(sim_str))

    if num_processes > ARTIFICIAL_PROCESS_LIMIT and (sim_str=='spectre' or sim_str == 'hspice'):
        print(f"RUN_SIMULATION ERROR: Requested {num_processes} Cadence/HSPICE licenses but limited to {ARTIFICIAL_PROCESS_LIMIT} licenses")
        exit(-1)

    # Verify all SPICE file paths exist
    for fp in spice_files_list:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"RUN_SIMULATION ERR: The specified path '{fp}' does not exist. Please check the path and try again.")
    
    # Prepare arguments with file paths and simulation indices
    simulation_args = [(spice_filepath, idx) for idx, spice_filepath in enumerate(spice_files_list)]

    with multiprocessing.Pool(num_processes) as pool:
        # Select appropriate function for the simulator
        if sim_str == 'spectre':
            results = pool.map(run_cadence_spectre, simulation_args)
        elif sim_str == 'ltspice':
            results = pool.map(run_ltspice, simulation_args)
        elif sim_str == 'hspice':
            results = pool.map(run_synopsys_hspice, simulation_args)
        else:
            raise ValueError(f"RUN_SIMULATION: Simulator Error {simulator}. (10/28/2024)")

    # Display results
    for result in results:
        if result:
            print(result)

def write_input_spike_file(fp, name, first_net, second_net, pwl_filepath, vdd, simulator='spectre', write_voltage_src = True, current_src = True):
    """
    Writes input spike PWL FIle data in the appropriate simulator format to a given file.

    Parameters:
    - fp (file object): The file pointer to write the spike input.
    - name (str): The identifier name for the spike input.
    - first_net (str): The name of the first network node.
    - second_net (str): The name of the second network node.
    - pwl_filepath (str): Path to the file containing PWL (Piecewise Linear) data.
    - vdd (float): The supply voltage value.
    - simulator (str, optional): The simulator type, default is 'spectre'. Supported options: 'spectre', 'ltspice'.

    Raises:
    - ValueError: If the specified simulator is 'hspice' or an unrecognized type.
    """
    sim_str = idiot_proof_sim_string(simulator)
    assert(is_simulator_real(sim_str))

    # Write voltage for second vdd to connect the current connect
    vdd_connection = first_net
    if write_voltage_src:
        write_voltage(fp, f"vdd_generate_{name}", first_net, 0, vdd)
    else:
        first_net = 'vdd'

    # If current source, want first net to be Vdd so current can flow from vdd to second net
    # If voltage source, want first net to be output, and second net to be 0 (since compared against 0)
    if not current_src:
        first_net = second_net 
        second_net = 0

    if sim_str == 'spectre':
        write_line_with_newline(fp, f"{name} {first_net} {second_net} PWL PWLFILE='{pwl_filepath}'")

    elif sim_str == 'ltspice':
        write_line_with_newline(fp, f"{name} {first_net} {second_net} PWL FILE={pwl_filepath}")

    elif sim_str == 'hspice':
        write_line_with_newline(fp, f"{name} {first_net} {second_net} PWL PWLFILE='{pwl_filepath}'")
    else:
        raise ValueError(f"write_pwl_file: Simulator Error {simulator}. (10/28/2024)")

def write_sim_specific_simulation_information(fp, NUMBER_OF_INPUTS, probe_all_nets=True, voltage_nets=[],current_nets=[],simulator='spectre'):
    """
    Writes simulation-specific settings to a given file based on the simulator.

    Parameters:
    - fp (file object): The file pointer to write the simulation information.
    - simulator (str, optional): The simulator type, default is 'spectre'. Supported options: 'spectre', 'ltspice'.

    Raises:
    - ValueError: If the specified simulator is 'hspice' or an unrecognized type.
    
    Notes:
    - For 'spectre', this function writes language settings and probing commands for voltage and current.
    - For 'ltspice', no additional commands are currently specified.
    """
    sim_str = idiot_proof_sim_string(simulator)
    assert(is_simulator_real(sim_str))

    if sim_str == 'spectre':
        write_line_with_newline(fp, 'simulator lang=spice')
        write_line_with_newline(fp, '.options maxstep=1n')

        if probe_all_nets:
            write_line_with_newline(fp, '.probe v(*)')
            write_line_with_newline(fp, '.probe i(*)')
        else:
            # Write Specific Voltages to Probe
            voltage_line = '.probe '
            for i in voltage_nets:
                voltage_line+=i
            write_line_with_newline(fp, voltage_line)

            # Write Specific Currents to Probe
            current_line = '.probe '
            for i in current_nets:
                current_line+=i
            write_line_with_newline(fp, current_line)


    elif sim_str == 'ltspice':
        # Nothing to do here for the time being
        nothing = 'to_do'
    elif sim_str == 'hspice':
        write_line_with_newline(fp, '.OPTION INGOLD=2 ARTIST=2 PSF=2')
        #write_line_with_newline(fp, '.OPTION INTERP=1')
        write_line_with_newline(fp, '.OPTION DELMAX=.1NS')
        write_line_with_newline(fp, '.OPTION RELTOL=1e-6 ABSTOL=1e-12 VABSTOL=1e-12')
        #write_line_with_newline(fp, '.OPTION POST')
        write_line_with_newline(fp, '.OPTION PROBE')
        
        nodes = [f"v{i}" for i in range(NUMBER_OF_INPUTS)]
        probe_line = ".PROBE " + " ".join([f"v({node})" for node in nodes])
        write_line_with_newline(fp, probe_line)
        write_line_with_newline(fp, '.PROBE v(vdd) v(vss)')
        write_line_with_newline(fp, '.PROBE i(vdd) i(vss)')
        write_line_with_newline(fp, '.PROBE v(out1)')

    else:
        raise ValueError(f"write_sim_specific_simulation_information: Simulator Error {simulator}. (10/28/2024)")


def write_voltage(fp, voltage_name, net_name, gnd, voltage):
    """
    Writes a voltage source definition to a given file.

    Parameters:
    - fp (file object): The file pointer to write the voltage source.
    - voltage_name (str): The name of the voltage source.
    - net_name (str): The name of the network node connected to the voltage source.
    - gnd (str): The name of the ground node.
    - voltage (float): The voltage value to apply.
    """
    write_line_with_newline(fp, f"{voltage_name} {net_name} {gnd} {voltage}")

def write_capacitance(fp, cap_name, net_name, second_net, cap_femto):
    """
    Writes a capacitance definition to a given file.

    Parameters:
    - fp (file object): The file pointer to write the capacitance.
    - cap_name (str): The name of the capacitor.
    - net_name (str): The name of the first network node.
    - second_net (str): The name of the second network node.
    - cap_femto (float): The capacitance value in femtofarads (fF).
    """
    write_line_with_newline(fp, f"{cap_name} {net_name} {second_net} {cap_femto}fF IC=0V")

def change_SPICE_param_in_file(infile, outfile, objects_to_modify, zero_weight_nets):
    """
    Modifies a SPICE file by updating the final parameter value of specified object names.

    Parameters:
    - spice_filename (str): The path to the input SPICE file.
    - output_filename (str): The path to the output file where modified content will be saved.
    - objects_to_modify (dict): A dictionary where keys are SPICE object names (e.g., "V1", "R2") 
      and values are the new values to replace in the file.

    Notes:
    - Only the last value in a line matching the object name will be replaced (typically the SPICE parameter).
    - The modified content is saved in `output_filename`, leaving the original file unmodified.
    """
    for line in infile:
        # Split the line into parts to check if the first part is an object to modify
        parts = line.split()
        if parts and parts[0] in objects_to_modify:
            # Replace the last element with the new value for this object
            parts[-1] = str(objects_to_modify[parts[0]])
            

        if parts and parts[0] in zero_weight_nets:
            parts[-2] = str(0)
        
        line = ' '.join(parts) + '\n'  # Rejoin the parts into a line

        outfile.write(line)

def print_and_log(fd, string_to_write):
    print(string_to_write)
    fd.write(str(string_to_write)+"\n")
    