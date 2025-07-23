import numpy as np
import matplotlib.pyplot as plt
import time
import bisect
import random
from tools_helper import *

def find_number_less_than(lst, target):
    """
    Finds the largest number in a sorted list that is less than a specified target.

    Parameters:
    - lst (list of numbers): A sorted list of numbers in ascending order.
    - target (number): The target number to compare against.

    Returns:
    - number or None: The largest number in `lst` that is less than `target`. 
      Returns `None` if no such number exists.
    """
    index = bisect.bisect_left(lst, target)
    if index > 0:
        return lst[index - 1]
    else:
        return None  # No number in the list is less than the target

def find_number_greater_than(lst, target):
    """
    Finds the smallest number in a sorted list that is greater than a specified target.

    Parameters:
    - lst (list of numbers): A sorted list of numbers in ascending order.
    - target (number): The target number to compare against.

    Returns:
    - number or None: The smallest number in `lst` that is greater than `target`. 
      Returns `None` if no such number exists.
    """
    index = bisect.bisect_right(lst, target)
    if index < len(lst):
        return lst[index]
    else:
        return None  # No number in the list is greater than the target

def can_neuron_fire(new_firing_time, firing_times_list, refractory_period):
    """
    Determines if a neuron can fire at a specified time, considering a refractory period.

    Parameters:
    - new_firing_time (number): The proposed firing time for the neuron.
    - firing_times_list (list of numbers): A sorted list of past firing times for the neuron.
    - refractory_period (number): The minimum allowable time between consecutive firings.

    Returns:
    - bool: `True` if the neuron can fire at `new_firing_time` without violating the refractory period.
            `False` otherwise.

    Notes:
    - The function checks the closest firing times before and after `new_firing_time`.
    - The neuron can fire if no firing times are within the `refractory_period` of the proposed `new_firing_time`.
    """
    # Assumes that firing_time_list_exists
        #Determine if we can fire with this neuron and the refractory period
    firing_time_greater = find_number_greater_than(firing_times_list, new_firing_time)
    firing_time_lesser = find_number_less_than(firing_times_list, new_firing_time)

    firing_time_refractory_less = new_firing_time - refractory_period
    firing_time_refractory_greater = new_firing_time + refractory_period

    neuron_fire = False

    if firing_time_greater == None and firing_time_lesser == None:
        # We can do it
        neuron_fire = True
    elif firing_time_greater == None and firing_time_lesser != None:
        # check lesser side only
        if firing_time_lesser < firing_time_refractory_less:
            neuron_fire = True
    elif firing_time_greater != None and firing_time_lesser == None:
        # check higher side only
        if firing_time_greater > firing_time_refractory_greater:
            neuron_fire = True
    elif firing_time_greater != None and firing_time_lesser != None:
        # check both sides
        if (firing_time_lesser < firing_time_refractory_less) and (firing_time_greater > firing_time_refractory_greater):
            neuron_fire = True
    
    return neuron_fire


def generate_spike_map_digital(num_neurons, num_spikes, refractory_period, total_time, sampling_frequency, digital_frequency, timeout, input_delay_ratio):
    '''
    Very similar to generate_spike_map. Only difference is that now, we are bound by the digital_timestep,
    meaning that spikes can ONLY be generated at a digital timestep, since spikes are bound by the digital
    timestep. 
    '''
    num_samples = int(total_time * sampling_frequency)
    num_digital_samples = int(total_time * digital_frequency)
    digital_index_step = int(num_samples / num_digital_samples)

    if (digital_index_step < 1):
        print("Error occurred! Digital Timestep smaller than the Analog Timestep!")
        exit(1)

    # Determine which digital indices are not allowed to have spikes (purposeful delay periods)
    digital_timestep_delay_areas = []
    num_delay_digital_time_periods = int(num_digital_samples * input_delay_ratio)
    curr_num_delay_periods = 0

    while curr_num_delay_periods < num_delay_digital_time_periods:
        # Randomly choose time period
        timestep = random.randint(0, num_digital_samples-1) # Note randint is inclusive, so -1

        # Check if already taken
        if timestep in digital_timestep_delay_areas:
            continue
        
        # Not taken, put into digital tiemstep delay areas and continue
        digital_timestep_delay_areas.append(timestep)
        curr_num_delay_periods+=1

    # Create empty spike map
    spike_map = np.zeros(num_samples, dtype=np.int32)

    firing_times = {}

    start_time = time.time()

    curr_spikes_in_map = 0

    while curr_spikes_in_map < num_spikes:
        # Choose a random time
        firing_time = np.random.uniform() * total_time

        # Convert random time to digital time
        digital_index = int(firing_time * digital_frequency)
        spike_map_index = digital_index * digital_index_step
        digital_firing_time = spike_map_index / sampling_frequency

        # Dirty fix to make it so that spike map index does not occur on first 0 -> 5 entries
        SPIKE_INDEX_MIN = 5
        if spike_map_index < SPIKE_INDEX_MIN:
            continue

        # Check if delay timestep
        if digital_index in digital_timestep_delay_areas:
            continue

        neuron_can_fire = False

        for neuron_idx in range(num_neurons):
            if neuron_idx not in firing_times.keys():
                # New neuron
                firing_times[neuron_idx] = []
                neuron_can_fire = True
            else:
                neuron_can_fire = can_neuron_fire(digital_firing_time, firing_times[neuron_idx], refractory_period)

            if neuron_can_fire:
                spike_map[spike_map_index] += 1
                firing_times[neuron_idx].append(digital_firing_time)
                firing_times[neuron_idx].sort()
                curr_spikes_in_map += 1
                break

        # Check if we have exceeded the timeout period first
        elapsed_time = time.time() - start_time

        if elapsed_time > timeout:
            raise TimeoutError(f"Unable to generate {num_spikes} spikes with {num_neurons} neurons and refractory period of {refractory_period} given total time of {total_time} and sampling frequency of {sampling_frequency} within {timeout} seconds.")

        # Check for a Value Error in which it is not possible to actually implement this
    if curr_spikes_in_map != num_spikes:
        raise ValueError(f"Unable to generate {num_spikes} spikes with {num_neurons} neurons and refractory period of {refractory_period} given total time of {total_time} and sampling frequency of {sampling_frequency}.")

    return spike_map

    
def generate_spike_map(num_neurons: int, num_spikes: int, refractory_period: int, total_time: float, sampling_frequency: int, timeout: int, input_delay_ratio: float) -> np.ndarray:
    """
    Generates a numpy array representing the spikes of a set of neurons over a period of time.
    Note that this generates uniform spike map with uniform sampling frequency. That said, that is
    true when this is put in as input, but then, we will extract the output and that will be at the
    theoretical same time scale as everything else (so it will get captured in a different way)

    Args:
        num_neurons: The number of neurons.
        num_spikes: The total number of spikes that should occur.
        refractory_period: The refractory period of each neuron, in seconds.
        total_time: The total period of time over which spikes should be generated, in seconds.
        sampling_frequency: The number of samples taken per second.
        timeout: The maximum time in seconds allowed for the function to run.

    Returns:
        A numpy array of shape (sampling_frequency * total_time,) representing the number of spikes at each point in time.

    Raises:
        ValueError: If it is not possible to generate `num_spikes` spikes given the input parameters.
        TimeoutError: If the function exceeds the specified timeout period.
    """

    # Calculate the number of samples in the output vector
    num_samples = int(total_time * sampling_frequency)

    # Empty spike map and firing_time dictionary per neuron
    spike_map = np.zeros(num_samples, dtype=np.int32)
    firing_times = {}

    # Determine delay areas :)
    delay_times = []
    num_delay_times = input_delay_ratio * num_samples
    curr_num_delay_times = 0

    while curr_num_delay_times < num_delay_times:
        random_time = random.randint(0, num_samples-1)

        if random_time in delay_times:
            continue

        delay_times.append(random_time)
        curr_num_delay_times+=1


    start_time = time.time()
    # Loop until we have added the desired number of spikes to the spike map
    curr_spikes_in_map = 0
    while curr_spikes_in_map < num_spikes:
        # Choose a random time and convert into samples time
        firing_time = np.random.uniform() * total_time
        spike_time_idx = int(firing_time * sampling_frequency)
        real_firing_time = spike_time_idx / sampling_frequency

        if spike_time_idx in delay_times:
            continue

        # Check if any neuron can fire at this time
        neuron_can_fire = False
        for neuron_idx in range(num_neurons):
            if neuron_idx not in firing_times.keys():
                # New neuron
                firing_times[neuron_idx] = []
                neuron_can_fire = True
            else:
                # Otherwise, check if the neuron is in its refractory period
                neuron_can_fire = can_neuron_fire(real_firing_time, firing_times[neuron_idx], refractory_period)

            if neuron_can_fire:
                # If the neuron can fire, add it to the spike map and the firing times lis
                spike_map[spike_time_idx] += 1
                firing_times[neuron_idx].append(real_firing_time)
                firing_times[neuron_idx].sort()
                curr_spikes_in_map += 1
                break

        # Check if we have exceeded the timeout period first
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise TimeoutError(f"Unable to generate {num_spikes} spikes with {num_neurons} neurons and refractory period of {refractory_period} given total time of {total_time} and sampling frequency of {sampling_frequency} within {timeout} seconds.")

    # Check for a Value Error in which it is not possible to actually implement this
    if curr_spikes_in_map != num_spikes:
        raise ValueError(f"Unable to generate {num_spikes} spikes with {num_neurons} neurons and refractory period of {refractory_period} given total time of {total_time} and sampling frequency of {sampling_frequency}.")

    return spike_map

def generate_output_vector(spike_map: np.ndarray, spike_footprint: np.ndarray, weight_low: float, weight_high: float) -> np.ndarray:
    """
    Generate an output vector from a spike map and a spike footprint.

    Args:
        spike_map (np.ndarray): A one-dimensional array representing the spike map.
        spike_footprint (np.ndarray): A one-dimensional array representing the spike footprint.

    Returns:
        np.ndarray: A one-dimensional array representing the output vector.
    """

    # Initialize the output vector
    output_vector = np.zeros_like(spike_map, dtype=np.float64)

    # Iterate over the spike map
    for i in range(len(spike_map)):
        # Check if there is a spike at this time
        if spike_map[i] > 0:
            num_firings = spike_map[i]
            weights = np.random.uniform(weight_low * num_firings, weight_high*num_firings,size=1)
            start_idx = i
            end_idx = min(i + len(spike_footprint), len(output_vector))

            output_vector[start_idx:end_idx] += spike_footprint[:end_idx-start_idx] * weights[0]

    return output_vector

def create_pwl_file(filename, time_vector, output_vector):
    """
    Creates a PWL (Piecewise Linear) file with specified time and output values.

    Parameters:
    - filename (str): The name of the file to create.
    - time_vector (list of float): A list of time values for each PWL point.
    - output_vector (list of float): A list of output values for each PWL point, 
      corresponding to the times in `time_vector`.

    Notes:
    - The function writes each pair of (time, output) values as a line in the file, formatted as "time, output".
    """
    with open((filename), 'w') as f:
        for (t,v) in zip(time_vector, output_vector):
            # Key value pairs for pwl file
            f.write("{}, {}\n".format(t,v))

def interpolate_spike_footprint_to_sampling_frequency(spike_footprint, spike_footprint_time, sampling_period):
    """
    Interpolates a spike footprint to a specified sampling frequency.

    Parameters:
    - spike_footprint (array-like): Original spike amplitude values over time.
    - spike_footprint_time (array-like): Time points corresponding to `spike_footprint` values.
    - sampling_period (float): Desired sampling period for the interpolation.

    Returns:
    - numpy.ndarray: Interpolated spike footprint with values at the specified sampling frequency.

    Notes:
    - The function checks that `spike_footprint` and `spike_footprint_time` are of the same length.
    - Uses linear interpolation to create a new spike footprint with the desired sampling period.
    """
    # First error checking to make sure that spike footprint and spike_footprint_time are the same size
    if len(spike_footprint) != len(spike_footprint_time):
        print("Error Occurred in interpolate_spike_footprint_to_sampling_frequency. Spike footprint length not equal to spike_footprint_time")

    spike_footprint_total_time = spike_footprint_time[-1] - spike_footprint_time[0] 
    num_samples_in_spike = int(spike_footprint_total_time / sampling_period)
    interpolated_spike_time = np.linspace(0, spike_footprint_total_time, num_samples_in_spike)

    # Convert spike time to absolute time
    spike_footprint_time_subtracted = spike_footprint_time - spike_footprint_time[0]

    # Generate linear interpolated numpy array
    converted_spike_footprint = np.interp(interpolated_spike_time, spike_footprint_time_subtracted, spike_footprint)

    return converted_spike_footprint


# TODO: In the future, would it be possible to auto detect when the spike starts and ends? Could be difficult
#       Perhaps a better input to test this is just a constant current so that over some time, we know that it will spike...
def analyze_spike_file(spice_file, output_spike_name, spike_start, spike_end, sampling_period, PLOT_SPIKE_BOUNDS, simulator='spectre'):
    """
    Analyzes a spike file generated from a SPICE simulation, extracts a representative spike footprint, 
    and interpolates it to a specified sampling frequency.

    Parameters:
    - spice_file (str): The path to the SPICE file for analysis.
    - output_spike_name (str): The signal name to extract from the simulation file.
    - spike_start (int): The starting index of the spike within the output signal.
    - spike_end (int): The ending index of the spike within the output signal.
    - sampling_period (float): Desired sampling period for the interpolation.
    - PLOT_SPIKE_BOUNDS (bool): Whether to plot spike boundaries and interpolated spike footprint for visualization.
    - simulator (str, optional): The simulator to use (default is 'spectre').

    Returns:
    - numpy.ndarray: The interpolated spike footprint at the specified sampling frequency.

    Notes:
    - The function first runs the SPICE simulation file, then reads the file to extract the specified spike signal.
    - If `PLOT_SPIKE_BOUNDS` is `True`, the function plots the original and interpolated spike footprints with spike boundaries.
    """
    if spike_start > spike_end:
        print("Fatal Error Occurred. Spike start greater than spike end")
        exit(1)

    print("Running SPICE file to determine Representative Spike Footprint")
    print(spice_file)

    # Run SPICE File
    num_processes = 1
    run_simulation([spice_file], num_processes, simulator=simulator)

    # Read File and extract spike
    raw_obj = read_simulation_file(spice_file, simulator=simulator)

    # Get time and I(inj), and I(C)
    time_vec = get_signal("time", raw_obj, simulator=simulator)
    output_current_spike = get_signal(output_spike_name, raw_obj, simulator=simulator)

    if PLOT_SPIKE_BOUNDS:
        plt.figure(0)
        plt.plot(output_current_spike, '-x')
        plt.axvline(x=spike_start, color='red')
        plt.axvline(x=spike_end, color='red')
        plt.title("Identify Spike Bounds for representative spike :)")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude (A)")



    spike_footprint = output_current_spike[spike_start:spike_end]
    interpolated_spike_footprint = interpolate_spike_footprint_to_sampling_frequency(spike_footprint, time_vec[spike_start:spike_end], sampling_period)

    if PLOT_SPIKE_BOUNDS:
        plt.figure(1)
        plt.plot(interpolated_spike_footprint, '-x')
        # plt.axvline(x=spike_start, color='red')
        # plt.axvline(x=spike_end, color='red')
        plt.title("Interpolated Spike Footprint")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude (A)")

        plt.show()

        print("Exit Gracefully due to setting PLOT_SPIKE_BOUNDS")
        #exit()

    return interpolated_spike_footprint

def create_pwl_from_input_pulses(filename, time_of_each_input, transition_time, input_vector):
    """
    Generates a PWL file based on input vector with specified voltage levels and timing characteristics.

    This is for inputs that do not conform to the non-spiking case :)
    
    Parameters:
    - filename (str): Path to the output PWL file.
    - vdd (float): Voltage level for a high signal.
    - vss (float): Voltage level for a low signal.
    - time_of_each_input (float): Total duration for each input level in seconds.
    - transition_time (float): Time taken for the signal to rise or fall.
    - input_vector (list of int): List where `1` represents `vdd` and `-1` represents `vss`.
    
    The generated PWL waveform transitions between `vdd` and `vss` as specified by `input_vector`,
    with each level held for the remainder of `time_of_each_input` after a `transition_time`.
    """
    # Initialize the time counter
    current_time = 0.0

    with open(filename, 'w') as f:
        f.write(f"{current_time}, {0}\n")

        for i, target_voltage in enumerate(input_vector):

            # Transition from the previous voltage to the current target voltage
            current_time += transition_time
            f.write(f"{current_time}, {target_voltage}\n")

            # Hold the target voltage for the remainder of the input period
            current_time += (time_of_each_input - transition_time)
            f.write(f"{current_time}, {target_voltage}\n")

def map_weight_to_memristive_crossbar(weight, r_low, r_high):
    # Note this is for -1, 1, or 0 only as we only support the low and high stable states of the memristive crossbar
    # This function is adapted from the IMAC-sim work (output is pos weight connection, and negative weight connection)
    if weight == 0:
        return (r_low, r_low)
    elif weight == -1:
        return (r_high, r_low)
    elif weight == 1:
        return (r_low, r_high)
    
def separate_v_params(knob_params):
    """
    Separates parameters that start with "V" in their names from others.

    This function iterates through a list of parameter tuples, each containing
    a name, minimum value, maximum value, and type character. It checks if the
    name of each parameter starts with "V" and categorizes it accordingly.

    Parameters:
    -----------
    knob_params : list of tuples
        A list of tuples representing parameters, where each tuple contains:
        - name (str): Name of the parameter.
        - min_val (float): Minimum value of the parameter.
        - max_val (float): Maximum value of the parameter.
        - type_char (str): Type character associated with the parameter.

    Returns:
    --------
    tuple
        A tuple of two lists:
        - v_params (list of tuples): List of parameter tuples with names starting with "V".
        - other_params (list of tuples): List of parameter tuples with names that do not start with "V".

    Example:
    --------
    >>> KNOB_PARAMS = [("V_sf", 0.5, 1.5, 'c'), ("I_adap", 0.2, 0.8, 'd'), ("V_leak", 0.5, 1.5, 'c')]
    >>> v_params, other_params = separate_v_params(KNOB_PARAMS)
    >>> print("V Parameters:", v_params)
    V Parameters: [('V_sf', 0.5, 1.5, 'c'), ('V_leak', 0.5, 1.5, 'c')]
    >>> print("Other Parameters:", other_params)
    Other Parameters: [('I_adap', 0.2, 0.8, 'd')]

    """
    v_params = []
    other_params = []

    for name, min_val, max_val, type_char in knob_params:
        if name.startswith("V"):
            v_params.append((name, min_val, max_val, type_char))
        else:
            other_params.append((name, min_val, max_val, type_char))
    
    return v_params, other_params


def coin_flip(probability):
    return np.random.choice([1, 0], p=[probability, 1 - probability])