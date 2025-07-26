import numpy as np
from scipy.sparse.linalg import spsolve
import bisect
from stat_helpers import normalize

class SpikeData:
    def __init__(self, run_name, run_number, index, event_type, random_knobs, peak, index_range):
        # House Keeping :)
        self.run_name = run_name
        self.run_number = run_number
        self.index = index
        self.event_type = event_type      # input, output
        self.randomized_knob_parameters = random_knobs      # Note: This is a list of tuples ("Name of Parameter", int(val))

        # Spike Event Details :)
        self.detected_peak_index = peak
        self.index_range = index_range

    def __str__(self):
        return (
            f"Test Name: {self.run_name}, "
            f"Run Number: {self.run_number}, "
            f"Index: {self.index}, "
            f"Event Type: {self.event_type}, "
            f"True Range: {self.index_range}, "
            f"Peak Index: {self.detected_peak_index}, "
            f"Randomized Knob Parameters: {self.randomized_knob_parameters}"
        )
        
class SpikeEvent:
    # NOTE: first_data is always the first thing to put and the second thing can be None
    def __init__(self, event_type, first_data, second_data, time_vec, digital_freq):
        # House Keeping
        self.run_name = first_data.run_name
        self.run_number = first_data.run_number
        self.index = first_data.index
        self.randomized_knob_parameters = first_data.randomized_knob_parameters  

        # Event Type and house keeping
        self.event_type = event_type

        # Container to hold both of the two datum
        self.input_spike = first_data
        self.output_spike = second_data 

        # State Information, Energy and Latency Information, and Transient Simulation Information
        self.input_peak_amplitude = None
        self.input_total_charge = None
        #self.input_total_time = None
        self.cap_voltage_at_input_start = None

        # We are trying to predict these :)
        self.cap_voltage_at_output_end = None
        self.energy = None
        self.latency = None 

        # House keeping for events
        self.event_start_index = None
        self.event_end_index = None

        # Enforce digital timestep guidelines here.
        if event_type == 'in-out' or event_type == 'in-no_out':
            self.start_digital_timestep = np.floor(time_vec[first_data.index_range[0]] * digital_freq)
            self.end_digital_timestep = self.start_digital_timestep + 1
        else:
            # Both leak events are the same, capture period in between
            self.start_digital_timestep = np.ceil(time_vec[first_data.index_range[0]] * digital_freq)
            self.end_digital_timestep = np.floor(time_vec[second_data.index_range[0]] * digital_freq)
        
        # Period:
        self.period = 1 / digital_freq
        self.input_total_time = (self.end_digital_timestep - self.start_digital_timestep) * self.period

        # Convert back and find the closest index
        actual_digital_start_time = self.start_digital_timestep / digital_freq 
        actual_digital_end_time = self.end_digital_timestep / digital_freq 

        self.event_start_index = np.abs(time_vec - actual_digital_start_time).argmin()
        self.event_end_index = np.abs(time_vec - actual_digital_end_time).argmin()

        # (Note this is the timestep in which the guy is actually evaluated :O)
        # Calculate Digital Time Step
        if event_type == 'in-out' or event_type == 'in-no_out':
            # Input Spike, choose event_start
            event_index = self.start_digital_timestep
        else:
            # Leak Event, choose event_end
            event_index = self.end_digital_timestep

        self.digital_time_step = event_index

    def last_guy(self, index, time_step):
        #print(index, time_step)
        self.event_end_index = index
        self.digital_time_step = time_step

        self.input_total_time = (time_step - self.start_digital_timestep) * self.period


    def spike_to_dict_for_df(self, random_knob_list):
        spike_dict = {}

        # House Keeping
        spike_dict["Run_Name"] = self.run_name
        spike_dict["Run_Number"] = self.run_number
        spike_dict["Input_Spike_Index"] = self.index
        spike_dict["Event_Type"] = self.event_type
        spike_dict['Event_Start_Index'] = self.event_start_index
        spike_dict['Event_End_Index'] = self.event_end_index
        spike_dict['Digital_Time_Step'] = self.digital_time_step

        # Actual Data
        spike_dict["Input_Peak_Amplitude"] = self.input_peak_amplitude
        spike_dict["Input_Total_Charge"] = self.input_total_charge
        spike_dict["Input_Total_Time"] = self.input_total_time
        spike_dict["Cap_Voltage_At_Input_Start"] = self.cap_voltage_at_input_start
        spike_dict["Cap_Voltage_At_Output_End"] = self.cap_voltage_at_output_end
        spike_dict["Energy"] = self.energy
        spike_dict["Latency"] = self.latency

        for knob in random_knob_list:
            knob_value = self.randomized_knob_parameters[knob]
            spike_dict[knob] = knob_value

        return spike_dict
    
    def __str__(self):
        return (f"SpikeEvent(\n"
                f"  Run_Name: {self.run_name},\n"
                f"  Run_Number: {self.run_number},\n"
                f"  Spike_Index: {self.index},\n"
                f"  Event_Type: {self.event_type},\n"
                f"  Input_Peak_Amplitude: {self.input_peak_amplitude},\n"
                f"  Cap_Voltage_At_Input_Peak: {self.cap_voltage_at_input_peak},\n"
                f"  Cap_Voltage_At_Output_End: {self.cap_voltage_at_output_end},\n"
                f"  Energy: {self.energy},\n"
                f"  Latency: {self.latency}\n"
                f")")


def identify_spikes(step_function_vector, adjust_start_end_margins_percent):
    spikes = []
    in_spike = False
    spike_start = 0
    spike_counter = 0
    vector_length = step_function_vector.shape[0]

    for i, value in enumerate(step_function_vector):
        if value == 1 and not in_spike:
            # Start of spike here
            in_spike = True
            spike_start = i
        elif value == 0 and in_spike:
            # End spike
            in_spike = False
            middle_of_spike = spike_start + int((i - spike_start)/ 2)

            # Note this is relative, and assumes that at a spike we have a relatively similar amount of points since a lot is happening
            spike_length = i - spike_start
            spike_margin_adjustment = int(spike_length * adjust_start_end_margins_percent)
            adjusted_start = max(spike_start-spike_margin_adjustment, 0)
            adjusted_end = min(i+spike_margin_adjustment, vector_length-1)

            spike_data = (spike_counter, middle_of_spike, adjusted_start, adjusted_end)
            spikes.append(spike_data)
            spike_counter+=1
    
    return spikes 

def baseline_als(y, lam, p, niter=10):
    # Lambda between 10^2 to 10^9, 0.001 <= p <= 0.1 for signal with positive peaks
    # From https://stackoverflow.com/questions/29156532/python-baseline-correction-library
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def find_closest_less_than(sorted_list, x):
    # NOTE: THIS MUTATES THE LIST, use where x not in list
    index = bisect.bisect_right(sorted_list, (x,))  # Use (x,) to create a tuple for comparison
    if index:
        closest_tuple = sorted_list[index - 1]
        del sorted_list[index - 1]  # Remove the closest tuple from the list
        return closest_tuple, sorted_list
    return None, sorted_list

def find_earlier_input_spike(sorted_list, x):
    # NOTE: THIS DOES NOT MUTATE THE LIST, use where x is in list
    index = bisect.bisect_left(sorted_list, (x,))  # Use (x,) to create a tuple for comparison
    if index:
        closest_tuple = sorted_list[index - 1]
        return closest_tuple
    return None

def find_closest_greater_than(sorted_list, x):
    # For use with same list, where index, x, is in list

    # NOTE: THIS DOES NOT MUTATE THE LIST
    index = bisect.bisect_right(sorted_list, (x,))
    if index+1 < len(sorted_list):
        return sorted_list[index+1]
    return None

def find_next_input_spike(sorted_list, x):
    # For use with different list (where x not in list)
    # NOTE: THIS DOES NOT MUTATE THE LIST
    index = bisect.bisect_right(sorted_list, (x,))
    if index < len(sorted_list):
        return sorted_list[index]
    return None

def prune_to_zero(vector, threshold):
    return np.where(np.abs(vector) < threshold, 0, vector)

def find_start_of_spike(arr, start_search_index, end_search_index):
    for i in range(start_search_index, end_search_index+1, -1):
        if arr[i] > 0 and arr[i - 1] <= 0:
            return i-1
    return None

def find_end_of_spike(arr, start_search_index, end_search_index, N=5):
    # Gradient must be negative for a couple of samples before count :)
    count_negative = 0

    for i in range(start_search_index, end_search_index-1):
        if arr[i] < 0:
            count_negative+=1
        else:
            count_negative=0

        if count_negative > N and arr[i] < 0 and arr[i+1] >= 0:
            return i+1
    return None

# Note that this cannot handle if the input spike is larger than the digital timestep... 
def identify_nice_input_spike(mask, timestep_start_index, timestep_end_index):
    assert(timestep_end_index >= timestep_start_index)      # Make sure that this is not backwards like last time...

    in_spike = False
    spike_start = None

    current_index = timestep_start_index
    max_num_steps = timestep_end_index - timestep_start_index


    for i in range(max_num_steps):
        mask_index = mask[current_index]

        if mask_index == 1 and not in_spike:
            # Start of spike here
            in_spike = True
            spike_start = current_index
        elif mask_index == 0 and in_spike:
            # End spike
            in_spike = False

            spike_data = (True, spike_start, current_index)
            return spike_data 
        
        current_index+=1
    
    return (False, None, None)


def get_gradient_where_spike(data, gradient, time_vector):
    # Normalize gradient
    gradient_norm = normalize(gradient)
    gradient_norm = gradient_norm - gradient_norm[0]   # Zero it again

    # Normalize data
    data_norm = normalize(data)
    data_norm = data_norm - data_norm[0]    # Zero out baseline 


    # Get zero crossings
    zero_crossings = np.diff(np.sign(gradient_norm))
    zero_crossings = np.pad(zero_crossings, (0, 1), 'constant')
    
    # Positive to negative crossing indicates end of a spike, negative to positive indicates start
    spike_starts = (zero_crossings > 0)
    spike_ends = (zero_crossings < 0)

    # Initialize spike mask
    spike_mask = np.zeros_like(gradient, dtype=bool)
    
    # Mark regions between significant spike starts and ends as spikes
    in_spike = False
    for i in range(len(spike_mask) - 1):
        if spike_starts[i]:
            in_spike = True
        if spike_ends[i]:
            in_spike = False
        spike_mask[i + 1] = in_spike

    # Now, make sure the amplitude at these points is greater than some threshold
    true_mask = spike_mask * (data_norm) > 0.01     # Greater than 1 percent of the nominal gradient value :)
    
    return true_mask
    
def identify_if_output_spike(mask, mask_start_spike, timestep_start_index, timestep_end_index, timestep, verbose=False):
    # New mask which combines the two
    subset_of_mask = mask[timestep_start_index: timestep_end_index+1]    # Note the need to include +1 because otherwise the last timestep is not added in :)
    spike_or_not = np.sum(subset_of_mask) > 0

    # If so, find start of when mask goes from 0 to 1
    max_timesteps = timestep_end_index - timestep_start_index
    prev = 1
    good_index = None

    for i in range(max_timesteps):
        if prev == 0 and mask[timestep_start_index+i] == 1:
            good_index=timestep_start_index+i-1
            break
        else:
            prev = mask[timestep_start_index+i] == 1

    if spike_or_not == True and good_index == None:
        if verbose:
            print(f"Spike without Input Spike (Perhaps delayed?) Timestep {timestep}")
        #assert(good_index != None)

    spike_or_not = good_index != None
    
    return spike_or_not,good_index

def create_event_list_no_spike(timesteps_with_no_change, all_timesteps):
    events = []
    start = all_timesteps[0]  # Initialize the first event starting point

    for i in range(1, len(all_timesteps)):
        # If the current timestep is in the "no change" list, and the previous timestep was also in "no change", we continue
        if all_timesteps[i] in timesteps_with_no_change and all_timesteps[i - 1] in timesteps_with_no_change:
            continue
        
        # Otherwise, we have reached the end of a "leak" or "in-out" event
        end = all_timesteps[i]

        # Determine if the event is leak or in-out
        if start in timesteps_with_no_change:
            label = "leak"  # No change in inputs, event is leak
        else:
            label = "in-out"  # Change in inputs, event is in-out
            
        events.append((start, end, label))
        # Reset the start for the next event
        start = end

    # Handle the last event extending to the end of the simulation
    if start != all_timesteps[-1]:
        label = "leak" if start in timesteps_with_no_change else "in-out"
        events.append((start, "End", label))  # Label as 'End' for clarity

    # Check if the last timestep is part of the no-change list and handle accordingly
    if all_timesteps[-1] in timesteps_with_no_change:
        # If the last timestep is in the no-change list, we want to extend the last event to "End"
        if events and events[-1][0] == all_timesteps[-2]:  # Check if the last event is continuous
            last_event_start = events[-1][1]
            last_event_end = "End"
            last_event_label = "leak"
            events.append((last_event_start, last_event_end, last_event_label))
    else:
        # If the last timestep is in-out and not in no-change list
        if start == all_timesteps[-1]:  # No new event was recorded for this last timestep
            events.append((all_timesteps[-1], "End", "in-out"))

    # Bandaid fix. Get rid of 'End', 'End'
    if events[-1][0] == 'End' and events[-1][1] == 'End':
        events.pop()


    return events