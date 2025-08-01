import numpy as np
import pandas as pd
import os
from collections import defaultdict

class NeuronLayer:
    def __init__(self, layer_name, num_neurons, num_time_steps, period, neuron_params, weights, next_layer_num_neurons, weight_scaling_factor,
                 std_scaler, neuron_state_model, e_static_model, spike_or_not_model, 
                 e_model, l_model, save_path, load_in_mlp_models=False):
        # Initialize the layer with necessary parameters
        self.name = layer_name
        self.num_neurons = num_neurons
        self.num_time_steps = num_time_steps
        self.period = period
        self.neuron_params = np.tile(neuron_params, (num_neurons, 1))

        self.weights = weights
        self.next_layer_num_neurons = next_layer_num_neurons
        self.weight_scaling_factor = weight_scaling_factor
        self.save_path = save_path

        # MLP Model Add Ons
        self.load_in_mlp_models = load_in_mlp_models
        self.std_scaler = std_scaler

        # LASANA ML Models for inference
        self.neuron_state_model = neuron_state_model
        self.e_static_model = e_static_model
        self.spike_or_not_model = spike_or_not_model
        self.e_model = e_model
        self.l_model = l_model


        # State variables
        self.global_neuron_state = np.zeros(self.num_neurons)
        self.time_since_last_update = np.ones(self.num_neurons) * -1
        
        # Event tracking
        self.leak_events_neuron_state_start_per_time_step = defaultdict(list)
        self.leak_events_neuron_state_per_time_step = defaultdict(list)
        self.leak_events_energy_per_time_step = defaultdict(list)

        self.spike_events_neuron_state_start_per_time_step = defaultdict(list)
        self.spike_events_weights_per_time_step = defaultdict(list)
        self.spike_events_neuron_state_per_time_step = defaultdict(list)
        self.spike_events_energy_per_time_step = defaultdict(list)
        self.spike_events_latency_per_time_step = defaultdict(list)
        self.spike_events_spike_or_not_per_time_step = defaultdict(list)

        # Initialize time step counter
        self.time_step_id = 0

    def step(self, spike_events):
        """Process a single timestep with external spike events.
        
        Note that in this case, spike_events are just an array of the weights where each index in the array corresponds to
        the weight for that neuron on this timestep
        """

        if self.time_step_id >= self.num_time_steps:
            raise ValueError("Simulation has already completed all timesteps.")

        # Extract spike neuron ids from the events
        spike_neuron_ids = np.nonzero(spike_events)[0]

        # Handle neurons that are not spiking at the last timestep
        if self.time_step_id == self.num_time_steps - 1:
            self.handle_non_spiking_neurons(self.time_step_id, spike_neuron_ids)

        if spike_neuron_ids.size == 0:
            if self.next_layer_num_neurons == 0:
                # Last layer
                next_layer_input = np.zeros(self.num_neurons)
            else:
                next_layer_input = np.zeros(self.next_layer_num_neurons)
        else:
            # If there are spikes :) 

            # Process leak events
            self.handle_leak_events(self.time_step_id, spike_neuron_ids)
            
            # Process spike events
            spike_array = self.handle_spike_events(self.time_step_id, spike_neuron_ids, spike_events)

            # Apply weights to spiking neurons and generate input for the next layer
            next_layer_input = self.apply_weights_and_generate_next_layer_input(spike_array)

        # Increment the timestep counter
        self.time_step_id += 1

        return next_layer_input

    def handle_non_spiking_neurons(self, time_step_id, spike_neuron_ids):
        """At last timestep, end all events as a leak event if there were no more spikes on the last timestep"""
        ids_not_spiking = np.setdiff1d(np.arange(self.num_neurons), spike_neuron_ids)

        if ids_not_spiking.shape[0] == 0:
            # If there are no ids that do not have spikes, just end :)
            return 

        times_to_include_in_timing_event = ((time_step_id - self.time_since_last_update[ids_not_spiking]) * self.period).reshape(-1, 1)
        neuron_states = self.global_neuron_state[ids_not_spiking].reshape(-1, 1)
        leak_params = self.neuron_params[ids_not_spiking]
        all_params_for_ml = np.concatenate((neuron_states, np.zeros((len(ids_not_spiking), 1)), times_to_include_in_timing_event, leak_params), axis=1)

        if self.load_in_mlp_models:
            all_params_for_ml = self.std_scaler.transform(all_params_for_ml)

        next_neuron_state = self.neuron_state_model.predict(all_params_for_ml)
        energy = self.e_static_model.predict(all_params_for_ml)

        self.global_neuron_state[ids_not_spiking] = next_neuron_state
        self.leak_events_neuron_state_start_per_time_step[time_step_id].append((ids_not_spiking, neuron_states))
        self.leak_events_neuron_state_per_time_step[time_step_id].append((ids_not_spiking, next_neuron_state))
        self.leak_events_energy_per_time_step[time_step_id].append((ids_not_spiking, energy))

    def handle_leak_events(self, time_step_id, spike_neuron_ids):
        """Handle neurons undergoing leak events"""
        time_since_last = self.time_since_last_update[spike_neuron_ids]
        mask = time_since_last < (time_step_id - 1)
        neurons_with_leak = spike_neuron_ids[mask]

        if neurons_with_leak.size > 0:
            times = ((time_step_id - time_since_last[mask] - 1) * self.period).reshape(-1, 1)
            leak_params = self.neuron_params[neurons_with_leak]
            neuron_states = self.global_neuron_state[neurons_with_leak].reshape(-1, 1)
            all_params_for_ml = np.concatenate((neuron_states, np.zeros((len(neurons_with_leak), 1)), times, leak_params), axis=1)

            if self.load_in_mlp_models:
                all_params_for_ml = self.std_scaler.transform(all_params_for_ml)

            next_neuron_state = self.neuron_state_model.predict(all_params_for_ml)
            energy = self.e_static_model.predict(all_params_for_ml)

            self.global_neuron_state[neurons_with_leak] = next_neuron_state
            self.leak_events_neuron_state_per_time_step[time_step_id].append((neurons_with_leak, next_neuron_state))
            self.leak_events_neuron_state_start_per_time_step[time_step_id].append((neurons_with_leak, neuron_states))
            self.leak_events_energy_per_time_step[time_step_id].append((neurons_with_leak, energy))

    def handle_spike_events(self, time_step_id, spike_neuron_ids, spike_events):
        """Handle spike events for the neurons"""
        if not spike_neuron_ids.size:
            return
        
        params = self.neuron_params[spike_neuron_ids]
        times = np.ones((spike_neuron_ids.shape[0], 1)) * self.period
        weights = spike_events[spike_neuron_ids].reshape(-1,1)
        neuron_states = self.global_neuron_state[spike_neuron_ids].reshape(-1, 1)

        all_together = np.concatenate((neuron_states, weights, times, params), axis=1)

        if self.load_in_mlp_models:
            all_together = self.std_scaler.transform(all_together)

        # Spike or Not?
        spike_or_not = self.spike_or_not_model.predict(all_together)
        
        # Calculate next state and energy
        next_neuron_state = self.neuron_state_model.predict(all_together)
        static_energy = self.e_static_model.predict(all_together) * np.logical_not(spike_or_not)
        energy = self.e_model.predict(all_together) * spike_or_not
        energy = energy + static_energy

        # Latency
        latency = self.l_model.predict(all_together) * spike_or_not

        # Update states and track events
        self.global_neuron_state[spike_neuron_ids] = next_neuron_state
        self.time_since_last_update[spike_neuron_ids] = time_step_id

        self.spike_events_weights_per_time_step[time_step_id].append((spike_neuron_ids, weights))
        self.spike_events_neuron_state_start_per_time_step[time_step_id].append((spike_neuron_ids, neuron_states))
        self.spike_events_neuron_state_per_time_step[time_step_id].append((spike_neuron_ids, next_neuron_state))
        self.spike_events_energy_per_time_step[time_step_id].append((spike_neuron_ids, energy))
        self.spike_events_latency_per_time_step[time_step_id].append((spike_neuron_ids, latency))
        self.spike_events_spike_or_not_per_time_step[time_step_id].append((spike_neuron_ids, spike_or_not))

        # Return an array of all neurons and whether they are spiking this term :)
        spike_array = np.zeros(self.num_neurons)
        spike_array[spike_neuron_ids] = spike_or_not
        return spike_array

    def apply_weights_and_generate_next_layer_input(self, spike_array):
        """Generate the input for the next layer based on spike events and weights."""
        next_layer_input = np.zeros(self.next_layer_num_neurons)

        if self.next_layer_num_neurons == 0:
            return spike_array
        else:
            # Loop through each index in the spike array
            for i in range(spike_array.shape[0]):
                if spike_array[i] != 0:                 
                    next_layer_input += (self.weights[:, i] * self.weight_scaling_factor)
            return next_layer_input
        
    def save_layer_information(self, image_num):
        #print("Saving Layer Information After Run!")

        # Post Processing to make sense of everything
        energy_per_neuron_event = defaultdict(list)
        latency_per_neuron_event = defaultdict(list)
        start_neuron_state_per_neuron_event = defaultdict(list)
        neuron_state_per_neuron_event = defaultdict(list)
        spike_or_not_per_neuron_event = defaultdict(list)
        event_timeline = defaultdict(list)
        weights = defaultdict(list)

        per_timestep_guy = defaultdict(list)

        for time_step_id in range(self.num_time_steps):
            # Handle Leak Event First
            leak_time = self.leak_events_neuron_state_per_time_step[time_step_id]

            if len(leak_time) > 0:
                # Combine guys 
                leak_ids, neuron_state_s = leak_time[0]
                leak_ids, neuron_state_prev = self.leak_events_neuron_state_start_per_time_step[time_step_id][0]
                leak_ids, energy = self.leak_events_energy_per_time_step[time_step_id][0]

                for i in range(leak_ids.shape[0]):
                    leak_id = leak_ids[i]

                    energy_per_neuron_event[leak_id].append(energy[i] * 10**-12)
                    latency_per_neuron_event[leak_id].append(0)
                    start_neuron_state_per_neuron_event[leak_id].append(neuron_state_prev[i][0])
                    neuron_state_per_neuron_event[leak_id].append(neuron_state_s[i])
                    spike_or_not_per_neuron_event[leak_id].append(0)
                    event_timeline[leak_id].append("leak")
                    weights[leak_id].append(0)

                    # Put the leak timestep guy in here
                    # df[["Run_Number","Event_Type", 'Output_Spike', 'Energy', "Latency", "Cap_Voltage_At_Output_End"]]
                    timestep_leak_info = (leak_id, time_step_id,"leak", 0, 0, energy[i]* 10**-12, 0, neuron_state_prev[i][0], neuron_state_s[i])
                    per_timestep_guy[time_step_id].append(timestep_leak_info)

            # Handle Spike Events
            spike_time = self.spike_events_neuron_state_per_time_step[time_step_id]

            if len(spike_time) > 0:
                spike_ids, neuron_state_s = spike_time[0]
                spike_ids, neuron_state_prev = self.spike_events_neuron_state_start_per_time_step[time_step_id][0]
                spike_ids, energy = self.spike_events_energy_per_time_step[time_step_id][0]
                spike_ids, latency = self.spike_events_latency_per_time_step[time_step_id][0]
                spike_ids, spike_or_not = self.spike_events_spike_or_not_per_time_step[time_step_id][0]
                spike_ids, cool_weights = self.spike_events_weights_per_time_step[time_step_id][0]
                spike_ids = np.array(spike_ids)

                for i in range (spike_ids.shape[0]):
                    spike_id = spike_ids[i]
                    energy_per_neuron_event[spike_id].append(energy[i]* 10**-12)
                    latency_per_neuron_event[spike_id].append(latency[i]* 10**-9)
                    neuron_state_per_neuron_event[spike_id].append(neuron_state_s[i])
                    start_neuron_state_per_neuron_event[spike_id].append(neuron_state_prev[i][0])
                    spike_or_not_per_neuron_event[spike_id].append(spike_or_not[i])

                    if spike_or_not[i]:
                        event = 'in-out'
                    else:
                        event = 'in-no_out'
                    event_timeline[spike_id].append(event)
                    weights[spike_id].append(cool_weights[i][0])

                    # Put spike timestep guy in here
                    timestep_spike_info = (spike_id, time_step_id, event, cool_weights[i][0], spike_or_not[i], energy[i]* 10**-12, latency[i]* 10**-9, neuron_state_prev[i][0], neuron_state_s[i])
                    per_timestep_guy[time_step_id].append(timestep_spike_info)

        # For each timestep guy, convert into a dataframe
        pd_columns = ["Neuron_Num","Digital_Time_Step","Event_Type", 'Weight', 'Output_Spike', 'Energy', "Latency", "Cap_Voltage_At_Input_Start","Cap_Voltage_At_Output_End"]
        
        all_guys = []
        # Convert each list of tuples into a DataFrame
        for key, value in per_timestep_guy.items():
            df = pd.DataFrame(value, columns=pd_columns)
            all_guys.append(df)

        mega_df = pd.concat(all_guys)
        df_sorted = mega_df.sort_values(by=['Neuron_Num', 'Digital_Time_Step'])

        save_location = f"{image_num}_spike_info_{self.name}.csv"
        df_sorted.to_csv(os.path.join(self.save_path, save_location), index=False)