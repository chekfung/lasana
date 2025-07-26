import numpy as np
import time
from collections import defaultdict

class CrossBar:
    def __init__(self, number_of_circuits, circuit_params,
                 e_model, l_model, e_static_model, behavior_model,
                 CLOCK_PERIOD):
        # Params
        self.number_of_circuits = number_of_circuits
        self.possible_circuit_ids = np.arange(number_of_circuits)
        self.circuit_params = circuit_params
        self.CLOCK_PERIOD = CLOCK_PERIOD
        self.NUMBER_OF_INPUTS = 33

        # Models
        self.e_model = e_model
        self.l_model = l_model
        self.e_static_model = e_static_model
        self.behavior_model = behavior_model

        self.time_since_last_update = np.ones(number_of_circuits) * -1
        self.last_output_tracking = np.zeros((number_of_circuits, 1))

        # Setup last_output_tracking based on input zero to get started
        times = (np.ones((self.number_of_circuits, 1)) * self.CLOCK_PERIOD)
        input_voltages = np.zeros((self.number_of_circuits, self.NUMBER_OF_INPUTS))
        all_together = np.concatenate((times, self.last_output_tracking, input_voltages, self.circuit_params), axis=1)
        self.last_output_tracking = self.behavior_model.predict(all_together)


        # Book keeping
        self.leak_events_energy_per_time_step = defaultdict(list)
        self.input_events_energy_per_time_step = defaultdict(list)
        self.input_events_latency_per_time_step = defaultdict(list)
        self.input_events_behavior_per_time_step = defaultdict(list)

    def step(self, time_step_id, input_events, is_final_event):
        input_circuit_ids = np.arange(self.number_of_circuits)

        if is_final_event:
            # Clean up static energy calculation for everyone that does not have an input
            ids_not_changing = np.setdiff1d(self.possible_circuit_ids, input_circuit_ids)

            if ids_not_changing.size != 0:
                input_voltages = np.zeros((ids_not_changing.shape[0], self.NUMBER_OF_INPUTS))
                times_to_include = ((time_step_id - self.time_since_last_update[ids_not_changing]) * self.CLOCK_PERIOD).reshape(-1, 1)
                leak_params = self.circuit_params[ids_not_changing]
                last_outputs = self.last_output_tracking[ids_not_changing].reshape(-1, 1)
                all_params = np.concatenate((times_to_include, last_outputs, input_voltages, leak_params), axis=1)
                energy = self.e_static_model.predict(all_params)
                self.leak_events_energy_per_time_step[time_step_id].append((ids_not_changing, energy))

        if input_events.shape[0] != 0:
            time_since_last = self.time_since_last_update[input_circuit_ids]
            mask = time_since_last < (time_step_id - 1)
            circuits_with_leak = input_circuit_ids[mask]

            if circuits_with_leak.shape[0] > 0:
                times = ((time_step_id - time_since_last[mask] - 1) * self.CLOCK_PERIOD).reshape(-1, 1)
                input_voltages = np.zeros((circuits_with_leak.shape[0], self.NUMBER_OF_INPUTS))
                leak_params = self.circuit_params[circuits_with_leak]
                last_outputs = self.last_output_tracking[circuits_with_leak].reshape(-1, 1)
                all_params = np.concatenate((times, last_outputs, input_voltages, leak_params), axis=1)
                energy = self.e_static_model.predict(all_params)

                # Housekeeping
                self.leak_events_energy_per_time_step[time_step_id].append((circuits_with_leak, energy))

            params = self.circuit_params[input_circuit_ids]
            times = np.ones((input_circuit_ids.shape[0], 1)) * self.CLOCK_PERIOD
            last_outputs = self.last_output_tracking[input_circuit_ids].reshape(-1, 1)
            all_together = np.concatenate((times, last_outputs, input_events, params), axis=1)

            output = self.behavior_model.predict(all_together)
            energy = self.e_model.predict(all_together)
            latency = self.l_model.predict(all_together)

            self.last_output_tracking[input_circuit_ids] = output
            self.time_since_last_update[input_circuit_ids] = time_step_id

            # Housekeeping
            self.input_events_energy_per_time_step[time_step_id].append((input_circuit_ids, energy))
            self.input_events_latency_per_time_step[time_step_id].append((input_circuit_ids, latency))
            self.input_events_behavior_per_time_step[time_step_id].append((input_circuit_ids, output))
        
        return (output, latency, energy, last_outputs)