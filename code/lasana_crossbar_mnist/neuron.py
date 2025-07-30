import numpy as np
from collections import defaultdict

class DigitalNeuron:
    def __init__(self, number_of_circuits,
                 e_model, l_model, e_static_model, behavior_model,
                 CLOCK_PERIOD):
        # Params
        self.number_of_circuits = number_of_circuits
        self.possible_circuit_ids = np.arange(number_of_circuits)
        self.CLOCK_PERIOD = CLOCK_PERIOD
        self.NUMBER_OF_INPUTS = 1

        # Models
        self.e_model = None
        self.l_model = None
        self.e_static_model = None
        self.behavior_model = None

        self.time_since_last_update = np.ones(number_of_circuits) * -1
        self.last_output_tracking = np.zeros((number_of_circuits, 1))

        # Setup so that not starting at zero, but instead based on what they are starting on
        input_voltages = np.zeros((self.number_of_circuits, self.NUMBER_OF_INPUTS))
        self.last_output_tracking = 0.8 / (1+np.exp(11*input_voltages))

        # Book keeping
        self.leak_events_energy_per_time_step = defaultdict(list)
        self.input_events_energy_per_time_step = defaultdict(list)
        self.input_events_latency_per_time_step = defaultdict(list)
        self.input_events_behavior_per_time_step = defaultdict(list)

    def step(self, time_step_id, input_events, is_final_event):
        input_circuit_ids = np.arange(self.number_of_circuits)

        # Analytical Sigmoid that IMAC Sim trains on.
        output = 0.8 / (1+np.exp(11*input_events))
        output = output.reshape(-1)

        # Do not predict energy and latency here
        energy = np.zeros((self.number_of_circuits))
        latency = np.zeros((self.number_of_circuits))

        # House Keeping
        self.last_output_tracking[input_circuit_ids] = output.reshape(-1,1)
        self.time_since_last_update[input_circuit_ids] = time_step_id
        last_outputs = self.last_output_tracking[input_circuit_ids].reshape(-1, 1)

        self.input_events_energy_per_time_step[time_step_id].append((input_circuit_ids, energy))
        self.input_events_latency_per_time_step[time_step_id].append((input_circuit_ids, latency))
        self.input_events_behavior_per_time_step[time_step_id].append((input_circuit_ids, output))
        
        return (output, latency, energy, last_outputs)