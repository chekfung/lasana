* include Libraries for the 45nm LP PTM Model
.include "../libraries/45nm_LP.pm"

* Create subcircuit for the neuron
* Assume that PMOS is just 2x width of NMOS
.subckt lif_neuron spike_in v_lk v_sf v_rtr v_adap v_spk vdd gnd

* Memory
Cmem spike_in gnd 500fF IC=0V

* Source Follower (increase linear integration, modulate neuron threshold voltage)
M1 vdd spike_in v_in gnd NMOS l=0.045u w=0.3u
M2 v_in v_sf gnd gnd NMOS l=0.045u w=0.3u

* Inverter (reduces switching short-circuit currents at input)
M3 m3_gate m3_gate vdd vdd PMOS l=0.045u w=0.6u
M4 v_o1 v_in m3_gate vdd PMOS l=0.045u w=0.6u
M5 v_o1 v_in gnd gnd NMOS l=0.045u w=0.3u

* Feedback Loop V_o1
M6 m6_m7_net v_o1 vdd vdd PMOS l=0.045u w=0.6u
M7 spike_in m3_gate m6_m7_net vdd PMOS l=0.045u w=0.6u

* Feedback Loop V_o2 (Inverter with controlable slew rate)(Set arbitrary refractory periods)
M8 m8_m9 m8_m9 vdd vdd PMOS l=0.045u w=0.6u

M9 v_o2 v_o1 m8_m9 vdd PMOS l=0.045u w=0.6u
M10 v_o2 v_o1 m10_m11_net gnd NMOS l=0.045u w=0.3u

M11 m10_m11_net v_rtr gnd gnd NMOS l=0.045u w=0.6u

M12 spike_in v_o2 gnd gnd NMOS l=0.045u w=2.5u

* Output Circuit Inverter (Generate fast digital spike occurence)
M13 v_spk v_o1 vdd vdd PMOS l=0.045u w=0.6u
M14 v_spk v_o1 gnd gnd NMOS l=0.045u w=0.3u

* Adaptive Feedback (STDP)
M15 m15_m16_net v_o1 vdd vdd PMOS l=0.045u w=0.6u
M16 m_17_gate_in v_adap m15_m16_net vdd PMOS l=0.045u w=0.6u

M17 m_17_gate_in m_17_gate_in gnd gnd NMOS l=0.045u w=0.3u
M18 m_17_gate_in v_spk v_ca gnd NMOS l=0.045u w=0.3u
M19 spike_in v_ca gnd gnd NMOS l=0.045u w=0.3u

* Constant current leak transistor
M20 spike_in v_lk gnd gnd NMOS l=0.045u w=1u

.ends
