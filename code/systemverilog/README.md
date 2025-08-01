# SystemVerilog RNM Models: LIF Neurons & MAC Units
This folder contains **SystemVerilog Real Number Models (SV-RNM)** and testbench implementations of:
- Indiveri Leaky Integrate and Fire Neurons
- 32x1 PCM Crossbar Rows

## Files
- **lif_neuron_rnm_tb.sv**: File contains the testbench which instantiate X number of LIF neurons from lif_neuron_rnm.sv, and then attempts to run the respective inputs from the PWL files that were generated in the past from `testbench_generation.py`. 

- **lif_neuron_rnm.sv**: File contains the source code for the SV-RNM model of the Indiveri LIF neuron. We note that one can uncomment the static and dynamic comments to count the number of static (E2 events) versus dynamic (E1, E3 events) such that we can simulate the amount of time it would take LASANA to provide power and performance annotation after simulation.

- **mac_unit_many_tb.sv**: File contains the testbench which instantiates X number of MAC units from mac_unit.sv. Unlike the previous testbenches, this testbench creates the inputs from scratch (due to the amount of files that need to be open (32x20000 at the largest)), making the process I/O bound, rather than compute bound. We give the benefit of the doubt here and assume that it is bound by simulation, not by I/O.

- **mac_unit.sv**: File contains the source code for the SV-RNM model of the PCM 32-input Crossbar row. 

## Usage
These files require access to Cadence Spectre Xcelium. We run ours on Spectre Xcelium 23.03.

The following function runs the neuron file in question

`(time xrun lif_neuron_rnm_tb.sv lif_neuron_rnm.sv -sv) &>> "lif_neuron_rnm_tb_log_counters_real.txt"`
