#mapIMAC module connects the hidden layers and sets the configurations in the SPICE netlist
import os
from mapPartitionIMAC import *
from tools_helper_imac import *

"""
This script builds upon and extends work presented in the following publication:

Md Hasibul Amin, Mohammed E. Elbtity, and Ramtin Zand. 2023.
IMAC-Sim: A Circuit-level Simulator For In-Memory Analog Computing Architectures.
In Proceedings of the Great Lakes Symposium on VLSI 2023 (GLSVLSI '23),
Association for Computing Machinery, New York, NY, USA, 659–664.
https://doi.org/10.1145/3583781.3590264

We acknowledge that most of the neural network structure, SPICE simulation framework,
and core evaluation code in this file are derived from or adapted based on the
IMAC-Sim framework presented in the above publication. All credit for the foundational
SPICE code and in-memory analog computing simulation logic belongs to the original authors.

We however, adapt the code to fit our own architectural paradigm and to get relevant statistics
from the SPICE simulation. As such, this file as been provided to recreate the Crossbar SPICE
simulations, if such tools are available on whatever machine is being run.
"""

def mapIMAC(nodes,xbar_length,hpar,vpar,metal,T,H,L,W,D,eps,rho,weight_var,testnum,data_dir,spice_dir,vdd,vss,tsampling):
    # For each layer / activation layer, create separate file
    layers = {}
    layer_cuts = {}
    layers_to_run = []

    for i in range(len(nodes)-1):
        # Create File and Header
        f=open(os.path.join(spice_dir,f'full_layer_{i+1}_crossbar.sp'), "w")
        layers_to_run.append(f'full_layer_{i+1}_crossbar.sp')

        # Write Header
        f.write(f"*Layer {i+1} Crossbar Array\n")
        f.write(".lib './models' ptm14hp\n")    #the transistor library can be changed here (The current format does not use transistor for the weighted array)	
        f.write('.include diff'+str(i+1)+'.sp\n')
        f.write('.option ingold=2 artist=2 psf=2\n')
        f.write('.OPTION DELMAX=.1NS\n')
        f.write('.OPTION RELTOL=1e-6 ABSTOL=1e-12 VABSTOL=1e-12\n')
        f.write('.option probe\n')
        f.write(".op\n")
        f.write(".PARAM VddVal=%f\n"%vdd)
        f.write(".PARAM VssVal=%f\n"%vss)
        f.write(".PARAM tsampling=%fn\n"%tsampling)
        
        things_to_probe = []

        # Create Partitions for Layer
        hor_cut, vert_cut, layer_keys = mapPartition(nodes[i],nodes[i+1],xbar_length, i+1, hpar[i],vpar[i],metal,T,H,L,W,D,eps,rho,weight_var,data_dir,spice_dir)

        # Include all of the created files
        for x_id, y_id, split_r in layer_keys:
            format_string = f".include layer_{i+1}_{x_id}_{y_id}_{split_r}.sp\n"
            f.write(format_string)

        ## Create Crossbar Invocations
        layer_output_neurons = nodes[i+1]

        # Note: If same layer, same y_id, and same row, go to the same output number 
        input_name = "layer_{}_in{} "

        if i!=0:
            input_name = "layer_{}_neuron_output_{} "

        print(f"writing layer {i+1}")
        f.write(f"\n\n********** Layer {i+1} **********\n")

        for (x_id, y_id, split_r) in layer_keys:
            # Use separate vdd, vss 
            vdd_name = f"vdd_{i+1}_{x_id}_{y_id}_{split_r}"
            vss_name = f"vss_{i+1}_{x_id}_{y_id}_{split_r}"
            f.write(f"{vss_name} {vss_name} 0 DC VssVal\n")
            f.write(f"{vdd_name} {vdd_name} 0 DC VddVal\n")
            f.write(f"Xlayer_{i+1}_{x_id}_{y_id}_{split_r} {vdd_name} {vss_name} 0 ")
            
            # Keep track of things to probe
            things_to_probe.append(f"v({vdd_name})")
            things_to_probe.append(f"v({vss_name})")
            things_to_probe.append(f"i({vdd_name})")
            things_to_probe.append(f"i({vss_name})")

            # Calculate input ranges 
            low_range_x = hor_cut[x_id-1]
            high_range_x = hor_cut[x_id]

            if x_id == len(hor_cut)-1:
                high_range_x -= 1

            # Calculate y
            low_range_y = vert_cut[y_id-1]
            global_y_value = (low_range_y - 1) + split_r
            
            # Write inputs 
            for j in range(low_range_x, high_range_x):
                f.write(input_name.format(i, j))

            # Write zeroes for 32 crossbar when partitioned funny
            low_index = hor_cut[x_id] - hor_cut[x_id-1]+1

            # On last horizontal index, get rid of bias counting as one of the 32 inputs that we need to write.
            if x_id == len(hor_cut)-1:
                low_index -= 1

            high_index = xbar_length

            #print(low_index, high_index+1)

            for j in range(low_index, high_index+1):
                f.write("0 ")

            # Write output
            output_name = f"layer_{i+1}_{x_id}_{y_id}_{split_r}_out"
            f.write(output_name + " ")
            things_to_probe.append(f"v({output_name})")
            
            f.write(f"layer_{i+1}_{x_id}_{y_id}_{split_r}\n")

        # Connect to Output Capacitor 
        f.write(f"\n\n********** Output Capacitors in Layer {i+1} **********\n")
        for (x_id, y_id, split_r) in layer_keys:
            f.write(f"C_{x_id}_{y_id}_{split_r} layer_{i+1}_{x_id}_{y_id}_{split_r}_out 0 500f\n")
        
        # Write Input
        if i == 0:
            f.write("\n\n**********Input Test****************\n\n")
            c=open(data_dir+'/'+'data_sim.txt', "r")
            input_str = c.readlines()[0].split()
            input_num = [float(num) for num in input_str]
            for line in range(nodes[0]):
                f.write("v%d layer_0_in%d 0 PWL( 0n 0 "%(line+1,line+1))
                things_to_probe.append(f"i(v{line+1})")
                things_to_probe.append(f"v(layer_0_in{line+1})")
                for image in range(testnum+1):
                    if image == 0:
                        # Set everything to zero in the first stage in order to have everything settle first.
                        f.write("%fn %f %fn %f "%(image*tsampling+(0.1*tsampling),0,(image+1)*tsampling,0))
                    else:
                        f.write("%fn %f %fn %f "%(image*tsampling+(0.1*tsampling),input_num[line+(image-1)*nodes[0]],(image+1)*tsampling,input_num[line+(image-1)*nodes[0]]))
                f.write(")\n")
            c.close()
        else:
            # Get neuron layer outputs from previous layer 
            # Write input based on previous stuff 
            f.write("\n\n**********Input Test****************\n\n")
            
            for n in range(nodes[i]):
                name = input_name.format(i, n+1)
                filepath = os.path.join("pwl_files", f"neuron_{i}_{n}_out.txt")
                write_input_spike_file(f, f"v{n+1}", "first_net_not_used", name, filepath, 0, simulator='hspice', write_voltage_src=False, current_src=False)

                things_to_probe.append(f"i(v{n+1})")
                things_to_probe.append(f"v({name})")

        # Write transient analysis
        f.write(".TRAN 0.1n %d*tsampling\n"%(testnum+1))

        # Write Probes
        for guy in things_to_probe:
            f.write(f".probe {guy}\n")

        # Write .end 
        f.write(".end")

        f.close() 

        layers[i+1] = layer_keys
        layer_cuts[i+1] = (hor_cut, vert_cut)

    return (layers_to_run, None, layers, layer_cuts)
			
			
