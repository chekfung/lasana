#mapLayer module writes the subcircuit netlist for each of the layers separately

import random
import numpy as np
import resource
import os

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

# Quick Band Aid Fix to Make My Life a little easier 
# Increase the number of files I can open at a time 
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
print(f"Soft limit: {soft}")
print(f"Hard limit: {hard}")

resource.setrlimit(resource.RLIMIT_NOFILE, (min(20000, hard), hard))  # Change 4096 as needed

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
print(f"Soft limit: {soft}")
print(f"Hard limit: {hard}")


def find_partition(partitions, index):
    """
    Given a list of split indices and a target index,
    returns (partition_number, index_within_partition).

    # Note Indices are 1-indexed, and partitions are non-inclusive
    # That is, 1:30 means that the partition is from 1 to 29
    
    Example:
    partitions = [1, 30, 50]
    index = 35
    => (2, 6)
    """
    for i in range(len(partitions) - 1):
        start = partitions[i]
        end = partitions[i + 1]
        if start <= index < end:
            return i+1, (index - start) + 1
    raise ValueError(f"Index {index} not found in any partition range.")


def mapPartition(layer1,layer2, xbar_length, LayerNUM,hpar,vpar,metal,T,H,L,W,D,eps,rho,weight_var,data_dir,spice_dir): 
    # updating the resistivity for specific technology node
    l0 = 39e-9 # Mean free path of electrons in Cu
    d = metal # average grain size, equal to wire width
    p=0.25 # specular scattering fraction
    R=0.3 # probability for electron to reflect at the grain boundary
    alpha = l0*R/(d*(1-R)) # parameter for MS model
    dsur_scatt = 0.75*(1-p)*l0/metal # surface scattering
    dgrain_scatt = pow((1-3*alpha/2+3*pow(alpha,2)-3*pow(alpha,3)*np.log(1+1/alpha)),-1) # grain boundary scattering
    rho_new = rho * (dsur_scatt + dgrain_scatt) # new resistivity
    layer1_wb = layer1+1 # number of bitcell in a row including weights and bias

    # Determine where vertical and horizontal partitions are (from IMAC Sim)
    # NOTE: if [1,30, 58], the first partition is 1,29, and the second is 30, 57  as the top end is noninclusive.
    horizontal_cuts = [1]

    n_hpar=1 # horizontal partition number
    c=1 # column number
    r=1 # row number
    posw_r=open(data_dir+'/'+'posweight'+str(LayerNUM)+".txt", "r") # read the positive line conductances
    for line in posw_r:
        if (float(line)!=0):
            if (r < layer2+1):
                r+=1
            else:
                c+=1
                r=1
                if (c == int(layer1_wb*n_hpar/hpar+min((layer1_wb%hpar)/n_hpar,1)+1)):
                    n_hpar+=1

                    # Horizontal Partition Here
                    horizontal_cuts.append(c)
                r+=1
        else:
            r+=1
    horizontal_cuts.append(layer1_wb+1)
    posw_r.close()

    # Formally Print horizontal cuts 
    output = f"Horizontal partitions Layer: {LayerNUM}: "
    output += " | ".join(f"[{horizontal_cuts[i]}:{horizontal_cuts[i+1]-1}]" for i in range(len(horizontal_cuts) - 1))
    print(output)
    print(f"Difference: {np.diff(horizontal_cuts)}")
    print(f"Note: Last Horizontal Partition gets the bias line")

    # Determine Vertical Partitions 
    vertical_cuts = [1]
    
    # writing the circuit for vertical line parasitic resistances
    parasitic_res = rho_new*W/(metal*T)
    for i in range(layer1_wb):
        n_vpar=1 # vertical partition number
        c=i+1 # column number
        for j in range(layer2):
            r=j+1 # row number
            if (i == layer1): # only for the bias line (last line to write btw)
                if (j == 0):
                    temp=1
                elif (j == int(layer2*n_vpar/vpar+min((layer2%vpar)/n_vpar,1))):
                    temp=1
                    n_vpar+=1
                    vertical_cuts.append(r)
                else:
                    temp=1
    
    vertical_cuts.append(layer2+1)
    output = f"Vertical partitions Layer: {LayerNUM}: "
    output += " | ".join(f"[{vertical_cuts[i]}:{vertical_cuts[i+1]-1}]" for i in range(len(vertical_cuts) - 1))
    print(output)
    print(f"Difference: {np.diff(vertical_cuts)}")
    print(f"Note: Each Vertical Guy gets own bias. Each Partition's initial wire connects to vdd")

    # Open file descriptors for all possible things that we need to open (Each guy is xbar_length x 1) 
    # File descriptor named: partioned_layer_{layer_num}_{hpar}_{vpar}_{split_vpar}.sp
    # We split vpar since we have 32x1 based mac units.
    open_fd = {}    # Index into this using hpar, vpar, split_vpar index

    for x_id in range(hpar):
        for y_id in range(vpar):
            # Vertical refers to having to split up input into multiple 
            # In our case, even though we will have the xbar_length x xbar_length, we will split to 32x1 for parallezability
            # Get vertical cuts length 
            new_range_low = vertical_cuts[y_id]
            new_range_high = vertical_cuts[y_id+1]
            new_range = (new_range_high - new_range_low)

            for split_vpar in range(new_range):
                file_template = f"layer_{LayerNUM}_{x_id+1}_{y_id+1}_{split_vpar+1}.sp"
                fd = open(os.path.join(spice_dir,file_template), "w")

                # Write subcircuit definition 
                fd.write(f".SUBCKT layer_{LayerNUM}_{x_id+1}_{y_id+1}_{split_vpar+1}"+" vdd vss 0 ")

                # Number of input and output in the circuit definition 
                for i in range(xbar_length):
                    fd.write("in%d "%(i+1))
                
                fd.write("out%d"%(1))

                fd.write("\n\n**********Positive Weighted Array**********\n")

                open_fd[(x_id+1, y_id+1, split_vpar+1)] = fd    

    
    # Open all the weights and bias files 
    posw_r=open(data_dir+'/'+'posweight'+str(LayerNUM)+".txt", "r") # read the positive line conductances
    negw_r=open(data_dir+'/'+'negweight'+str(LayerNUM)+".txt", "r")
    posb_r=open(data_dir+'/'+'posbias'+str(LayerNUM)+".txt", "r")
    negb_r=open(data_dir+'/'+'negbias'+str(LayerNUM)+".txt", "r")

    # Write Positive Array 
    n_hpar=1 # horizontal partition number
    c=1 # column number
    r=1 # row number

    low_resistance = np.inf
    high_resistance = -np.inf

    for line in posw_r:
        if (float(line)!=0):
            if float(line) < low_resistance:
                low_resistance = float(line)
            
            if float(line) > high_resistance:
                high_resistance = float(line)

            if (r < layer2+1):
                # Calculate Indices  (Row is vertical partitioning, while column is vertical partitioning)
                y_id, split_r = find_partition(vertical_cuts, r)
                x_id, split_c = find_partition(horizontal_cuts, c)
                open_fd[(x_id, y_id, split_r)].write("Rwpos%d_%d in%d_%d sp%d_%d %f\n"% (split_c,split_r, split_c,split_r,split_c,split_r,float(line)))

                r+=1
            else:
                c+=1
                r=1
                if (c == int(layer1_wb*n_hpar/hpar+min((layer1_wb%hpar)/n_hpar,1)+1)):
                    n_hpar+=1

                # Calculate Indices  (Row is vertical partitioning, while column is vertical partitioning)
                y_id, split_r = find_partition(vertical_cuts, r)
                x_id, split_c = find_partition(horizontal_cuts, c)

                open_fd[(x_id, y_id, split_r)].write("Rwpos%d_%d in%d_%d sp%d_%d %f\n"% (split_c,split_r, split_c,split_r,split_c,split_r,float(line)))

                r+=1
        else:
            r+=1
    
    # Add the missing things to make it actually a Xbar_length x Xbar_length crossbar array (Note: for weight of 0, it is )
    # Go through and write each of the files 
    for key, file in open_fd.items():
        (x_id, y_id, split_r) = key

        # Memristor ID to start on 
        low_index = horizontal_cuts[x_id] - horizontal_cuts[x_id-1]+1

        # On last horizontal index, get rid of bias counting as one of the 32 inputs that we need to write.
        if x_id == len(horizontal_cuts)-1:
            low_index -= 1

        high_index = xbar_length

        open_fd[(x_id, y_id, split_r)].write("\n\n**********Positive Weighted Array Extras **********\n")

        for new_c in range(low_index, high_index+1):
            open_fd[(x_id, y_id, split_r)].write("Rwpos%d_%d in%d_%d 0 %f\n"% (new_c,split_r, new_c,split_r, low_resistance))
            #positive_weights[(x_id, y_id, split_r)].append(float(low_resistance))

    # Write Negative Array
    for key, file in open_fd.items():
        try:
            file.write("\n\n**********Negative Weighted Array**********\n")

        except Exception as e:
            print(f"Error {key}: {e}")
            
    n_hpar=1 # horizontal partition number
    c=1 # column number
    r=1 # row number
    for line in negw_r:
        if (float(line)!=0):
            if (r < layer2+1):
                y_id, split_r = find_partition(vertical_cuts, r)
                x_id, split_c = find_partition(horizontal_cuts, c)
                open_fd[(x_id, y_id, split_r)].write("Rwneg%d_%d in%d_%d sn%d_%d %f\n"% (split_c,split_r,split_c,split_r,split_c,split_r,float(line)))

                r+=1
            else:
                c+=1
                r=1
                if (c == int(layer1_wb*n_hpar/hpar+min((layer1_wb%hpar)/n_hpar,1)+1)):
                    n_hpar+=1

                y_id, split_r = find_partition(vertical_cuts, r)
                x_id, split_c = find_partition(horizontal_cuts, c)
                open_fd[(x_id, y_id, split_r)].write("Rwneg%d_%d in%d_%d sn%d_%d %f\n"% (split_c,split_r,split_c,split_r,split_c,split_r,float(line)))
                r+=1
        else:
            r+=1
    
    # Write the extras to pad to xbar_length x xbar_length
    for key, file in open_fd.items():
        (x_id, y_id, split_r) = key

        # Memristor ID to start on 
        low_index = horizontal_cuts[x_id] - horizontal_cuts[x_id-1]+1

        # On last horizontal index, get rid of bias counting as one of the 32 inputs that we need to write.
        if x_id == len(horizontal_cuts)-1:
            low_index -= 1

        high_index = xbar_length

        open_fd[(x_id, y_id, split_r)].write("\n\n**********Negative Weighted Array Extras **********\n")

        for new_c in range(low_index, high_index+1):
            open_fd[(x_id, y_id, split_r)].write("Rwneg%d_%d in%d_%d 0 %f\n"% (new_c,split_r, new_c,split_r, low_resistance))

    # Write 0 bias numbers for partitions where there is nothing happening 
    for i in range(hpar-1):
        for y_id in range(vpar):
            vpar_index = y_id+1

            new_range_low = vertical_cuts[y_id]
            new_range_high = vertical_cuts[y_id+1]
            new_range = (new_range_high - new_range_low)

            for split_vpar in range(new_range):
                open_fd[(i+1, vpar_index, split_vpar+1)].write("\n\n**********Zero Positive Biases**********\n")
                open_fd[(i+1, vpar_index, split_vpar+1)].write("Rbpos%d vd%d sp%d_%d %f\n"% (1,1,xbar_length+1,split_vpar+1, low_resistance))

                open_fd[(i+1, vpar_index, split_vpar+1)].write("\n\n**********Zero Negative Biases**********\n")
                open_fd[(i+1, vpar_index, split_vpar+1)].write("Rbneg%d vd%d sn%d_%d %f\n"% (1,1,xbar_length+1,split_vpar+1,low_resistance))
                

    # Biases need to be written for every vertical partition, but only for the last horizontal partition
    for y_id in range(vpar):
        vpar_index = y_id+1

        new_range_low = vertical_cuts[y_id]
        new_range_high = vertical_cuts[y_id+1]
        new_range = (new_range_high - new_range_low)

        for split_vpar in range(new_range):
            # Write positive bias
            open_fd[(hpar, vpar_index, split_vpar+1)].write("\n\n**********Positive Biases**********\n")

            line = posb_r.readline()

            if (float(line) != 0):
                open_fd[(hpar, vpar_index, split_vpar+1)].write("Rbpos%d vd%d sp%d_%d %f\n"% (1,1,xbar_length+1,split_vpar+1,float(line)))

            # Write negative bias
            open_fd[(hpar, vpar_index, split_vpar+1)].write("\n\n**********Negative Biases**********\n")

            line = negb_r.readline()

            if (float(line) != 0):
                open_fd[(hpar, vpar_index, split_vpar+1)].write("Rbneg%d vd%d sn%d_%d %f\n"% (1,1,xbar_length+1,split_vpar+1,float(line)))

    # writing the circuit for vertical line parasitic resistances (only one vertical line for each row BTW)
    parasitic_res = rho_new*W/(metal*T)

    for key, file in open_fd.items():
        (x_id, y_id, split_r) = key

        file.write("\n\n**********Parasitic Resistances for Vertical Lines**********\n")
        
        for i in range(xbar_length+1):
            n_vpar=1 # vertical partition number
            c=i+1 # column number
            for j in range(1):
                r=j+1 # row number
                if (i == xbar_length): # only for the bias line
                    if (j == 0):
                        file.write("Rbias%d vdd vd%d %f\n"% (r,r,parasitic_res))
                
                else: # the input connected vertical lines
                    if (j == 0):
                        file.write("Rin%d_%d in%d in%d_%d %f\n"% (c,r,c,c,split_r,parasitic_res))

    # Write Horizontal Line Parasitic Resistances
    hor_parasitic_res = rho_new*L/(metal*T)

    for key, file in open_fd.items():
        (x_id, y_id, split_r) = key

        file.write("\n\n**********Parasitic Resistances for I+ and I- Lines****************\n")
        n_hpar=1 # horizontal partition number
        for i in range(xbar_length+1):
            c=i+1 # column number
            for j in range(1):
                r=j+1 # row number
                if (i == xbar_length):
                    file.write("Rsp%d_%d sp%d_%d sp%d_p%d %f\n"% (c,split_r,c,split_r,r,n_hpar,hor_parasitic_res))
                    file.write("Rsn%d_%d sn%d_%d sn%d_p%d %f\n"% (c,split_r,c,split_r,r,n_hpar,hor_parasitic_res))

                else:
                    file.write("Rsp%d_%d sp%d_%d sp%d_%d %f\n"% (c,split_r,c,split_r,c+1,split_r,hor_parasitic_res))
                    file.write("Rsn%d_%d sn%d_%d sn%d_%d %f\n"% (c,split_r,c,split_r,c+1,split_r,hor_parasitic_res))

    # Write Diff and then output
    for key, file in open_fd.items():
        (x_id, y_id, split_r) = key

        # writing the circuit for Op-AMPS and connecting resistors
        file.write("\n\n**********Weight Differntial Op-AMPS and Connecting Resistors****************\n")

        file.write("XDIFFw%d_p%d sp%d_p%d sn%d_p%d nin%d_%d diff%d\n"% (1,1,1,1,1,1,1,1,LayerNUM))
        file.write("Rconn%d_p%d nin%d_%d out%d 1m\n"% (1,1,1,1,1))

        # Write end of the guy
        file.write(f".ENDS layer_{LayerNUM}_{x_id}_{y_id}_{split_r}")

    # Close All File Descriptors 
    posw_r.close()
    negw_r.close()
    posb_r.close()
    negb_r.close()

    # Close all other file descriptors
    # Close all file descriptors
    for key, file in open_fd.items():
        try:
            file.close()
            #print(f"Closed: {key}")
        except Exception as e:
            print(f"Error closing {key}: {e}")

    return (horizontal_cuts, vertical_cuts, open_fd.keys())
