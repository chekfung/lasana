#testIMAC is the parent file for the python framework, which initiates the simulation and analyzes the results from SPICE

import re
import os
import time
import math
import random
import mapIMACPartition_Separate_Files
import mapWB
import numpy as np
import csv
from tools_helper import *
import matplotlib.pyplot as plt 

start = time.time()

testnum=10000 #Number of input test cases to run
testnum_per_batch=10 #Number of test cases in a single batch, testnum should be divisible by this number
firstimage=0 #start the test inputs from this image\
csv_name = '5_12_25_8_bit_quantization.csv'#'5_11_2025_DIGITAL_NEURON.csv'
csv_folder = 'separated_csvs_5_12_25_8_bit_quantization'#'separated_csvs_5_11_25_DIGITAL_NEURON'

# Quantization Constants
USE_QUANTIZATION = True
DAC_BITS = 8

#list of inputs start
data_dir='data' #The directory where data files are located
spice_dir='test_spice' #The directory where spice files are located
dataset_file='test_data.csv' #Name of the dataset file
label_file='test_labels.csv' #Name of the label file
weight_var=0.0 #percentage variation in the resistance of the synapses
vdd=0.8 #The positive supply voltage
vss=-0.8 #The negative supply voltage
tsampling=4 #The sampling time in nanosecond    # FIXME: Change this to 5ns later on :)
nodes=[400,120,84,10] #Network Topology, an array which defines the DNN model size
xbar=[32,32] #The crossbar size
gain=[30,30,10] #Array for the differential amplifier gains of all hidden layers
tech_node=9e-9 #The technology node e.g. 9nm, 45nm etc.
metal=3*tech_node #Width of the metal line for parasitic calculation
T=22e-9 #Metal thickness
H=20e-9 #Inter metal layer spacing
L=15*tech_node #length of the bitcell
W=12*tech_node #width of the bitcell
D=5*tech_node #distance between I+ and I- lines
eps = 20*8.854e-12 #permittivity of oxide
rho = 1.9e-8 #resistivity of metal
#rlow=5e3 #Low resistance level of the memristive device
#rhigh=15e3 #High resistance level of the memristive device
rlow=78000
rhigh=202000
#list of inputs end

hpar=[math.ceil((x+1)/xbar[0]) for x in nodes] #Calculating the horizontal partitioning array for all hidden layers
hpar.pop() #The last value in the array is removed for hpar
vpar=[math.ceil(x/xbar[1]) for x in nodes] #Calculating the vertical partitioning array for all hidden layers
vpar.pop(0) #The first value in the array is removed for vpar

print('Rlow=%f'%rlow)
print('Rhigh=%f'%rhigh)
print('Horizontal partitions = '+str(hpar))
print('Vertical partitions = '+str(vpar))

def adc(voltage, v_min=-0.8, v_max=0.8, bits=8):
    """
    Converts an analog voltage to a digital ADC value with saturation.
    """
    levels = 2 ** bits
    step_size = (v_max - v_min) / (levels - 1)

    # Saturate voltage
    voltage = np.clip(voltage, v_min, v_max)


    return np.round((voltage - v_min) / step_size).astype(int)


def dac(adc_value, v_min=-0.8, v_max=0.8, bits=8):
    """
    Converts digital ADC values (scalar or array) back to analog voltages.
    """
    levels = 2 ** bits
    step_size = (v_max - v_min) / (levels - 1)

    adc_value = np.clip(adc_value, 0, levels - 1)  # element-wise clipping

    return v_min + adc_value * step_size

def create_pwl_from_input_pulses(filename, time_of_each_input, transition_time, input_vector):
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

#function to update the device resistances in the neuron.sp file, which includes the spice file for activation function
def update_neuron (rlow,rhigh):
    ff=open(spice_dir+'/'+'neuron.sp', "r+")
    i=0
    data= ff.readlines()
    for line in data:
        i+=1
        if 'Rlow' in line:
            data[i-1]='Rlow in2 input ' + str(rlow) +'\n'
        if 'Rhigh' in line:
            data[i-1]='Rhigh input out ' + str(rhigh) +'\n'

    ff.seek(0)
    ff.truncate()
    ff.writelines(data)
    ff.close()

#function to update the gain for the differential amplifiers
def update_diff (gain, LayerNUM):
    name=spice_dir+'/'+'diff{}.sp'.format(LayerNUM)
    ff=open(name, "r+")
    i=0
    data= ff.readlines()
    for line in data:
        i+=1
        if 'Gain' in line:
            data[i-1]='*Differential Amplifier with Gain=' + str(gain) +'\n'
        if 'R3' in line:
            data[i-1]='R3 n1 out1 ' + str(gain) +'k\n'
        if 'R4' in line:
            data[i-1]='R4 n2 0 ' + str(gain) +'k\n'
    ff.seek(0)
    ff.truncate()
    ff.writelines(data)
    ff.close()


def min_max_normalization(vector):
    """
    Performs min-max normalization on the input vector, scaling its values to the range [0, 1].

    Parameters:
    - vector: NumPy array or list containing the values to be normalized.

    Returns:
    - normalized: The normalized vector.
    - global_min: The minimum value of the original vector.
    - global_max: The maximum value of the original vector.
    """
    global_min = np.min(vector)
    global_max = np.max(vector)
    normalized = (vector - global_min) / (global_max - global_min)

    return normalized, global_min, global_max

#function to extract the measured voltage at a specific time in the output text file genrated by SPICE
def findat (line):
    i=0
    m=0
    while (m == 0):
        i+=1;
        if (line[i]=='='):
            s1=i+1;
        if (line[i]=='w'):
            s2=i-1;
            m=1;
        if (line[i]=='\n'):
            s2=i;
            m=1;
    volt=line[s1:s2]
    volt=volt.replace(" ","")
    volt=volt.replace("m","e-3")
    volt=volt.replace("u","e-6")
    volt=volt.replace("n","e-9")
    volt=volt.replace("p","e-12")
    volt=volt.replace("f","e-15")
    volt=volt.replace("a","e-18")
    volt=volt.replace("k","e3")
    volt=volt.replace("x","e6")
    volt=volt.replace("g","e9")
    volt=volt.replace("t","e12")
    return volt


# Check whether output CSV file exists
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)
    print(f"Created folder: {csv_folder}")
else:
    print(f"Folder already exists: {csv_folder}")

# Check whether output CSV file exists
pwl_folder = "pwl_files"
if not os.path.exists(pwl_folder):
    os.makedirs(pwl_folder)
    print(f"Created folder: {pwl_folder}")
else:
    print(f"Folder already exists: {pwl_folder}")

#dataset preprocessing
dataset = np.genfromtxt(data_dir+'/'+dataset_file,delimiter=',')
dataset_flat = dataset.flatten()
dataset_bin = np.sign(dataset_flat)

data_w = open(data_dir+'/'+'testinput.txt', "w")
for i in range(len(dataset_bin)):
    data_w.write("%f\n"%(float(dataset_bin[i])))	
data_w.close()

#label preprocessing
label = np.genfromtxt(data_dir+'/'+'test_labels.csv',delimiter=',')
label_flat = label.flatten()
label_w = open(data_dir+'/'+'testlabel.txt', "w")
for i in range(len(label_flat)):
    label_w.write("%f\n"%(float(label_flat[i])))	
label_w.close()

data_r=open(data_dir+'/'+'testinput.txt', "r")   # testinput.txt includes the test images from the MNIST Dataset
label_r=open(data_dir+'/'+'testlabel.txt', "r")  # testlabel.txt includes the labels of the test images in the MNIST Dataset
data_all=data_r.readlines() #data_all contains all test images
label_all=label_r.readlines() #label_all contains all labels
length=len(nodes) #length contains the number of layers in DNN model
update_neuron(rlow,rhigh) #updates the resistances in the neuron
for i in range(len(nodes)-1):
    update_diff(gain[i],i+1) #updates the differential amplifier gains
mapWB.mapWB(length,rlow,rhigh,nodes,data_dir,weight_var) #calling mapWB which sets the corresponding resistance value for weights and biases
batch=testnum//testnum_per_batch #calculates the number of batch for the simulation
image_num=0 #number of image in the simulation
testimage=firstimage
err=[] #the array containing error information for each test case
pwr_list=[] #the array containing power information for each test case

# Create CSV to store eveything :)
headers = ['image_num', 'golden_label', 'predicted_label', 'energy'] + \
          [f'output{j}' for j in range(10)]

per_circuit_header = ['circuit_name', 'latency', 'energy', 'output_value', 'previous_output_value']

file = open(csv_name, mode='w', newline='')
writer = csv.writer(file)
writer.writerow(headers)

for i in range(batch):
    out_list=[]
    label_list=[]
    data_sim=data_all[int(testimage*nodes[0]):int((testimage+testnum_per_batch)*nodes[0])]
    label_sim=label_all[int(testimage*nodes[len(nodes)-1]):int((testimage+testnum_per_batch)*nodes[len(nodes)-1])]
    for value in label_sim:
        label_list.append(float(value))
    sim_w=open(data_dir+'/'+'data_sim.txt', "w")
    for j in range(int(testnum_per_batch*nodes[0])):	
        sim_w.write("%f "%(float(data_sim[j])*vdd))	
    sim_w.close()

    print("Calling mapIMAC.mapIMAC with the following arguments:\n")
    print(f"nodes: {nodes}")
    print(f"length: {length}")
    print(f"hpar: {hpar}")
    print(f"vpar: {vpar}")
    print(f"metal: {metal}")
    print(f"T: {T}")
    print(f"H: {H}")
    print(f"L: {L}")
    print(f"W: {W}")
    print(f"D: {D}")
    print(f"eps: {eps}")
    print(f"rho: {rho}")
    print(f"weight_var: {weight_var}")
    print(f"testnum_per_batch: {testnum_per_batch}")
    print(f"data_dir: {data_dir}")

    # Assume square partitions
    layers_to_run, _, layers_keys, layer_cuts = mapIMACPartition_Separate_Files.mapIMAC(nodes,xbar[0],hpar,vpar,metal,T,H,L,W,D,eps,rho,weight_var,testnum_per_batch,data_dir,spice_dir,vdd,vss,tsampling)
    print("Running Classifier")

    # Run Files, in order :)
    # Open file descriptors for everything in a batch
    batch_energies = np.zeros(testnum_per_batch)
    fd = []
    csv_writers = []

    for j in range(testnum_per_batch):
        real_image_id = (image_num + j) + firstimage

        # Create CSV :)
        filename = os.path.join(csv_folder, f"image_{real_image_id}_inference.csv")
        per_inference_fd = open(filename, mode='w', newline='')
        image_writer = csv.writer(per_inference_fd)
        image_writer.writerow(per_circuit_header)

        fd.append(per_inference_fd)
        csv_writers.append(image_writer)

    # Run through and run each of the nodes
    # Then analyze and then create PWL files to run the next file :)
    for layer_num in range(len(nodes)-1):
        # Crossbar to run
        print(f'Batch: {i} Run Layer: {layer_num+1}')
        layer_filepath = layers_to_run[layer_num]
        os.chdir(spice_dir)
        os.system(f'hspice {layer_filepath} > output_crossbar_{layer_num+1}.txt')
        os.chdir('..')

        # Get PSF File
        # Convert .tr0 log to psf to read
        print("Convert to PSF")
        runs_filepath = os.path.join(spice_dir, os.path.splitext(layer_filepath)[0])
        cmd2 = f'psf -i {runs_filepath}.tr0 -o {runs_filepath}.psf'
        exit_code = os.system(cmd2)

        if exit_code != 0:
            error_message = f"Error converting to PSF simulation #{int}, exit code: {exit_code}"
            print(error_message)
            exit()

        # Read psf file
        print("Read PSF")
        sim_obj = read_simulation_file(f'{runs_filepath}.psf', simulator='hspice')
        #print_signal_names(sim_obj, simulator='hspice')
        time_vec = get_signal('time', sim_obj, simulator='hspice')

        # Calculate Everything
        for j in range(testnum_per_batch):
            real_image_id = (image_num + j) + firstimage

            start_of_event = (j+1) * (tsampling * 10**-9) # Plus one since starting inference is always 0 for intiialization
            end_of_event = (j+2) * (tsampling *10**-9)  # Plus 2 since end of event (+1) but also for same reason above.

            # Convert to the indices that work with our SPICE simulation.
            bounds_mask = (time_vec >= start_of_event) & (time_vec <= end_of_event)
            valid_timestep_indices = np.where(bounds_mask)

            # Fix the end index to be plus one (so that we slightly overlap things.)
            start_index = valid_timestep_indices[0][0]  
            end_index_i = valid_timestep_indices[0][-1]+1
            end_index = np.min([end_index_i, len(time_vec)-1])      # Join the end of the index with the start of the new guy :)

            layer_crossbar_items = layers_keys[layer_num+1]
            hor_cut, vert_cut = layer_cuts[layer_num+1]

            input_name = "layer_{}_in{}"

            if layer_num!=0:
                input_name = "layer_{}_neuron_output_{}"

            # Go through each cross bar
            for x_id, y_id, split_r in layer_crossbar_items:
                # Calculate Energy (from unique vdd, vss)
                vdd_sig = get_signal(f"vdd_{layer_num+1}_{x_id}_{y_id}_{split_r}", sim_obj, simulator='hspice')
                i_vdd_sig = get_signal(f"i(vdd_{layer_num+1}_{x_id}_{y_id}_{split_r})", sim_obj, simulator='hspice')
                vss_sig = get_signal(f"vss_{layer_num+1}_{x_id}_{y_id}_{split_r}", sim_obj, simulator='hspice')
                i_vss_sig = get_signal(f"i(vss_{layer_num+1}_{x_id}_{y_id}_{split_r})", sim_obj, simulator='hspice')

                vdd_pwr = np.abs(vdd_sig[start_index:end_index+1] * i_vdd_sig[start_index:end_index+1])
                vss_pwr = np.abs(vss_sig[start_index:end_index+1] * i_vss_sig[start_index:end_index+1])
                total_pwr = vdd_pwr + vss_pwr

                event_energy = np.trapz(total_pwr, time_vec[start_index:end_index+1])
                batch_energies[j]+=event_energy

                # Calculate Latency (Defined as first input to last output)
                start_dynamic = start_index

                # Get Output Signal
                output_signal = get_signal(f"layer_{layer_num+1}_{x_id}_{y_id}_{split_r}_out", sim_obj, simulator='hspice')
                subset_output = output_signal[start_index:end_index]

                # Find Voltage Swing 
                initial_voltage = output_signal[start_index]
                end_voltage = output_signal[end_index-2]
                voltage_swing = end_voltage-initial_voltage
                threshold = 0.9

                threshold_voltage = initial_voltage + (voltage_swing * threshold)

                if voltage_swing >= 0:
                    crossing_index = np.where(subset_output >= threshold_voltage)[0]
                else:
                    crossing_index = np.where(subset_output <= threshold_voltage)[0]
                #print(crossing_index)

                p = len(crossing_index) - 1

                while p > 0 and crossing_index[p] - crossing_index[p - 1] == 1:
                    p -= 1
                crossing_index = crossing_index[p]

                # Interpolation between i-1 and i
                subset_time = time_vec[start_index:end_index]
                
                if crossing_index == 0:
                    # Can't interpolate before first point, fallback to index 0
                    crossing_time = subset_time[crossing_index]
                else:
                    x0, y0 = subset_time[crossing_index-1], subset_output[crossing_index-1]
                    x1, y1 = subset_time[crossing_index], subset_output[crossing_index]
                    y_thresh = threshold_voltage

                    if y1-y0 == 0:
                        crossing_time = subset_time[crossing_index]
                    else:
                        crossing_time = x0 + (y_thresh - y0) / (y1 - y0) * (x1 - x0)
                    
                event_latency = crossing_time - time_vec[start_dynamic]

                if j == 0:
                    previous_output = 0
                else:
                    previous_output = output_signal[start_index]

                current_output = output_signal[end_index]

                # Put everything together
                row = [f"layer_{layer_num+1}_{x_id}_{y_id}_{split_r}", event_latency, event_energy, current_output, previous_output]
                csv_writers[j].writerow(row)

        # Create PWL Files :)
        layer_crossbar_items = layers_keys[layer_num+1]

        # Note first zero is not used and is always just zero :)
        input_digital_neuron_signals = np.zeros((nodes[layer_num+1], testnum_per_batch+1))

        # Go through each cross bar
        for x_id, y_id, split_r in layer_crossbar_items:
            # Get Output Signal
            output_signal = get_signal(f"layer_{layer_num+1}_{x_id}_{y_id}_{split_r}_out", sim_obj, simulator='hspice')
            output_voltages = np.zeros(testnum_per_batch+1)

            # Get for each inference
            for m in range(testnum_per_batch):
                real_image_id = (image_num + m) + firstimage

                start_of_event = (m+1) * (tsampling * 10**-9)
                end_of_event = (m+2) * (tsampling *10**-9)

                # Convert to the indices that work with our SPICE simulation.
                bounds_mask = (time_vec >= start_of_event) & (time_vec <= end_of_event)
                valid_timestep_indices = np.where(bounds_mask)

                # Fix the end index to be plus one (so that we slightly overlap things.)
                start_index = valid_timestep_indices[0][0]  
                end_index_i = valid_timestep_indices[0][-1]+1
                end_index = np.min([end_index_i, len(time_vec)-1])      # Join the end of the index with the start of the new guy :)

                # Get Truncated Output
                truncated_output_signal = output_signal[start_index:end_index]

                output_voltages[m+1] = truncated_output_signal[-1]

            # Determine which neuron this goes to.
            # Horzontal partition does not matter, just vertical partition and the row ID.
            hor_cut, vert_cut = layer_cuts[layer_num+1]
            low_range_y = vert_cut[y_id-1]
            global_y_value = (low_range_y - 1) + split_r

            # Note: np zero-indexed, so subtract 1
            input_digital_neuron_signals[global_y_value-1, :] = input_digital_neuron_signals[global_y_value-1, :] + output_voltages

        if USE_QUANTIZATION:
            for time_guy in range(testnum_per_batch):
                input_digital_neuron_signals[:,time_guy+1] = dac(adc(input_digital_neuron_signals[:, time_guy+1], bits=DAC_BITS), bits=DAC_BITS)

        ## Neuron to run
        # Trained Inverse Sigmoid Neuron from IMAC-SIM
        layer_neuron_outputs = np.zeros((nodes[layer_num+1], testnum_per_batch))

        for time_guy in range(testnum_per_batch):
            layer_neuron_outputs[:,time_guy] = 0.8 / (1+np.exp(11*input_digital_neuron_signals[:, time_guy+1]))

        for time_guy in range(testnum_per_batch):
            layer_neuron_outputs[:,time_guy] = dac(adc(layer_neuron_outputs[:,time_guy], bits=DAC_BITS), bits=DAC_BITS)

        if layer_num == len(nodes)-2:
            for test_id in range(testnum_per_batch):
                for x in range(nodes[layer_num+1]):
                    out_list.append(float(layer_neuron_outputs[x, test_id]))

        layer_output_neurons = nodes[layer_num+1]

        # Calculate Everything
        for j in range(testnum_per_batch):
            real_image_id = (image_num + j) + firstimage

            for k in range(layer_output_neurons):
                neuron_name = f"Xsig_layer_{layer_num+1}_{k+1}"

                neuron_event_energy = 0
                event_latency = 0

                # Output and Previous Output
                if j == 0:
                    previous_output = 0
                else:
                    previous_output = layer_neuron_outputs[k, j-1]

                current_output = layer_neuron_outputs[k, j]

                row = [neuron_name, event_latency, neuron_event_energy, current_output, previous_output]
                csv_writers[j].writerow(row)

        # Write PWL Files
        for k in range(layer_output_neurons):
            # output vector definition
            output_voltages = np.zeros(testnum_per_batch+1)

            for m in range(testnum_per_batch):
                output_voltages[m+1] = layer_neuron_outputs[k, m]

            create_pwl_from_input_pulses(os.path.join(pwl_folder, f"neuron_{layer_num+1}_{k}_out.txt"), tsampling*10**-9, (0.1*tsampling*10**-9), output_voltages)

    for n in range(testnum_per_batch):
        pwr_list.append(batch_energies[n] * 10**12)
    
    for file_descrip in fd:
        file_descrip.close()

    for j in range(testnum_per_batch):
        # Compute Output Voltages and labels
        actual_label = np.argmax(label_list[nodes[len(nodes)-1]*j:nodes[len(nodes)-1]*(j+1)])
        print(f'Actual label: {actual_label}')
        err.append(int(0))
        list_max=max(out_list[nodes[len(nodes)-1]*j:nodes[len(nodes)-1]*(j+1)])

        out_voltages = [out_list[nodes[len(nodes)-1]*j:nodes[len(nodes)-1]*(j+1)]][0]
        print(f'Output voltages: {out_voltages}')

        for k in range (nodes[len(nodes)-1]):
            # Convert to Max
            if (out_list[nodes[len(nodes)-1]*j+k]==list_max):    # the neuron generating maximum output value represents the corrosponding class
                out_list[nodes[len(nodes)-1]*j+k]=1.0
            else:
                out_list[nodes[len(nodes)-1]*j+k]=0.0

            # Compute Error
            if (err[j+image_num]==0):
                if (out_list[nodes[len(nodes)-1]*j+k] != label_list[nodes[len(nodes)-1]*j+k]):
                    err[j+image_num]=1
        
        predicted_label = np.argmax(out_list[nodes[len(nodes)-1]*j:nodes[len(nodes)-1]*(j+1)])
        print(f'Predicted label: {predicted_label}')

        # Print correct or not
        if err[j+image_num]==1:
            print("Wrong prediction!")
        else:
            print("Correct prediction")

        energy_consumed = float(pwr_list[j+image_num])
        print("Energy consumption = %f pJ"%energy_consumed)
        print("sum error= %d"%(sum(err)))

        real_image_id = (image_num + j) + firstimage

        row = [real_image_id] + [actual_label] + [predicted_label] + [energy_consumed* 10**-12] + out_voltages
        writer.writerow(row)

    image_num = image_num + testnum_per_batch
    testimage = testimage + testnum_per_batch

file.close()
print(f"End-to-End CSV Created: {csv_name}")

#Area Calculation
xbar_num = sum(np.multiply(hpar, vpar))
xbar_area = W*L*xbar[0]*xbar[1]*xbar_num*1e12
switch_area = 0.56*xbar_num*(xbar[0]+xbar[1])
area = xbar_area + switch_area
print("Total area = "+str(area)+" \u00b5m^2")

print("Task completed!")
print("Total error= %d"%(sum(err)))
err_w=open("error.txt", "w")
err_w.write("Number of wrong recognitions in %d input image(s) = %d\n"% (image_num, sum(err)))
err_w.close()
data_r.close()
label_r.close()

print("error rate = %f"%(sum(err)/float(testnum)))   #calculate error rate
print("accuracy = %f%%"%(100-(sum(err)/float(testnum))*100))   #calculate accuracy

#measure the run time
end = time.time()
second=math.floor(end-start)
minute=math.floor(second/60)
hour=math.floor(minute/60)
tmin=minute-(60*hour)
tsec=second-(hour*3600)-(tmin*60)

print("Program Execution Time = %d hours %d minutes %d seconds"%(hour,tmin,tsec))

