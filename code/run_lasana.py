import subprocess
import sys
import os
import shutil

'''
Headless script for artifact analysis for MLCAD 2025.
Author: Jason Ho
'''

# List of Python file names you want to execute
python_files = [
    #"testbench_generation.py",             # Script disabled as requires spectre, hspice installation which is not available in Code Ocean
    #"circuit_dataset_creation.py",         # Script disabled as requires spectre, hspice installation which is not available in Code Ocean
    "predict_dynamic_energy_ml_model.py",
    "predict_latency_ml_model.py",
    "predict_state_ml_model.py",
    "predict_spike_behavior_ml_model.py",
    "predict_static_energy_ml_model.py"
]  

python_files_pcm_crossbar = [
    #"testbench_generation.py",             # Script disabled as requires spectre, hspice installation which is not available in Code Ocean
    #"circuit_dataset_creation.py",         # Script disabled as requires spectre, hspice installation which is not available in Code Ocean
    "pcm_crossbar_predict_dynamic_energy_ml_model.py",
    "pcm_crossbar_predict_latency_ml_model.py",
    "pcm_crossbar_predict_behavior_ml_model.py",
    "pcm_crossbar_predict_static_energy_ml_model.py"
]  

def run_python_files(files, option=None, arg=None):
    for file in files:
        try:
            print('\n---------------------------------------')
            print(f"Running {file} from run_lasana.py")

            # Build the base command
            cmd = ['python', file]
            if option and arg:
                cmd.extend([option, arg])
            elif option:
                cmd.extend([option])

            # Run the Python file as a subprocess and print output to stdout in real-time
            process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)

            # Wait for the process to complete
            process.wait()

            if process.returncode == 0:
                print(f"Successfully executed {file}")
            else:
                print(f"Error occurred while running {file}")

        except Exception as e:
            print(f"Error occurred while running {file}: {e}")
            
def copy_folder_contents(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        dst_path = os.path.join(dst_folder, item)

        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)

    print(f"Copied contents from {src_folder} to {dst_folder}")

if __name__ == "__main__":
    # ------------------------------------ #

    ## 1. Run through LASANA model generation with a pre-evaluated circuit dataset since Cadence, Synopsys tools unavailable in Code Ocean
    ## If one wants to generate the dataset from scratch, the scripts have been commented out above and can be run from this script as well.
    ## This is testbench_generation.py and circuit_dataset_creation.py
    ## Note: All of the ML models have been fixed to a certain seed for reproducability sake, but this can be disabled with the "DETERMINISTIC"
    ##       flag in each of the respective config files.
    ## Generates Table I, Table II, Figure 6 (a,b,c,d), Figure 7 (a,b,c,d) 
    # Run LASANA for spiking dataset

    CONFIG_SPIKING_NEURON = 'config_spiking_neuron'
    print('\n\n---------------------------------------')
    print(f"Running the following files for Spiking Neuron: [{python_files}]")
    run_python_files(python_files, '--config', CONFIG_SPIKING_NEURON)

    # Run LASANA for PCM crossbar with gain of 10
    CONFIG_DIFF_10 = 'config_pcm_crossbar_gain_10'
    print('\n\n---------------------------------------')
    print(f"Running the following files for PCM Crossbar Gain 10: [{python_files_pcm_crossbar}]")
    run_python_files(python_files_pcm_crossbar, '--config', CONFIG_DIFF_10)

    # Run LASANA for PCM crossbar with gain of 30
    CONFIG_DIFF_30 = 'config_pcm_crossbar_gain_30'
    print('\n\n---------------------------------------')
    print(f"Running the following files for PCM Crossbar Gain 30: [{python_files_pcm_crossbar}]")
    run_python_files(python_files_pcm_crossbar, '--config', CONFIG_DIFF_30)

    # Copy file contents of the ML models over
    copy_folder_contents('../data/spiking_neuron_run/ml_models', '../results/spiking_neuron_ml_models')
    copy_folder_contents('../data/pcm_crossbar_diff_10_run/ml_models', '../results/pcm_crossbar_diff_10_ml_models')
    copy_folder_contents('../data/pcm_crossbar_diff_30_run/ml_models', '../results/pcm_crossbar_diff_30_ml_models')

    
    
    # ------------------------------------ #

    ## 2. Recreate behavioral error error propagation experiment on 20k neuron layer
    ## Creates Table III, and Figure 8
    print(f"Running error propagation experiments for the LASANA Spiking Neuron")
    print('\n\n---------------------------------------')
    run_python_files(['ml_inference_wrapper_spiking_neuron.py'])                # First run predicted
    print('\n\n---------------------------------------')
    run_python_files(['ml_inference_wrapper_spiking_neuron.py'], '--oracle')    # After run oracle

    # Create table I and II and III from the paper and deposit in the results folder.
    print(f"Creating Table I, II, III from the paper")
    run_python_files(['create_table_i_ii_iii.py'])
    
    copy_folder_contents('../data/ml_inference_wrapper_intermediate_results', '../results/ml_inference_wrapper_intermediate_results')


    # ------------------------------------ #

    ## 3. Get timing / scaling information for LASANA spiking runtime
    ## Note: Since it is not possible to run the SPICE / SV-RNM tools as they are not available in Code Ocean, we just run our runtime analysis
    ##       on the spiking neuron. 
    ## Note: The script for running the same timing is available at ml_inference_wrapper_pcm_crossbar.py, but is not run here.
    print('\n\n---------------------------------------')
    print(f"Running Timing / Scaling experiments for the LASANA Spiking Neuron")
    run_python_files(['ml_inference_wrapper_spiking_neuron_timing.py'])

    # ------------------------------------ #

    ## 4. Run LASANA Spiking MNIST, and compare against golden results
    ## Recreates partial results that are found in Section V.E (MNIST and Spiking MNIST Case Study) in the paper.
    ## Since it is not possible to run the respective SPICE models for the two experiments, the code has been provided in the following scripts: 
    ## Spiking Neuron: spice_mnist.py
    ## PCM Crossbar: spice_crossbar_mnist.py
    ## Instead, due to space limitations, the first 500 inferences of each of the two test datasets have been provided in /data/crossbar_mnist_golden_results for
    ## the spiking neuron and /data/spiking_mnist_golden_results for the crossbar array.

    # Run LASANA Spiking MNIST
    os.chdir("lasana_spiking_mnist")
    print('\n\n---------------------------------------')
    print(f"Running first 500 test images of LASANA Spiking MNIST")
    run_python_files(['run_mnist_lasana.py'])     

    # Run LASANA Crossbar MNIST
    os.chdir('..')
    os.chdir("lasana_crossbar_mnist")
    print('\n\n---------------------------------------')
    print(f"Running first 500 test images of LASANA Crossbar MNIST")
    run_python_files(['imac_mnist.py'])      
    os.chdir('..')

    # Run Spiking MNIST (Commented out due to lack of CAD tools)
    # run_python_files(['spice_mnist.py'])

    # Run Crossbar MNIST (Commented out due to lack of CAD tools)
    # run_python_files(['spice_crossbar_mnist.py'])

    # Run Comparison Scripts

    # Run Spiking MNIST comparison scripts on first 500 inferences
    print('\n\n---------------------------------------')
    print("Running Spiking MNIST Comparison Script")    
    run_python_files(['spice_versus_lasana_spiking_comparison.py'])

    # Run Crossbar MNIST comparison scripts on first 500 inferences
    print('\n\n---------------------------------------')
    print("Running Crossbar MNIST Comparison Script")
    run_python_files(['spice_versus_lasana_imac_comparison.py'])

    # Move all the figures into a figures folder at the end
    source_folder = '../results'
    destination_folder = '../results/figures'
    os.makedirs(destination_folder, exist_ok=True) 

    for file in os.listdir(source_folder):
        if file.endswith('.png'):
            shutil.move(os.path.join(source_folder, file), os.path.join(destination_folder, file))

    # ------------------------------------ #
        
    print("All Runs Finished for LASANA!")
    print("Exiting gracefully :)")

