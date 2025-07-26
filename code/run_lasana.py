import subprocess
import sys

# TODO: Update this to run everything! :)

# List of Python file names you want to execute
python_files = [
    "testbench_generation.py",
    "circuit_dataset_creation.py",
    "predict_dynamic_energy_ml_model.py",
    "predict_latency_ml_model.py",
    "predict_state_ml_model.py",
    "predict_spike_behavior_ml_model.py",
    "predict_static_energy_ml_model.py"
]  

python_files_mac_unit = [
    "testbench_generation.py",
    "circuit_dataset_creation.py",
    "mac_unit_predict_dynamic_energy_ml_model.py",
    "mac_unit_predict_latency_ml_model.py",
    "mac_unit_predict_behavior_ml_model.py",
    "mac_unit_predict_static_energy_ml_model.py"
]  


def run_python_files(files):
    for file in files:
        try:
            print('\n\n---------------------------------------')
            print(f"Running {file}...")
            # Run the python file as a subprocess and print the output to stdout in real-time
            process = subprocess.Popen(['python', file, '--config', CONFIG], stdout=sys.stdout, stderr=sys.stderr, text=True)

            # Wait for the process to complete
            process.wait()

            if process.returncode == 0:
                print(f"Successfully executed {file}")
            else:
                print(f"Error occurred while running {file}")

        except Exception as e:
            print(f"Error occurred while running {file}: {e}")

if __name__ == "__main__":
    RUN_MAC_UNIT = False
    CONFIG = 'config_spiking_neuron'

    RUN_MAC_UNIT = True
    CONFIG = 'config_pcm_crossbar_gain_10'

    if RUN_MAC_UNIT:
        print(f"Running the following files: [{python_files_mac_unit}]")
        run_python_files(python_files_mac_unit)
    else:
        print(f"Running the following files: [{python_files}]")
        run_python_files(python_files)
        
    print("All Runs Finished for LASANA!")
    print("Exiting gracefully :)")

