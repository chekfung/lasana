import os
from pathlib import Path
import subprocess
import sys

CONFIG_DIR = "io_space_analysis/configs"

python_files = [
    "testbench_generation.py",             
    "circuit_dataset_creation.py", 
    "predict_dynamic_energy_ml_model.py",
    "predict_latency_ml_model.py",
    "predict_state_ml_model.py",
    "predict_spike_behavior_ml_model.py",
    "predict_static_energy_ml_model.py"
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

if __name__ == "__main__":
    cmds = []

    config_dir = Path(CONFIG_DIR)
    for config in os.listdir(config_dir):
        config_name = Path(config).stem
        print('\n\n---------------------------------------')
        print(f"Running the following files: [{python_files}]")
        print(f'''With current configuration: "{config_name}" defined in file "{config}"''')

        # run_python_files(python_files, '--config', config_name)