import os
from pathlib import Path
import subprocess
import sys
import time

from datetime import datetime, timezone

CONFIG_DIR = "io_space_analysis/configs"
DATA_DIR = "../data"

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
    run_start_time = datetime.now(timezone.utc)
    run_info_name = "run_info_" + run_start_time.strftime("%Y_%d_%m-%H_%M_%S") + ".txt"
    run_info_file = Path(DATA_DIR) / run_info_name

    print(f"Writing run information to {run_info_file}")
    with open(run_info_file, "w") as f:
        f.write(f'''Start simulations at {run_start_time.strftime("%H:%M:%S")} UTC\n\n''')

    config_dir = Path(CONFIG_DIR)
    for config in os.listdir(config_dir):
        config_name = "io_space_analysis.configs" + str(Path(config).stem)
        config_start_time = datetime.now(timezone.utc)

        with open(run_info_file, "a") as f:
            f.write(f'''Processing configuration: "{config_name}" found in file: "{config}"\n''')
            f.write(f'''Began processing at {config_start_time.strftime("%H:%M:%S")}\n''')

        print('\n\n---------------------------------------')
        print(f"Running the following files: [{python_files}]")
        print(f'''With current configuration: "{config_name}" defined in file "{config}"''')

        run_python_files(python_files, '--config', config_name)

        config_end_time = datetime.now(timezone.utc)
        with open(run_info_file, "a") as f:
            f.write(f'''Finished processing at {config_end_time.strftime("%H:%M:%S")}\n''')
            f.write(f'''Total elapsed time: {config_start_time.second - config_end_time.second} seconds\n\n''')


