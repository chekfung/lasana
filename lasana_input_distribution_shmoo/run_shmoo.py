from datetime import datetime
from pathlib import Path
import csv
import subprocess
import sys


# -----------------------------
# Shmoo settings

POISSON_LAMBDAS = [0.5, 2.0, 8.0]

NORMAL_STANDARD_DEVIATIONS = [0.5, 1.0, 2.0]
UNIFORM_HALF_WIDTHS = [1.0, 2.0, 4.0]

NUMBER_OF_RUNS = 100
NUMBER_OF_PROCESSES = 20
DELAY_RATIO = 0.3


# -----------------------------
# File locations

SHMOO_DIRECTORY = Path(__file__).resolve().parent
LASANA_DIRECTORY = SHMOO_DIRECTORY.parent
LASANA_CODE_DIRECTORY = LASANA_DIRECTORY / "code"

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
output_directory = SHMOO_DIRECTORY / f"output_{timestamp}"
config_directory = output_directory / "configs"
run_directory = output_directory / "runs"
log_directory = output_directory / "logs"

for directory in [config_directory, run_directory, log_directory, output_directory / "tables", output_directory / "plots"]:
    directory.mkdir(parents=True)


# -----------------------------
# Config generation

def number_label(value):
    """Make a parameter value safe to use inside a SPICE run name."""
    return str(float(value)).replace(".", "p")


def create_config(run_name, input_generation, poisson_lambda, weight_family, weight_scale):
    """Write one external LASANA config for one point in the shmoo."""
    if input_generation == "poisson":
        spike_map_config = f"""
SPIKE_MAP_GENERATION_MODE = "direct_count"
DELAY_RATIO = {DELAY_RATIO}
NUM_FIRINGS_RANDOM_SEED = 43
NUM_FIRINGS_LAMBDA = {poisson_lambda}

def sample_num_firings(rng):
    return int(rng.poisson(NUM_FIRINGS_LAMBDA))
"""
    else:
        spike_map_config = f"""
SPIKE_MAP_GENERATION_MODE = "legacy"
DELAY_RATIO = {DELAY_RATIO}
"""

    if weight_family == "normal":
        weight_config = f"""
USE_UNIFORM_DISTRIBUTION = False
NORMAL_WEIGHT_STANDARD_DEVIATION = {weight_scale}

def sample_event_weight(rng, num_firings):
    weights = rng.normal(
        loc=NORMAL_WEIGHT_MEAN,
        scale=NORMAL_WEIGHT_STANDARD_DEVIATION,
        size=num_firings,
    )
    return weights.sum()
"""
    else:
        weight_config = f"""
USE_UNIFORM_DISTRIBUTION = True
WEIGHT_LOW = {-weight_scale}
WEIGHT_HIGH = {weight_scale}

def sample_event_weight(rng, num_firings):
    return rng.uniform(
        WEIGHT_LOW * num_firings,
        WEIGHT_HIGH * num_firings,
    )
"""

    config_text = f"""import random

import numpy as np

from config_spiking_neuron import *


RUN_NAME = {run_name!r}
RUN_OUTPUT_ROOT = {str(run_directory.resolve())!r}
NUMBER_OF_RUNS = {NUMBER_OF_RUNS}
NUM_PROCESSES = {NUMBER_OF_PROCESSES}

KNOB_PARAMS = [
    ("V_sf", 0.5, 0.5, "c"),
    ("V_adap", 0.5, 0.5, "c"),
    ("V_leak", 0.57, 0.57, "c"),
    ("V_rtr", 0.5, 0.5, "c"),
]

{spike_map_config}
WEIGHT_RANDOM_SEED = 42
{weight_config}

random.seed(55)
np.random.seed(55)
"""

    config_path = config_directory / f"{run_name}.py"
    config_path.write_text(config_text)
    return config_path


# -----------------------------
# Run LASANA

def run_lasana_script(script_name, config_path, log_name):
    """Run one LASANA stage and redirect its output to the shmoo log directory."""
    command = [
        sys.executable,
        str(SHMOO_DIRECTORY / "run_lasana_headless.py"),
        str(LASANA_CODE_DIRECTORY / script_name),
        "--config",
        str(config_path),
    ]

    with (log_directory / log_name).open("w") as log_file:
        subprocess.run(
            command,
            cwd=LASANA_CODE_DIRECTORY,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )


shmoo_points = []
weight_sweeps = [
    ("normal", "sigma", NORMAL_STANDARD_DEVIATIONS),
    ("uniform", "half_width", UNIFORM_HALF_WIDTHS),
]

for poisson_lambda in POISSON_LAMBDAS:
    for weight_family, weight_scale_name, weight_scales in weight_sweeps:
        for weight_scale in weight_scales:
            run_name = (
                f"poisson_lam_{number_label(poisson_lambda)}_"
                f"{weight_family}_{weight_scale_name}_{number_label(weight_scale)}"
            )

            print(f"Running {run_name}")
            config_path = create_config(
                run_name,
                "poisson",
                poisson_lambda,
                weight_family,
                weight_scale,
            )

            shmoo_points.append({
                "run_name": run_name,
                "input_generation": "poisson",
                "weight_family": weight_family,
                "poisson_lambda": poisson_lambda,
                "weight_scale_name": weight_scale_name,
                "weight_scale": weight_scale,
                "config_path": str(config_path.resolve()),
                "run_directory": str((run_directory / run_name).resolve()),
            })

            run_lasana_script(
                "testbench_generation.py",
                config_path,
                f"{run_name}_testbench.log",
            )
            run_lasana_script(
                "circuit_dataset_creation.py",
                config_path,
                f"{run_name}_dataset.log",
            )


# -----------------------------
# Original LASANA comparison point

# This point changes only the spike-map generator. It retains the fixed knobs,
# delay ratio, historical aggregate-uniform [-2N, 2N] weights, and run count.
legacy_run_name = "legacy_uniform_half_width_2p0"
print(f"Running {legacy_run_name}")

legacy_config_path = create_config(
    legacy_run_name,
    "legacy",
    None,
    "uniform",
    2.0,
)

shmoo_points.append({
    "run_name": legacy_run_name,
    "input_generation": "legacy",
    "weight_family": "uniform",
    "poisson_lambda": "",
    "weight_scale_name": "half_width",
    "weight_scale": 2.0,
    "config_path": str(legacy_config_path.resolve()),
    "run_directory": str((run_directory / legacy_run_name).resolve()),
})

run_lasana_script(
    "testbench_generation.py",
    legacy_config_path,
    f"{legacy_run_name}_testbench.log",
)
run_lasana_script(
    "circuit_dataset_creation.py",
    legacy_config_path,
    f"{legacy_run_name}_dataset.log",
)


# -----------------------------
# Save the grid and create the plots

shmoo_points_path = output_directory / "shmoo_points.csv"
with shmoo_points_path.open("w", newline="") as shmoo_points_file:
    writer = csv.DictWriter(shmoo_points_file, fieldnames=shmoo_points[0].keys())
    writer.writeheader()
    writer.writerows(shmoo_points)

subprocess.run(
    [
        sys.executable,
        str(SHMOO_DIRECTORY / "analyze_shmoo.py"),
        "--output-dir",
        str(output_directory),
    ],
    cwd=LASANA_DIRECTORY,
    check=True,
)

print(f"Completed shmoo: {output_directory}")
