"""Run one LASANA script with a non-interactive matplotlib backend."""

import runpy
from pathlib import Path
import sys

import matplotlib


# The first argument is the LASANA script being wrapped. Leave the remaining
# arguments in place so that script still sees its normal --config CLI.
lasana_script = sys.argv[1]
sys.argv = sys.argv[1:]

# LASANA scripts import helpers from their own code directory.
sys.path.insert(0, str(Path(lasana_script).resolve().parent))

# Several historical scripts explicitly request TkAgg. Select the file-only
# backend first, then prevent those later calls from reopening a GUI backend.
matplotlib.use("Agg")
matplotlib.use = lambda *args, **kwargs: None

# Execute the wrapped file exactly as if it had been launched directly.
runpy.run_path(lasana_script, run_name="__main__")
