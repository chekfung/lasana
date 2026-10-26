import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys

'''
The point of this file is to be able to dynamically load config files into the LASANA pipeline
so that it is easy to run everything headlessly without having to change files and such.

Therefore, we load the config files at runtime which contain all of the hyperparameters required to 
run things :)
'''


def _load_config_from_path(config_path):
    # Resolve relative paths from the directory where LASANA was invoked.
    config_path = Path(config_path).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    # Use the resolved path to distinguish configs with the same filename.
    path_digest = hashlib.sha256(str(config_path).encode()).hexdigest()[:16]
    module_name = f"_lasana_config_{path_digest}"
    module_spec = importlib.util.spec_from_file_location(module_name, config_path)

    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Unable to create an import specification for config file: {config_path}")

    # Save the current import state so a failed config cannot leave partial state.
    config_module = importlib.util.module_from_spec(module_spec)
    previous_module = sys.modules.get(module_name)
    original_sys_path = list(sys.path)

    try:
        # Allow an external config to import other files from its own directory.
        sys.path.insert(0, str(config_path.parent))
        sys.modules[module_name] = config_module
        module_spec.loader.exec_module(config_module)

    except Exception as error:
        # Restore a previous module, or remove the incomplete module.
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module

        raise ImportError(f"Failed to load config file '{config_path}': {error}") from error

    finally:
        # Do not permanently change how the rest of LASANA resolves imports.
        sys.path[:] = original_sys_path

    return config_module


def inject_config(config_name, gbls):
    # A .py argument is a filesystem path. Other arguments retain the historical
    # module-name behavior, such as "config_spiking_neuron".
    if str(config_name).endswith(".py"):
        config_module = _load_config_from_path(config_name)
    else:
        config_module = importlib.import_module(config_name)

    # Copy public configuration values into the calling script's global scope.
    for key in dir(config_module):
        if not key.startswith("__"):
            gbls[key] = getattr(config_module, key)


# I have included an example usage case below that can be used to 
# add global hyperparameters from another file dynamically

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", required=True, help="Name of the config file (without .py)")
#     args = parser.parse_args()
#     inject_config(args.config, globals())

#     # Now all your config variables are available
#     print("r_low:", r_low)
#     print("KNOB_PARAMS:", KNOB_PARAMS)
#     print("WEIGHT_NET_NAMES_TO_CHANGE:", WEIGHT_NET_NAMES_TO_CHANGE)

#     print("Loop through and make sure something is happening")
#     for param in WEIGHT_NET_NAMES_TO_CHANGE:
#         print(param)
#         print(WEIGHT_NET_NAMES_TO_CHANGE[param])
