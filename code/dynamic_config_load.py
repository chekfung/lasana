import importlib

'''
The point of this file is to be able to dynamically load config files into the LASANA pipeline
so that it is easy to run everything headlessly without having to change files and such.

Therefore, we load the config files at runtime which contain all of the hyperparameters required to 
run things :)
'''

def inject_config(config_name, gbls):
    config_module = importlib.import_module(f"{config_name}")
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
