import importlib
import argparse

def inject_config(config_name, gbls):
    config_module = importlib.import_module(f"{config_name}")
    for key in dir(config_module):
        if not key.startswith("__"):
            gbls[key] = getattr(config_module, key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Name of the config file (without .py)")
    args = parser.parse_args()

    inject_config(args.config, globals())

    # ✅ Now all your config variables (even those created via loops) are accessible:
    print("r_low:", r_low)
    print("KNOB_PARAMS:", KNOB_PARAMS)
    print("WEIGHT_NET_NAMES_TO_CHANGE:", WEIGHT_NET_NAMES_TO_CHANGE)

    print("Loop through and make sure something is happening")
    for meow in WEIGHT_NET_NAMES_TO_CHANGE:
        print(meow)
        print(WEIGHT_NET_NAMES_TO_CHANGE[meow])
