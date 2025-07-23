NUMBER_OF_INPUTS = 4

weight_net_p = "Rwpos{}_1"
weight_net_n = 'Rwneg{}_1'

r_low = 'other'
r_high = 'other'

WEIGHT_NET_NAMES_TO_CHANGE = {}
KNOB_PARAMS = []

for i in range(1, NUMBER_OF_INPUTS+1):
    weight_name = f"weight_{i}"
    KNOB_PARAMS.append((weight_name, r_low, r_high, 'b'))
    WEIGHT_NET_NAMES_TO_CHANGE[weight_name] = (weight_net_p.format(i), weight_net_n.format(i))

# Assign bias
KNOB_PARAMS.append(("bias_1", r_low, r_high, 'b'))
WEIGHT_NET_NAMES_TO_CHANGE["bias_1"] = ("Rbpos1", "Rbneg1")
