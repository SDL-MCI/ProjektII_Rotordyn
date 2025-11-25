from data_processing_Jim import RotorDynEMA
import os
import numpy as np

# 1. Create with self           THIS IS ONLY FOR GENERATED DATA!!!!!!
rde = RotorDynEMA(
    fs=12800,   #1000 for sim
    T=2.0,      # for sim
    rot_speeds=[0],     # [500, 1000, 1500] for sim
    n_positions=6       # 20 for sim
)

"""
# 2. Run simulation             THIS IS ONLY FOR GENERATED DATA!!!!!!
rde.simulate_signals(
    n_impacts=3,
    impact_starts=[0.2, 0.6, 1.2]
)

"""
# Get Meas Data
base_dir = os.path.dirname(os.path.abspath(__file__))                   # Get the directory of the current script
file_path = os.path.join(base_dir, "..", "test", "rotor_acc_data.npz")  # Go one folder up and into "test"
file_path = os.path.abspath(file_path)                                  # Normalize the path (resolves "..")
data = np.load(file_path)

# 2. Set measured signals       THIS IS ONLY FOR Measured DATA!!!!!!
rde.set_measured_signals(
    F_all = data["F_all"],
    Y_all_y = data["Y_all_y"],
    Y_all_z = data["Y_all_z"]
)


# 3. Compute the FRFs
freq, H_y, H_z = rde.compute_frfs()
print("\Frequency axis:\n\n", freq.shape)
print("\n\n\n\n\n\nH_y:\n\n", H_y.shape)

# 4. Run EMA
df_list, df_multi = rde.run_ema_SIMO()
print("\n\n\n\n\n\nPrint dflist.head()\n\n",df_list.head())

# 5. Extract modes at one location
print("\n\n\n\n\n\nGET MODES AT get_modes_at(position=0, direction=Y, rpm=500)\n\n")
print(rde.get_modes_at(rpm=0, direction="Y"))


# 6. Plot stability diagramm
for rpm in [0]:   # Later rpm and dir form gui
    rde.plot_stability_diagram(rpm=rpm, direction="Y")
    rde.plot_stability_diagram(rpm=rpm, direction="Z")

"""
# 7. Plot Modeshapes
for rpm in [0]:
    for direction in ["Y", "Z"]:
        modes_df = rde.get_modes_at(rpm=rpm, direction=direction)
        n_modes = len(modes_df)
        for mode_index in range(n_modes):
            rde.plot_mode_shape(mode_index=mode_index, rpm=rpm, direction=direction)
"""