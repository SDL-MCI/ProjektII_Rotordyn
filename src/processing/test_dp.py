from data_processing_Jim import RotorDynEMA

# 1. Create with self           THIS IS ONLY FOR GENERATED DATA!!!!!!
rde = RotorDynEMA(
    fs=1000,
    T=2.0,
    rot_speeds=[500, 1000, 1500],
    n_positions=20
)

# 2. Run simulation             THIS IS ONLY FOR GENERATED DATA!!!!!!
rde.simulate_signals(
    n_impacts=3,
    impact_starts=[0.2, 0.6, 1.2]
)

"""
# 2. Set measured signals       THIS IS ONLY FOR Measured DATA!!!!!!
rde.set_measured_signals(
    t = ,
    F_all = ,
    Y_all_y = ,
    Y_all_z = 
)
"""

# 3. Compute the FRFs
freq, H_y, H_z = rde.compute_frfs()
print("\Frequency axis:\n\n", freq.shape)
print("\n\n\n\n\n\nH_y:\n\n", H_y.shape)

# 4. Run EMA
df_list, df_multi = rde.run_ema_SIMO()
print("\n\n\n\n\n\nPrint dflist.head()\n\n",df_list.head())

# 5. Extract modes at one location
print("\n\n\n\n\n\nGET MODES AT get_modes_at(position=0, direction=Y, rpm=500)\n\n")
print(rde.get_modes_at(rpm=500, direction="Y"))

"""
# 6. Plot stability diagramm
for rpm in [500, 1000, 1500]:   # Later rpm and dir form gui
    rde.plot_stability_diagram(rpm=rpm, direction="Y")
    rde.plot_stability_diagram(rpm=rpm, direction="Z")
"""

# 7. Plot Modeshapes
for rpm in [500, 1000, 1500]:
    for direction in ["Y", "Z"]:
        modes_df = rde.get_modes_at(rpm=rpm, direction=direction)
        n_modes = len(modes_df)
        for mode_index in range(n_modes):
            rde.plot_mode_shape(mode_index=mode_index, rpm=rpm, direction=direction)