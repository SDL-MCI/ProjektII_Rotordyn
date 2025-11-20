import numpy as np
import pandas as pd
from sdypy import EMA, FRF




# SIMULATING SIGNALS

fs = 1000
T = 10
n_samples = int(T*fs)
n_positions = 10
n_impacts = 3
impact_starts = [1, 3, 7]  # (s)
t = np.linspace(0, T, n_samples)

# Beispiel-Moden (gedämpfte Sinus)
f_modes = [51, 109]
z_modes = [0.02, 0.03]
a_modes = [1, 1]
noise_amp = 0.001

# Drehzahlen
rot_speeds = [500, 1000, 1500]  # rpm


F_all = np.zeros((n_positions, len(rot_speeds), n_samples))
Y_all_y = np.zeros((n_positions, len(rot_speeds), n_samples))
Y_all_z = np.zeros((n_positions, len(rot_speeds), n_samples))

for i_pos in range(n_positions):
    for i_speed, speed in enumerate(rot_speeds):
        F = np.zeros(n_samples)
        for start in impact_starts:
            impulse_width = 0.01  # s
            impulse_samples = int(impulse_width * fs)
            impulse_start_idx = int(fs * start)

            impulse = np.hanning(impulse_samples)
            impulse = impulse / np.sum(impulse)

            if impulse_start_idx + impulse_samples < n_samples:
                F[impulse_start_idx:impulse_start_idx + impulse_samples] = impulse
            else:
                F[impulse_start_idx:] = impulse[:n_samples - impulse_start_idx]

        F_all[i_pos, i_speed, :] = F

        # Output y-Richtung
        h_true_y = sum(a*np.exp(-z*2*np.pi*f*t)*np.sin(2*np.pi*f*t)
                       for f,z,a in zip(f_modes, z_modes, a_modes))
        Y_all_y[i_pos, i_speed, :] = np.convolve(F, h_true_y, mode='full')[:n_samples] + noise_amp*np.random.randn(n_samples)

        # Output z-Richtung
        h_true_z = sum(a*np.exp(-z*2*np.pi*f*t)*np.cos(2*np.pi*f*t)
                       for f,z,a in zip(f_modes, z_modes, a_modes))
        Y_all_z[i_pos, i_speed, :] = np.convolve(F, h_true_z, mode='full')[:n_samples] + noise_amp*np.random.randn(n_samples)



##########################################################################################

#MODAL ANALYSIS

#Inputs:
# F_all shape: (n_positions, n_rpms, n_samples)
# Y_all_y shape: (n_positions, n_rpms, n_samples)
# Y_all_z shape: (n_positions, n_rpms, n_samples)


results = []

for rpm_idx, rpm in enumerate(rot_speeds):
    for pos in range(n_positions):
        F_input = F_all[pos, rpm_idx, :][np.newaxis, :]

        # y-Richtung
        Y_output_y = Y_all_y[pos, rpm_idx, :][np.newaxis, :]
        frf_y = FRF.FRF(fs, F_input, Y_output_y)
        H_y = frf_y.get_H1()
        freq = frf_y.get_f_axis()

        # z-Richtung
        Y_output_z = Y_all_z[pos, rpm_idx, :][np.newaxis, :]
        frf_z = FRF.FRF(fs, F_input, Y_output_z)
        H_z = frf_z.get_H1()

        # Deleting 0 Hz
        freq = frf_y.get_f_axis()
        nonzero_idx = freq > 0
        H_y = H_y[:, :, nonzero_idx]
        freq = freq[nonzero_idx]
        H_z = H_z[:, :, nonzero_idx]



        # EMA Model für y-Richtung
        model_y = EMA.Model(np.array([H_y[0,0,:]]), freq, lower=10, upper=200, pol_order_high=20, frf_type='receptance')
        model_y.get_poles()
        poles_y = model_y.all_poles[-1]  # letzte Ordnung
        freq_y = np.abs(np.imag(poles_y))/(2*np.pi)
        zeta_y = -np.real(poles_y)/np.abs(poles_y)
        
        valid_y = (freq_y>10) & (freq_y<200) & (zeta_y>0) & (zeta_y<0.2) & (np.real(poles_y)<0)
        poles_y = poles_y[valid_y]
        freq_y = freq_y[valid_y]
        zeta_y = zeta_y[valid_y]
        
        # EMA Model für z-Richtung
        model_z = EMA.Model(np.array([H_z[0,0,:]]), freq, lower=10, upper=200, pol_order_high=20, frf_type='receptance')
        model_z.get_poles()
        poles_z = model_z.all_poles[-1]
        freq_z = np.abs(np.imag(poles_z))/(2*np.pi)
        zeta_z = -np.real(poles_z)/np.abs(poles_z)
        
        valid_z = (freq_z>10) & (freq_z<200) & (zeta_z>0) & (zeta_z<0.2) & (np.real(poles_z)<0)
        poles_z = poles_z[valid_z]
        freq_z = freq_z[valid_z]
        zeta_z = zeta_z[valid_z]
        
        # Ergebnisse speichern
        for f, z, p in zip(freq_y, zeta_y, poles_y):
            results.append({
                'RPM': rpm,
                'Position': pos,
                'Direction': 'Y',
                'Frequency_Hz': f,
                'Damping_zeta': z,
                'Pole_real': np.real(p),
                'Pole_imag': np.imag(p)
            })
        for f, z, p in zip(freq_z, zeta_z, poles_z):
            results.append({
                'RPM': rpm,
                'Position': pos,
                'Direction': 'Z',
                'Frequency_Hz': f,
                'Damping_zeta': z,
                'Pole_real': np.real(p),
                'Pole_imag': np.imag(p)
            })

# Alles in DataFrame
df_list = pd.DataFrame(results)

# Dataframe multiindex 
df_multi = df_list.set_index(['Position', 'Direction', 'RPM'])

print(df_multi.loc[(4, 'Y', 500)])



# CSV speichern
#df_list.to_csv("modal_analysis_results.csv", index=False)
