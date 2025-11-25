import numpy as np
import pandas as pd
from sdypy import EMA, FRF
import matplotlib.pyplot as plt

"""
Current flow:

simulate_signals() / set_measured_signals()
    ->
compute_frfs()                # once, cached
    ->
run_ema_SIMO()                # once, cached (all rpm + both directions)
    ->
plot_stability_diagram(rpm, direction)
    ->
plot_mode_shape(mode_index, rpm, direction)
    ->
get_modes_at(rpm, direction)
"""


class RotorDynEMA:
    def __init__(self, fs, T, rot_speeds, n_positions):
        # config
        #self.frf_type = "receptance"
        self.frf_type = "accelerance"


        self.fs = fs
        self.T = T
        self.n_samples = int(T*fs)                  # for generated data only....
        self.rot_speeds = np.array(rot_speeds)
        self.n_positions = n_positions

        # raw/time-domain data
        self.t = np.linspace(0, T, self.n_samples)  # for generated data only....
        self.F_all = None
        self.Y_all_y = None
        self.Y_all_z = None

        # FRFs
        self.freq = None
        self.H_y = None      # shape (n_rpm, n_positions, n_freq)
        self.H_z = None

        # EMA results
        self.ema_results = None    # list of dicts for DataFrame
        self.df_list = None
        self.df_multi = None

        # Mode shapes / full SIMO models per direction & RPM
        # self.modeshapes['Y'][rpm] = dict(A, nat_freq, nat_xi, freq, model)
        self.modeshapes = {'Y': {}, 'Z': {}}


    # Simulation of signals
    def simulate_signals(self, n_impacts, impact_starts):

        # Drehzahlen
        rot_speeds = [500, 1000, 1500]  # rpm
        # Beispiel-Moden (gedämpfte Sinus)
        f_modes = [51, 109]
        z_modes = [0.02, 0.03]
        a_modes = [1, 1]
        noise_amp = 0.001


        F_all = np.zeros((self.n_positions, len(rot_speeds), self.n_samples))
        Y_all_y = np.zeros((self.n_positions, len(rot_speeds), self.n_samples))
        Y_all_z = np.zeros((self.n_positions, len(rot_speeds), self.n_samples))

        for i_pos in range(self.n_positions):
            for i_speed, speed in enumerate(rot_speeds):
                F = np.zeros(self.n_samples)
                for start in impact_starts:
                    impulse_width = 0.01  # s
                    impulse_samples = int(impulse_width * self.fs)
                    impulse_start_idx = int(self.fs * start)

                    impulse = np.hanning(impulse_samples)
                    impulse = impulse / np.sum(impulse)

                    if impulse_start_idx + impulse_samples < self.n_samples:
                        F[impulse_start_idx:impulse_start_idx + impulse_samples] = impulse
                    else:
                        F[impulse_start_idx:] = impulse[:self.n_samples - impulse_start_idx]

                F_all[i_pos, i_speed, :] = F

                # Output y-Richtung
                h_true_y = sum(a*np.exp(-z*2*np.pi*f*self.t)*np.sin(2*np.pi*f*self.t)
                            for f,z,a in zip(f_modes, z_modes, a_modes))
                Y_all_y[i_pos, i_speed, :] = np.convolve(F, h_true_y, mode='full')[:self.n_samples] + noise_amp*np.random.randn(self.n_samples)

                # Output z-Richtung
                h_true_z = sum(a*np.exp(-z*2*np.pi*f*self.t)*np.cos(2*np.pi*f*self.t)
                            for f,z,a in zip(f_modes, z_modes, a_modes))
                Y_all_z[i_pos, i_speed, :] = np.convolve(F, h_true_z, mode='full')[:self.n_samples] + noise_amp*np.random.randn(self.n_samples)


        
        self.F_all = F_all
        self.Y_all_y = Y_all_y
        self.Y_all_z = Y_all_z

    
    # using with real data
    def set_measured_signals(self, F_all, Y_all_y, Y_all_z):
        """
        Attach real measurement data.
        Shapes:
        F_all      : (n_positions, n_rpms, n_samples)
        Y_all_y/z  : (n_positions, n_rpms, n_samples)
        """
        self.F_all = F_all
        self.Y_all_y = Y_all_y
        self.Y_all_z = Y_all_z

    # FRF computation dor all rpms and all pos with checking if already calculated
    def compute_frfs(self):
        """
        Compute FRFs for all RPMs and positions if not already done.
        Returns:
            freq : 1D array, frequency axis (without 0 Hz)
            H_y  : shape (n_rpm, n_positions, n_freq)
            H_z  : shape (n_rpm, n_positions, n_freq)
        """
        
        # check if calculated to pretend long calculatiion time
        if self.H_y is not None and self.H_z is not None and self.freq is not None:
            return self.freq, self.H_y, self.H_z

        assert self.F_all is not None, "No signals set. Call simulate_signals() or set_measured_signals() first."

        fs = self.fs
        n_rpms = len(self.rot_speeds)
        n_positions = self.n_positions

        # Storage structure
        H_y_list = []
        H_z_list = []

        for rpm_idx in range(n_rpms):
            H_y_pos = []
            H_z_pos = []

            for pos in range(n_positions):
                F_input = self.F_all[pos, rpm_idx, :][np.newaxis, :]
                Yy = self.Y_all_y[pos, rpm_idx, :][np.newaxis, :]
                Yz = self.Y_all_z[pos, rpm_idx, :][np.newaxis, :]

                frf_y = FRF.FRF(fs, F_input, Yy)
                frf_z = FRF.FRF(fs, F_input, Yz)

                H1_y = frf_y.get_H1()   # (1, 1, n_freq)
                H1_z = frf_z.get_H1()
                freq = frf_y.get_f_axis()

                if self.freq is None:
                    # assume same frequency axis for all
                    nonzero_idx = freq > 0
                    self.freq = freq[nonzero_idx]

                H_y_pos.append(H1_y[0, 0, freq > 0])
                H_z_pos.append(H1_z[0, 0, freq > 0])

            H_y_list.append(np.vstack(H_y_pos))  # (n_positions, n_freq)
            H_z_list.append(np.vstack(H_z_pos))

        self.H_y = np.stack(H_y_list, axis=0)  # (n_rpm, n_positions, n_freq)
        self.H_z = np.stack(H_z_list, axis=0)

        return self.freq, self.H_y, self.H_z

    # Run SIMO-Ema on each (rpm, direction) -> SIMO One H Input MS and Poles output
    def run_ema_SIMO(
            self,
            lower: float = 10.0,
            upper: float = 200.0,
            pol_order_high: int = 20,
            #interactive: bool = False,
            overwrite: bool = False,
        ):
        """
        Run SIMO EMA once for all RPMs and both directions (Y & Z).

        - Uses FRF matrix of shape (n_positions, n_freq_band) per (rpm, direction).
        - Runs EMA.Model (LSCF + LSFD).
        - Stores modal constants A (mode shapes) and natural frequencies/damping:
              self.modeshapes[direction][rpm] = {
                  'A': A,                 # shape (n_modes, n_positions)
                  'nat_freq': nat_freq,   # natural frequencies [Hz]
                  'nat_xi': nat_xi,       # damping ratios
                  'freq': freq_band,      # frequency axis used
                  'model': model          # full SDyPy model
              }

        Also builds a summary DataFrame self.df_list / self.df_multi with
        one row per (RPM, Direction, ModeIndex).
        """

        # if already done and not overwriting, just return
        if (self.ema_results is not None) and (not overwrite):
            return self.df_list, self.df_multi

        freq, H_y, H_z = self.compute_frfs()

        # reset containers
        self.ema_results = []
        self.df_list = None
        self.df_multi = None
        self.modeshapes = {'Y': {}, 'Z': {}}

        # frequency band
        band = (freq >= lower) & (freq <= upper)
        freq_band = freq[band]

        for direction in ("Y", "Z"):
            if direction == "Y":
                H = H_y
            else:
                H = H_z

            for rpm_idx, rpm in enumerate(self.rot_speeds):

                # FRF matrix for this rpm & direction: (n_positions, n_freq_band)
                frf_matrix = H[rpm_idx, :, :][:, band]

                # Build SIMO EMA model
                model = EMA.Model(
                    frf_matrix,
                    freq_band,
                    lower=lower,
                    upper=upper,
                    pol_order_high=pol_order_high,
                    frf_type=self.frf_type,
                )

                # LSCF: poles for many model orders
                # Least Squares Complex Frequency-domain
                # Find poles (eigenvalues)
                # Output: natural frequencies, damping, poles for each model order
                model.get_poles()

                # take the highest model order
                highest_order_index = len(model.all_poles) - 1   # complex poles from highest order
                poles = model.all_poles[highest_order_index]

                # convert to freq & damping
                freq_p = np.abs(np.imag(poles)) / (2*np.pi)
                zeta_p = -np.real(poles) / np.abs(poles)

                # automatically select poles
                valid = ((freq_p > lower) & (freq_p < upper) & (zeta_p > 0) & (zeta_p < 0.2) & (np.real(poles) < 0))
                # keep only positive imaginary poles (1 per physical mode)
                positive = np.imag(poles) > 0
                valid = valid & positive
                selected_poles = poles[valid]
                valid_indices = np.where(valid)[0]  # important for pole_ind

                # check if poles is found: 
                if selected_poles.size == 0:# or valid_indices.all() == 1:
                    print(f"[WARNING] No poles found for rpm={rpm}, direction={direction}")
                    continue

                # store these for later reporting
                nat_freq = freq_p[valid]
                nat_xi   = zeta_p[valid]

                # overwrite poles to auto selected
                model.poles = selected_poles
                model.pole_ind = np.column_stack([
                    np.full(len(selected_poles), highest_order_index, dtype=int),
                    valid_indices.astype(int)
                ])

                """
                # Optional interactive pole selection (SDyPy GUI)
                if interactive:
                    model.select_poles()
                """

                # LSFD: modal constants & reconstructed FRFs
                # Least Squares Frequency Domain
                # Estimate residues (modal constants)
                # mode shapes and reconstructed FRFs
                # A has shape (n_modes, n_positions)
                # Output: mode shapes (A matrix), modal FRFs, good-quality modal model
                H_rec, A_raw = model.get_constants()


                # We want: rows = modes, columns = positions.
                if A_raw.shape[0] == self.n_positions:
                    # rows = positions, cols = modes -> transpose
                    A_modes = A_raw.T
                elif A_raw.shape[1] == self.n_positions:
                    # rows = modes, cols = positions -> already good
                    A_modes = A_raw

                
                # Store full SIMO model for later (mode shapes, stability chart, etc.)
                self.modeshapes[direction][rpm] = {
                    "A": A_modes,
                    "nat_freq": nat_freq,
                    "nat_xi": nat_xi,
                    "freq": freq_band,
                    "model": model,
                }

                # Also build summary rows for each mode (if available)
                if nat_freq is not None and nat_xi is not None:
                    for mode_idx, (fn, xi) in enumerate(zip(nat_freq, nat_xi)):
                        self.ema_results.append({
                            "RPM": rpm,
                            "Direction": direction,
                            "ModeIndex": mode_idx,
                            "Frequency_Hz": fn,
                            "Damping_zeta": xi,
                        })

        # build DataFrames
        if len(self.ema_results) > 0:
            self.df_list = pd.DataFrame(self.ema_results)
            self.df_multi = self.df_list.set_index(["Direction", "RPM", "ModeIndex"])

        return self.df_list, self.df_multi
    
    # get modes at (rpm, dir) for easier acces to modes
    def get_modes_at(self, rpm, direction):
        direction = direction.upper()
        if direction not in ("Y", "Z"):
            raise ValueError("direction must be 'Y' or 'Z'")

        if self.df_list is None:
            self.run_ema_SIMO()

        mask = (self.df_list["RPM"] == rpm) & (self.df_list["Direction"] == direction)
        return self.df_list.loc[mask].reset_index(drop=True)
    
    # Plot Stability diagramm using computed EMA results for one rpm and one direction
    def plot_stability_diagram(
            self,
            rpm: float,
            direction: str,
        ):
        """
        Plot stability diagram for one RPM & direction using SIMO EMA MODEL:
            - x-axis: frequency [Hz]
            - y-axis: model order index
            - points: poles extracted from SIMO EMA
            - orange line: FRF magnitude |H(f)| on a second y-axis (log scale)
        """

        direction = direction.upper()
        if direction not in ("Y", "Z"):
            raise ValueError("direction must be 'Y' or 'Z'")

        # Ensure SIMO EMA has been run
        if (self.modeshapes[direction].get(rpm, None) is None):
            self.run_ema_SIMO()

        if rpm not in self.modeshapes[direction]:
            raise ValueError(f"No SIMO EMA data for rpm={rpm} and direction={direction}.")

        data = self.modeshapes[direction][rpm]
        model = data["model"]
        freq_band = data["freq"]

        # poles from all model orders
        all_poles = model.all_poles
        n_orders = len(all_poles)

        fig, ax1 = plt.subplots()

        for order_idx, poles in enumerate(all_poles):
            freq_p = np.abs(np.imag(poles)) / (2 * np.pi)
            ax1.scatter(
                freq_p,
                np.full_like(freq_p, order_idx + 1, dtype=float),
                marker="o",
                facecolor='none',
                edgecolors='blue',
                s=20,
                alpha=0.7,
            )

        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("Model order")
        ax1.set_ylim(0.5, n_orders + 0.5)
        ax1.set_xlim(freq_band[0], freq_band[-1])
        ax1.grid(True)

        # averaged FRF magnitude in this band
        freq, H_y, H_z = self.compute_frfs()
        band = (freq >= freq_band[0]) & (freq <= freq_band[-1])
        if direction == "Y":
            frf_band = H_y[self.rot_speeds == rpm, :, :][0][:, band]
        else:
            frf_band = H_z[self.rot_speeds == rpm, :, :][0][:, band]
        avg_mag = np.mean(np.abs(frf_band), axis=0)

        ax2 = ax1.twinx()
        ax2.plot(freq_band, avg_mag, color="orange", linewidth=2)
        ax2.set_yscale("log")
        ax2.set_ylabel("MAgnitude |FRF|")

        fig.suptitle(f"Stability Diagram: rpm={rpm}, direction={direction}")
        plt.tight_layout()
        plt.show()

    # Plot Modes shape
    def plot_mode_shape(
            self,
            mode_index: int,
            rpm: float,
            direction: str = "Y",
            real_part: bool = True,
            normalize: bool = True,
            x_positions=None,
        ):
        """
        Plot a single mode shape from SIMO EMA.

        Parameters
        ----------
        mode_index : int
            Index of the mode (0-based) in the A matrix.
        rpm : float
            Rotational speed at which SIMO EMA was run.
        direction : {'Y', 'Z'}
            Measurement direction.
        real_part : bool
            If True -> plots real(A); if False -> imag(A).
        normalize : bool
            If True -> scales mode shape to max abs value = 1.
        x_positions : array-like or None
            Optional physical coordinates of the measurement points.
            If None, uses indices 0..n_positions-1.
        """

        direction = direction.upper()
        if direction not in ("Y", "Z"):
            raise ValueError("direction must be 'Y' or 'Z'")

        # Ensure SIMO EMA has been run and this rpm/direction exists
        if rpm not in self.modeshapes[direction]:
            # try to compute SIMO EMA if not yet done
            self.run_ema_SIMO()
            if rpm not in self.modeshapes[direction]:
                raise ValueError(f"No SIMO EMA data for rpm={rpm}, direction={direction}")

        data = self.modeshapes[direction][rpm]
        A = data["A"]

        if A is None:
            raise ValueError(f"No mode shapes stored for rpm={rpm}, direction={direction}")

        # A shape: (n_modes, n_positions)
        n_modes, n_locs = A.shape
        if n_locs != self.n_positions:
            raise ValueError(
                f"Inconsistent A shape: {A.shape} does not match n_positions={self.n_positions}"
            )

        if mode_index < 0 or mode_index >= n_modes:
            raise IndexError(f"mode_index={mode_index} out of range (0..{n_modes-1})")

        # x-axis coordinates
        if x_positions is None:
            x_positions = np.arange(self.n_positions)
            x_label = "Position index"
        else:
            x_positions = np.asarray(x_positions)
            if x_positions.size != self.n_positions:
                raise ValueError(f"len(x_positions) must equal n_positions ({self.n_positions})")
            x_label = "Position"

        # pick the requested mode
        mode_vec = A[mode_index, :]   # shape (n_positions,)

        if real_part:
            y_vals = np.real(mode_vec)
            y_label = "Mode shape amplitude (real)"
        else:
            y_vals = np.imag(mode_vec)
            y_label = "Mode shape amplitude (imag)"

        # optional normalization to |max| = 1
        if normalize:
            max_val = np.max(np.abs(y_vals))
            if max_val > 0:
                y_vals = y_vals / max_val

        # try to include natural frequency in the title if available
        nat_freq = data.get("nat_freq", None)
        if nat_freq is not None and len(nat_freq) > mode_index:
            title_freq = f", f ≈ {nat_freq[mode_index]:.1f} Hz"
        else:
            title_freq = ""

        plt.figure()
        plt.plot(x_positions, y_vals, marker="o")
        plt.axhline(0.0, color="k", linewidth=0.8)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"Mode {mode_index} at {rpm} rpm ({direction}-dir){title_freq}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
