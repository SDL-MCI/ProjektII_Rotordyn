import nidaqmx
from nidaqmx.constants import TerminalConfiguration
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# --- Sensor and DAQ settings ---
# DISYNET DA 1202-225g: 2.5 V ± 1.8 V output range (±225 g)
V_ZERO = 2.5        # nominal output at 0 g
V_SENS = 1.8 / 225  # V per g (sensitivity)
CHANNEL = "Dev1/ai0"  # adjust to your NI USB-6002 input channel

# --- DAQ setup ---
with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan(
        CHANNEL,
        terminal_config=TerminalConfiguration.RSE,  # Single-ended
        min_val=0.0,
        max_val=5.0
    )

    print("Starting live measurement (press Ctrl+C to stop)...")
    time.sleep(1)

    try:
        while True:
            voltage = task.read()  # read one sample
            acceleration_g = (voltage - V_ZERO) / V_SENS
            print(f"Voltage: {voltage:6.3f} V | Acceleration: {acceleration_g:8.2f} g")
            time.sleep(0.1)  # 10 samples per second

    except KeyboardInterrupt:
        print("\nMeasurement stopped by user.")
