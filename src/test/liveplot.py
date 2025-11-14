import nidaqmx
from nidaqmx.constants import TerminalConfiguration
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# --- Sensor and DAQ settings ---
V_ZERO = 2.5
V_SENS = 1.8 / 225
CHANNEL = "Dev1/ai0"

# --- DAQ setup ---
with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan(
        CHANNEL,
        terminal_config=TerminalConfiguration.RSE,
        min_val=0.0,
        max_val=5.0
    )

    plt.ion()
    fig, ax = plt.subplots()
    x_data, y_data = [], []
    line, = ax.plot(x_data, y_data, 'b-')
    ax.set_xlabel("Sample")
    ax.set_ylabel("Acceleration [g]")
    ax.set_ylim(-20, 20)
    ax.grid(True)

    print("Starting live measurement (Ctrl+C to stop)...")
    try:
        count = 0
        while True:
            voltage = task.read()
            acceleration_g = (voltage - V_ZERO) / V_SENS
            print(f"Voltage: {voltage:6.3f} V | Acceleration: {acceleration_g:8.2f} g")

            x_data.append(count)
            y_data.append(acceleration_g)
            if len(x_data) > 200:  # keep last 200 samples visible
                x_data = x_data[-200:]
                y_data = y_data[-200:]

            line.set_xdata(x_data)
            line.set_ydata(y_data)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.05)
            count += 1

    except KeyboardInterrupt:
        print("\nMeasurement stopped by user.")
        plt.ioff()
        plt.show()
