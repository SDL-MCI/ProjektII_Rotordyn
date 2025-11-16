import nidaqmx
import time
from nidaqmx.constants import Edge

# --- Konfiguration ---
counter_channel = "Dev1/ctr0"
signal_source = "/Dev1/PFI0"
slots = 1
measurement_interval = 1.0  # 1 Sekunde pro Messung

with nidaqmx.Task() as task:
    task.ci_channels.add_ci_count_edges_chan(counter_channel, edge=Edge.RISING)
    task.ci_channels.all.ci_count_edges_term = signal_source
    task.start()

    last_count = task.read()
    
    print("Starte Echtzeit-Drehzahlmessung... (STRG+C zum Beenden)")

    try:
        while True:
            time.sleep(measurement_interval)
            current_count = task.read()
            
            delta_count = current_count - last_count
            last_count = current_count
            
            freq_hz = delta_count / measurement_interval
            rpm = freq_hz * 60.0 / slots

            print(f"Drehzahl: {rpm:8.1f} U/min   ")

    except KeyboardInterrupt:
        print("\nMessung beendet.")
