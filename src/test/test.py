import nidaqmx
from nidaqmx.system import System

# List all connected NI devices
system = System.local()
devices = system.devices

if len(devices) == 0:
    print("⚠️  No NI DAQ devices detected.")
else:
    print("✅ Detected NI DAQ device(s):")
    for device in devices:
        print(f" - {device.name}: {device.product_type}")
