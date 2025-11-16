import nidaqmx
from nidaqmx.constants import LineGrouping
import time

# Kanäle definieren
do_channel = "Dev1/port0/line0"
di_channel = "Dev1/port0/line1"

with nidaqmx.Task() as do_task, nidaqmx.Task() as di_task:
    # DO konfigurieren
    do_task.do_channels.add_do_chan(do_channel, line_grouping=LineGrouping.CHAN_PER_LINE)
    
    # DI konfigurieren
    di_task.di_channels.add_di_chan(di_channel, line_grouping=LineGrouping.CHAN_PER_LINE)
    
    while True:
        # High auf DO setzen
        do_task.write(True)
        time.sleep(0.5)
        di_val = di_task.read()
        print("DI liest:", di_val)
        
        # Low auf DO setzen
        do_task.write(False)
        time.sleep(0.5)
        di_val = di_task.read()
        print("DI liest:", di_val)
