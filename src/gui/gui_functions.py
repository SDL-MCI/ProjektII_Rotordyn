from PyQt6.QtWidgets import QMainWindow
from .rotorDyn import Ui_MainWindow

class RotorDynApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Button Events
        self.ui.checkAll_PushButton.clicked.connect(self.check_all_mps)
        self.ui.applySettings_pushButton.clicked.connect(self.apply_settings)
        self.ui.startMeasurement_pushButton.clicked.connect(self.start_measurement)


        # config
        self.configuration = {}

        # initial meas order
        self.update_meas_order()



    def update_meas_order(self, text: str | None = None):

        if text is not None:
            self.ui.measOrder_label.setText(text)
            return

        # Case for configurations
        cfg = self.configuration or {}
        rpms_ok = bool(cfg.get("RPM"))                      # min 1 rpm
        mps_ok = any(cfg.get("measPoints", []))             # min 1 MP

        if rpms_ok or mps_ok:
            self.ui.measOrder_label.setText("Please continue with measurements!")
        else:
            self.ui.measOrder_label.setText("Please enter configs!")


    # Button to check all Measurement Points
    def check_all_mps(self):
        for box in [
            self.ui.mp1_checkBox, self.ui.mp2_checkBox, self.ui.mp3_checkBox, self.ui.mp4_checkBox,
            self.ui.mp5_checkBox, self.ui.mp6_checkBox, self.ui.mp7_checkBox, self.ui.mp8_checkBox,
        ]:
            box.setChecked(True)

    # Button to apply settings
    def apply_settings(self):
        # RPM out of gui (empty = ignore)
        rpms = []
        for edit in [self.ui.rpm1_TextEdit, self.ui.rpm2_TextEdit, self.ui.rpm3_TextEdit]:
            txt = edit.toPlainText().strip()
            if txt:
                try:
                    rpms.append(int(float(txt)))
                except ValueError:
                    print("Error with RPM input")
                    pass

        # Safe MP as Array of 0, 1 if true
        mp_boxes = [
            self.ui.mp1_checkBox, self.ui.mp2_checkBox, self.ui.mp3_checkBox, self.ui.mp4_checkBox,
            self.ui.mp5_checkBox, self.ui.mp6_checkBox, self.ui.mp7_checkBox, self.ui.mp8_checkBox,
        ]
        meas_points = [1 if b.isChecked() else 0 for b in mp_boxes]

        # SpinBox Values
        n_of_exc = int(self.ui.nrofe_SpinBox.value())
        settl_time = float(self.ui.settlingtime_DoubleSpinBox.value())

        # Building a dictionary for config settings
        self.configuration = {
            "RPM": rpms,                # ex. [0, 1500, 3000]
            "measPoints": meas_points,  # ex. [1,0,1,0,0,1,0,0]
            "NOfExc": n_of_exc,         # int
            "SettlTime": settl_time,    # float
        }

        # Testing in terminal
        print("Configuration:", self.configuration)

        # Show im Measurement Tab
        self.ui.arrayOfRPM_label.setText(str(rpms))
        self.ui.arrayOfMP_label.setText(str(meas_points))

        # Show in Results
        rpm_labels = [self.ui.rpm1_label, self.ui.rpm2_label, self.ui.rpm3_label]

        for i, label in enumerate(rpm_labels):
            if i < len(rpms) and rpms[i] != "" and rpms[i] is not None:
                label.setText(str(rpms[i]))
            else:
                label.setText("-")

        # update meas order text
        self.update_meas_order()

    # start measurement button
    def start_measurement(self):
        self.update_meas_order("Please wait for full version")
