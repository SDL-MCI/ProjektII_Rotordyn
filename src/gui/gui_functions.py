import os
from PyQt6.QtWidgets import QMainWindow
from .rotorDyn import Ui_MainWindow
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6 import QtWidgets, QtCore

class RotorDynApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Insert Logo
        self.init_logos()


        # Button Events
        self.ui.checkAll_PushButton.clicked.connect(self.check_all_mps)
        self.ui.applySettings_pushButton.clicked.connect(self.apply_settings)
        self.ui.startMeasurement_pushButton.clicked.connect(self.start_measurement)


        # config
        self.configuration = {}

        # initial meas order
        self.update_meas_order()

    # insert logos
    def init_logos(self):

        svg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"logo_sdl.svg")

        logo_widgets = [
            self.ui.logoConf_widget,
            self.ui.logoMeas_widget,
            self.ui.logoRes_widget,
        ]

        for container in logo_widgets:
            svg = QSvgWidget(svg_path, parent=container)
            layout = QtWidgets.QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(svg, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

    # read combo box
    def _rpm_from_combo(self, combo):
        text = combo.currentText().strip()

        # "-" no entry
        if text == "-" or text == "":
            return None

        # get text withot Hz
        num_part = text.split()[0]

        try:
            return int(float(num_part))
        except ValueError:
            # ignore errors
            print("Invalid RPM in combo:", text)
            return None


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
        # RPM out of combo Box
        rpms = []
        for combo in [self.ui.rpm1_comboBox, self.ui.rpm2_comboBox, self.ui.rpm3_comboBox]:
            val = self._rpm_from_combo(combo)
            if val is not None:
                rpms.append(val)

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
