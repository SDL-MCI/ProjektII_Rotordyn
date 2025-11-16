import sys
from PyQt6.QtWidgets import QApplication
from gui.gui_functions import RotorDynApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RotorDynApp()
    window.show()
    sys.exit(app.exec())