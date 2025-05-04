from PySide6.QtWidgets import QDialog, QFormLayout, QCheckBox, QDialogButtonBox, QFileDialog
from datetime import datetime


class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.file_name = self.ask_user_for_path()

        self.setWindowTitle("Select Metrics to Export")
        self.setMinimumWidth(400)
        self.countCheck = QCheckBox("Contour Count")
        self.totalAreaCheck = QCheckBox("Total Contour Area")
        self.avgAreaCheck = QCheckBox("Average Contour Area (+ stdev, stderr, variance)")
        self.medianAreaCheck = QCheckBox("Median Contour Area (+ Q05, Q10, Q25, Q75, Q90, Q95)")
        self.descStatsCheck = QCheckBox("Other Descriptive Stats (min, max, range, skewness, kurtosis)")


        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QFormLayout(self)

        for cb in (self.countCheck, self.totalAreaCheck, self.avgAreaCheck, self.medianAreaCheck, self.descStatsCheck):
            cb.setChecked(True)
            layout.addRow(cb)

        layout.addWidget(self.buttons)

    def ask_user_for_path(self):
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self.parent,
            "Export",
            "export_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )

        if not file_path:
            return None

        if selected_filter.startswith("CSV") and not file_path.lower().endswith(".csv"):
            file_path += ".csv"
        elif selected_filter.startswith("Excel") and not file_path.lower().endswith(".xlsx"):
            file_path += ".xlsx"

        return file_path

    def run(self):
        if self.file_name is None:
            return False
        else:
            return self.exec() == QDialog.Accepted

    def get_selections(self):
        return {
            "count": self.countCheck.isChecked(),
            "total": self.totalAreaCheck.isChecked(),
            "avg": self.avgAreaCheck.isChecked(),
            "median": self.medianAreaCheck.isChecked(),
            "desc": self.descStatsCheck.isChecked()
        }

    def get_file_name(self):
        return self.file_name

