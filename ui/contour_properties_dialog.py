"""Read-only model probabilities and classification for selected contours."""
import math
from numbers import Real

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox,
                               QHeaderView, QLabel, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)

from config.strings import Strings


class ContourPropertiesDialog(QDialog):
    def __init__(self, data, indices, parent=None):
        super().__init__(parent)
        self.setWindowTitle(Strings.CONTOUR_PROPERTIES_TITLE)
        self.resize(min(1100, 370 + 180 * len(indices)), 430)
        layout = QVBoxLayout(self)
        labels = [Strings.DETECTOR_CONFIDENCE, Strings.MODEL_NICE_PROBABILITY,
                  Strings.MODEL_NOT_NICE_PROBABILITY, Strings.NICE_THRESHOLD_PROPERTY,
                  Strings.CURRENT_CLASSIFICATION, Strings.MODEL_CLASSIFICATION,
                  Strings.MANUAL_OVERRIDE, Strings.REFINED_AREA]
        self.table = QTableWidget(len(labels), len(indices) + 1, self)
        self.table.setHorizontalHeaderLabels([Strings.PROPERTY] + [
            Strings.CONTOUR_NUMBER.format(number=index + 1) for index in indices])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        for row, label in enumerate(labels):
            self.table.setItem(row, 0, QTableWidgetItem(label))
        records, scores = data.get("measurements", []), data.get("scores", [])
        for column, index in enumerate(indices, 1):
            record = records[index] if index < len(records) else {}
            score = record.get("detector_score")
            if score is None and index < len(scores):
                score = scores[index]
            nice = record.get("nice_probability")
            complement = 1 - nice if self._valid_probability(nice) else None
            values = [self._probability(score), self._probability(nice),
                      self._probability(complement),
                      self._probability(data.get("nice_threshold")),
                      self._classification(record.get("nice")),
                      self._classification(record.get("model_nice", record.get("nice"))),
                      self._classification(record["nice_override"]) if "nice_override" in record else Strings.NO_OVERRIDE,
                      str(record["area_mm2"]) if record.get("area_mm2") is not None else Strings.NOT_AVAILABLE]
            for row, value in enumerate(values):
                item = QTableWidgetItem(value)
                if row < 3:
                    raw = (score, nice, complement)[row]
                    if self._valid_probability(raw):
                        item.setToolTip(str(raw))
                self.table.setItem(row, column, item)
        layout.addWidget(self.table)
        note = QLabel(Strings.PROPERTIES_NOTE)
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _valid_probability(value):
        return isinstance(value, Real) and math.isfinite(value) and 0 <= value <= 1

    @classmethod
    def _probability(cls, value):
        return f"{value:.2%}" if cls._valid_probability(value) else Strings.NOT_AVAILABLE

    @staticmethod
    def _classification(value):
        return Strings.CLASS_NICE if value is True else Strings.CLASS_NOT_NICE if value is False else Strings.CLASS_UNKNOWN
