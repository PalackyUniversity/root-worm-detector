"""Undoable classification overrides without changing model probabilities."""
from copy import deepcopy

from PySide6.QtGui import QUndoCommand

from config.strings import Strings
from logic.image_logic import ImageLogic


class MarkContoursCommand(QUndoCommand):
    def __init__(self, data, indices, nice):
        super().__init__(Strings.MARK_NICE if nice else Strings.MARK_NOT_NICE)
        self.data = data
        self.before = deepcopy(data.get('measurements', []))
        self.after = deepcopy(self.before)
        while len(self.after) < len(data['contours']):
            self.after.append({})
        for index in set(indices):
            record = self.after[index]
            if 'model_nice' not in record:
                record['model_nice'] = record.get('nice')
            record['nice'] = record['nice_override'] = nice

    def _restore(self, records):
        self.data['measurements'] = deepcopy(records)
        ImageLogic.save_image_data(self.data)

    def redo(self):
        self._restore(self.after)

    def undo(self):
        self._restore(self.before)
