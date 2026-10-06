from PySide6.QtGui import QUndoCommand
from copy import deepcopy
import numpy as np
from logic.image_logic import ImageLogic
from config.strings import Strings


class ReclassifyContoursCommand(QUndoCommand):
    def __init__(self, image_data, indices, nice):
        super().__init__(Strings.RESTORE_CLASSIFICATION if nice is None else
                         Strings.MARK_NICE if nice else Strings.MARK_NOT_NICE)
        ImageLogic.preserve_original_annotations(image_data)
        self.image_data = image_data
        self.previous = {index: deepcopy(image_data["measurements"][index]) for index in set(indices)}
        self.updated = deepcopy(self.previous)
        for record in self.updated.values():
            record.setdefault("model_nice", record.get("nice"))
            if nice is None:
                record["nice"] = record["model_nice"]
                record.pop("nice_override", None)
            else:
                record["nice_override"] = nice
                record["nice"] = nice

    def _apply(self, records):
        for index, record in records.items():
            self.image_data["measurements"][index] = deepcopy(record)
        ImageLogic.save_image_data(self.image_data)

    def redo(self):
        self._apply(self.updated)

    def undo(self):
        self._apply(self.previous)


class AddContourCommand(QUndoCommand):
    """QUndoCommand for adding a contour to an image"""

    def __init__(self, image_data, contour_points, description=None):
        """
        Initialize the command

        Args:
            image_data: The image data dictionary to modify
            contour_points: List of (x,y) points forming the contour
            description: Optional command description
        """
        super().__init__(description or "Add Contour")
        self.prepared = ImageLogic.prepare_manual_contour(image_data, contour_points)
        ImageLogic.preserve_original_annotations(image_data)
        self.image_data = image_data
        self.contour_points = contour_points.copy()  # Make a copy to ensure we keep original points
        self.contour_index = None

    def redo(self):
        """Execute the command: add contour to the image data"""
        self.contour_index = ImageLogic.add_contour(self.image_data, self.contour_points,
                                                  prepared=self.prepared)

    def undo(self):
        """Undo the command: remove the added contour"""
        if self.contour_index is not None and "contours" in self.image_data:
            del self.image_data["contours"][self.contour_index]
            del self.image_data["scores"][self.contour_index]
            del self.image_data["measurements"][self.contour_index]
            ImageLogic.save_image_data(self.image_data)


class RemoveContoursCommand(QUndoCommand):
    """Command for removing contours from an image"""

    def __init__(self, image_data, indices, description="Remove Contours"):
        super().__init__(description)
        ImageLogic.preserve_original_annotations(image_data)
        self.image_data = image_data
        self.indices = sorted(indices, reverse=True)
        self.removed_contours = []
        self.removed_scores = []
        self.removed_measurements = []

    def redo(self):
        """Remove contours and their scores"""
        self.removed_contours = [
            (idx, self.image_data["contours"][idx])
            for idx in self.indices
            if "contours" in self.image_data and idx < len(self.image_data["contours"])
        ]
        self.removed_scores = [
            (idx, self.image_data["scores"][idx])
            for idx in self.indices
            if "scores" in self.image_data and idx < len(self.image_data["scores"])
        ]
        self.removed_measurements = [(index, self.image_data["measurements"][index]) for index in self.indices
                                     if index < len(self.image_data.get("measurements", []))]

        for idx, _ in self.removed_contours:
            del self.image_data["contours"][idx]
        for idx, _ in self.removed_scores:
            del self.image_data["scores"][idx]
        for index, _ in self.removed_measurements:
            del self.image_data["measurements"][index]

        ImageLogic.save_image_data(self.image_data)

    def undo(self):
        """Restore removed contours and their scores"""
        for idx, contour in reversed(self.removed_contours):
            self.image_data["contours"].insert(idx, contour)
        for idx, score in reversed(self.removed_scores):
            self.image_data["scores"].insert(idx, score)
        for index, record in reversed(self.removed_measurements):
            self.image_data["measurements"].insert(index, record)

        ImageLogic.save_image_data(self.image_data)


class ResetAnnotationsCommand(QUndoCommand):
    """Restore the current image's original detections as one undoable edit."""

    def __init__(self, image_data):
        super().__init__(Strings.RESET_ANNOTATIONS)
        ImageLogic.preserve_original_annotations(image_data)
        self.image_data = image_data
        self.previous = ImageLogic.annotation_snapshot(image_data)
        self.original = deepcopy(image_data["original_annotations"])

    def _apply(self, snapshot):
        self.image_data.update(deepcopy(snapshot))
        self.image_data["contours"] = [np.array(cnt, dtype=np.int32) for cnt in snapshot["contours"]]
        ImageLogic.save_image_data(self.image_data)

    def redo(self):
        self._apply(self.original)

    def undo(self):
        self._apply(self.previous)
