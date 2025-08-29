from PySide6.QtGui import QUndoCommand
from logic.image_logic import ImageLogic


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
        self.image_data = image_data
        self.contour_points = contour_points.copy()  # Make a copy to ensure we keep original points
        self.contour_index = None

    def redo(self):
        """Execute the command: add contour to the image data"""
        self.contour_index = ImageLogic.add_contour(self.image_data, self.contour_points)

    def undo(self):
        """Undo the command: remove the added contour"""
        if self.contour_index is not None and "contours" in self.image_data:
            del self.image_data["contours"][self.contour_index]
            ImageLogic.save_image_data(self.image_data)


class RemoveContoursCommand(QUndoCommand):
    """Command for removing contours from an image"""

    def __init__(self, image_data, indices, description="Remove Contours"):
        super().__init__(description)
        self.image_data = image_data
        self.indices = sorted(indices, reverse=True)
        self.removed_contours = []
        self.removed_scores = []

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

        for idx, _ in self.removed_contours:
            del self.image_data["contours"][idx]
        for idx, _ in self.removed_scores:
            del self.image_data["scores"][idx]

        ImageLogic.save_image_data(self.image_data)

    def undo(self):
        """Restore removed contours and their scores"""
        for idx, contour in reversed(self.removed_contours):
            self.image_data["contours"].insert(idx, contour)
        for idx, score in reversed(self.removed_scores):
            self.image_data["scores"].insert(idx, score)

        ImageLogic.save_image_data(self.image_data)
