from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QSplitter, QListWidget, QListWidgetItem, QLabel, QPushButton, QFileDialog, QScrollArea,
                               QProgressBar, QMenu, QMessageBox, QUndoView)
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QAction, QPalette, QUndoStack
from PySide6.QtCore import Qt, QPoint, QRect
from logic.prediction_logic import PredictionLogic
from logic.export_logic import ExportLogic
from logic.image_logic import ImageLogic
from logic.commands import AddContourCommand, RemoveContoursCommand
from ui.draggable_image_list import DraggableImageList
from ui.export_dialog import ExportDialog
from config.shortcuts import Shortcuts
from config.strings import Strings
from config.general import Config
from config.icons import Icons
import cv2
import os


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize variables
        self.__image_data = []
        self.__current_index = -1
        self.__drawing = False
        self.__current_contour = []
        self.__group_selected_indices = []
        self.__group_select_active = False
        self.__group_selection_start = None
        self.__group_selection_rect = None
        self.__zoom_factor = 1.0
        self.__effective_scale = 1
        self.__cancel_prediction = False
        self.__cross_preview_mode = False
        self._show_confidences = True  # Add this line to store toggle state
        self._show_contours = True     # Add toggle state for contours visibility

        # Initialize undo stack
        self.__undo_stack = QUndoStack(self)

        # For panning
        self._panning = False
        self._pan_start_pos = None
        self._pan_start_scroll = None
        self._pan_maybe = False  # <--- add this line

        # Window
        self.setWindowTitle(Strings.WINDOW_TITLE)
        self.setMinimumSize(Config.WINDOW_SIZE_MIN)
        self.resize(Config.WINDOW_SIZE_PREFERRED)

        # Add contour button
        self.button_contour_add = QPushButton()
        self.button_contour_add.setIcon(Icons.create_plus_icon())
        self.button_contour_add.setToolTip(Strings.ADD_CONTOUR)
        self.button_contour_add.setCheckable(True)
        self.button_contour_add.clicked.connect(self.start_drawing)

        # Remove contour button
        self.button_contour_remove = QPushButton()
        self.button_contour_remove.setIcon(Icons.create_remove_icon())
        self.button_contour_remove.setToolTip(Strings.REMOVE_CONTOUR)
        self.button_contour_remove.clicked.connect(self.remove_selected_contour)

        # Group selection button
        self.button_group_select = QPushButton()
        self.button_group_select.setIcon(Icons.create_group_select_icon())
        self.button_group_select.setToolTip(Strings.GROUP_SELECT)
        self.button_group_select.setCheckable(True)
        self.button_group_select.clicked.connect(self.start_group_selection)

        # Cross view button
        self.button_cross_view = QPushButton()
        self.button_cross_view.setIcon(Icons.create_dot_icon())
        self.button_cross_view.setToolTip(Strings.CROSS_PREVIEW_TOOLTIP)
        self.button_cross_view.setCheckable(True)
        self.button_cross_view.clicked.connect(self.toggle_cross_preview)

        # Predict button
        self.button_predict = QPushButton(Strings.PREDICT)
        self.button_predict.setFixedWidth(80)
        self.button_predict.clicked.connect(self.start_prediction)

        # Cancel button
        self.button_cancel = QPushButton(Strings.CANCEL)
        self.button_cancel.setFixedWidth(80)
        self.button_cancel.setStyleSheet("background-color: red; color: white;")
        self.button_cancel.setVisible(False)
        self.button_cancel.clicked.connect(self.cancel_prediction_process)

        # Zoom out button
        self.button_zoom_out = QPushButton()
        self.button_zoom_out.setIcon(Icons.create_zoom_out_icon())
        self.button_zoom_out.setFixedWidth(30)
        self.button_zoom_out.clicked.connect(self.zoom_step_out)

        # Zoom in button
        self.button_zoom_in = QPushButton()
        self.button_zoom_in.setIcon(Icons.create_zoom_in_icon())
        self.button_zoom_in.setFixedWidth(30)
        self.button_zoom_in.clicked.connect(self.zoom_step_in)

        # Zoom label
        self.label_zoom = QLabel("100%")

        # Preview image label
        self.label_image = QLabel(Strings.IMAGE_PREVIEW)
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_image.customContextMenuRequested.connect(self.show_preview_context_menu)
        self.label_image.mousePressEvent = self.preview_mouse_press
        self.label_image.mouseMoveEvent = self.preview_mouse_move
        self.label_image.mouseReleaseEvent = self.preview_mouse_release

        # Enable wheel event on label_image for zooming
        self.label_image.wheelEvent = self.preview_wheel_event

        # Image list panel (pass self to DraggableImageList)
        self.panel_image_list = DraggableImageList(self)
        self.panel_image_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.panel_image_list.customContextMenuRequested.connect(self.show_list_context_menu)
        self.panel_image_list.currentRowChanged.connect(self.on_image_selected)

        # Image panel
        self.panel_image = QScrollArea()
        self.panel_image.setWidget(self.label_image)
        self.panel_image.setWidgetResizable(True)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)

        # Middle panel
        panel_tool = QWidget()
        panel_tool.setMaximumWidth(100)
        panel_tool_layout = QVBoxLayout(panel_tool)
        panel_tool_layout.setContentsMargins(0, 0, 0, 0)
        panel_tool_layout.addWidget(self.button_contour_add)
        panel_tool_layout.addWidget(self.button_contour_remove)
        panel_tool_layout.addWidget(self.button_group_select)
        panel_tool_layout.addWidget(self.button_cross_view)
        panel_tool_layout.addStretch()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.panel_image_list)
        splitter.addWidget(self.panel_image)
        splitter.addWidget(panel_tool)
        splitter.setStretchFactor(1, 1)

        # Bottom panel
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.button_zoom_out)
        bottom_layout.addWidget(self.button_zoom_in)
        bottom_layout.addWidget(self.label_zoom)
        bottom_layout.addWidget(self.progress_bar)
        bottom_layout.addWidget(self.button_predict)
        bottom_layout.addWidget(self.button_cancel)

        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.addWidget(splitter)
        main_layout.addLayout(bottom_layout)
        self.setCentralWidget(central)

        # Menu -> File -> Import files
        self.menu_import_files = QAction(Strings.IMPORT_FILES, self)
        self.menu_import_files.setShortcuts(Shortcuts.IMPORT_FILES)
        self.menu_import_files.triggered.connect(self.import_files)

        # Menu -> File -> Import folder
        self.menu_import_folder = QAction(Strings.IMPORT_FOLDER, self)
        self.menu_import_folder.setShortcuts(Shortcuts.IMPORT_FOLDER)
        self.menu_import_folder.triggered.connect(self.import_folder)

        # Menu -> File -> Export
        self.menu_export = QAction(Strings.EXPORT, self)
        self.menu_export.setShortcuts(Shortcuts.EXPORT)
        self.menu_export.triggered.connect(self.export_data)

        # Menu -> Edit -> Undo
        self.menu_undo = QAction(Strings.EDIT_UNDO, self)
        self.menu_undo.setShortcuts(Shortcuts.UNDO)
        self.menu_undo.triggered.connect(self.__undo_stack.undo)
        self.__undo_stack.canUndoChanged.connect(self.update_undo_actions)

        # Menu -> Edit -> Redo
        self.menu_redo = QAction(Strings.EDIT_REDO, self)
        self.menu_redo.setShortcuts(Shortcuts.REDO)
        self.menu_redo.triggered.connect(self.__undo_stack.redo)
        self.__undo_stack.canRedoChanged.connect(self.update_undo_actions)

        # Menu -> Edit -> Add contour
        self.menu_add_contour = QAction(Strings.EDIT_ADD_CONTOUR, self)
        self.menu_add_contour.setShortcuts(Shortcuts.CONTOUR_ADD)
        self.menu_add_contour.triggered.connect(self.start_drawing)

        # Menu -> Edit -> Remove contour
        self.menu_remove_contour = QAction(Strings.EDIT_REMOVE_CONTOUR, self)
        self.menu_remove_contour.setShortcuts(Shortcuts.CONTOUR_DELETE)
        self.menu_remove_contour.triggered.connect(self.remove_selected_contour)

        # Menu -> Model -> Start prediction
        self.menu_start_prediction = QAction(Strings.START_PREDICTION, self)
        self.menu_start_prediction.setShortcuts(Shortcuts.PREDICTION_START)
        self.menu_start_prediction.triggered.connect(self.start_prediction)

        # Menu -> Model -> Cancel prediction
        self.menu_cancel_prediction = QAction(Strings.CANCEL_PREDICTION, self)
        self.menu_cancel_prediction.setShortcuts(Shortcuts.PREDICTION_STOP)
        self.menu_cancel_prediction.triggered.connect(self.cancel_prediction_process)
        self.menu_cancel_prediction.setEnabled(False)

        # Menu -> View -> Zoom in
        self.menu_zoom_in = QAction(Strings.ZOOM_IN, self)
        self.menu_zoom_in.setShortcuts(Shortcuts.ZOOM_IN)
        self.menu_zoom_in.triggered.connect(self.zoom_step_in)

        # Menu -> View -> Zoom out
        self.menu_zoom_out = QAction(Strings.ZOOM_OUT, self)
        self.menu_zoom_out.setShortcuts(Shortcuts.ZOOM_OUT)
        self.menu_zoom_out.triggered.connect(self.zoom_step_out)

        # Menu -> View -> Toggle confidences
        self.menu_toggle_confidences = QAction(Strings.SHOW_CONFIDENCES, self)
        self.menu_toggle_confidences.setCheckable(True)
        self.menu_toggle_confidences.setChecked(True)
        self.menu_toggle_confidences.setShortcuts(Shortcuts.TOGGLE_CONFIDENCES)
        self.menu_toggle_confidences.triggered.connect(self.toggle_confidences)

        # Menu -> View -> Toggle contours
        self.menu_toggle_contours = QAction(Strings.SHOW_CONTOURS, self)
        self.menu_toggle_contours.setCheckable(True)
        self.menu_toggle_contours.setChecked(True)
        self.menu_toggle_contours.setShortcuts(Shortcuts.TOGGLE_CONTOURS)
        self.menu_toggle_contours.triggered.connect(self.toggle_contours)

        # Menu -> View -> Toggle cross preview
        self.menu_toggle_cross_preview = QAction(Strings.CROSS_PREVIEW, self)
        self.menu_toggle_cross_preview.setCheckable(True)
        self.menu_toggle_cross_preview.setChecked(False)
        self.menu_toggle_cross_preview.setShortcuts(Shortcuts.TOGGLE_CROSS_PREVIEW)
        self.menu_toggle_cross_preview.triggered.connect(self.toggle_cross_preview_from_menu)

        # Menu -> Help -> About
        menu_about = QAction(Strings.ABOUT, self)
        menu_about.triggered.connect(self.show_about)

        # Menu setup
        menu = self.menuBar()

        menu_file = menu.addMenu(Strings.MENU_FILE)
        menu_edit = menu.addMenu(Strings.MENU_EDIT)
        menu_model = menu.addMenu(Strings.MENU_MODEL)
        menu_view = menu.addMenu(Strings.MENU_VIEW)
        menu_help = menu.addMenu(Strings.MENU_HELP)

        # Menu setup - File
        menu_file_import = menu_file.addMenu(Strings.IMPORT)
        menu_file_import.addAction(self.menu_import_files)
        menu_file_import.addAction(self.menu_import_folder)
        menu_file.addAction(self.menu_export)

        # Menu setup - Edit
        menu_edit.addAction(self.menu_undo)
        menu_edit.addAction(self.menu_redo)
        menu_edit.addSeparator()
        menu_edit.addAction(self.menu_add_contour)
        menu_edit.addAction(self.menu_remove_contour)

        # Menu setup - Model
        menu_model.addAction(self.menu_start_prediction)
        menu_model.addAction(self.menu_cancel_prediction)

        # Menu setup - View
        menu_view.addAction(self.menu_zoom_in)
        menu_view.addAction(self.menu_zoom_out)
        menu_view.addSeparator()
        menu_view.addAction(self.menu_toggle_confidences)
        menu_view.addAction(self.menu_toggle_contours)
        menu_view.addAction(self.menu_toggle_cross_preview)
        menu_view.addAction(self.menu_toggle_contours)  # Add toggle contours to View menu

        # Menu setup - Help
        menu_help.addAction(menu_about)

        self.update_image_list()
        self.update_controls()
        self.update_undo_actions()

    def update_undo_actions(self):
        """Update the state of undo/redo actions based on undo stack state"""
        self.menu_undo.setEnabled(self.__undo_stack.canUndo())
        self.menu_redo.setEnabled(self.__undo_stack.canRedo())
        self.update_preview()  # TODO this is maybe called multiple times

    def show_about(self):
        QMessageBox.information(
            self,
            Strings.ABOUT,
            f"{Config.APP_TITLE}\n{Strings.AUTHOR} Tadeáš Fryčák\n{Strings.VERSION} {Config.APP_VERSION}"
        )

    def show_list_context_menu(self, pos: QPoint):
        # Right click -> Delete image
        context_remove = QAction(Strings.CONTEXT_DELETE_IMAGE, self)
        context_remove.setIcon(Icons.create_remove_icon())
        context_remove.setEnabled(self.panel_image_list.count() > 0)

        # Right click -> Import image/s
        context_import_files = QAction(Strings.CONTEXT_IMPORT_FILES, self)
        context_import_files.triggered.connect(self.import_files)

        # Right click -> Import folder
        context_import_folder = QAction(Strings.CONTEXT_IMPORT_FOLDER, self)
        context_import_folder.triggered.connect(self.import_folder)

        # Right click - setup
        context = QMenu()
        context.addAction(context_remove)
        context.addSeparator()
        context_import = context.addMenu(Strings.IMPORT)
        context_import.addAction(context_import_files)
        context_import.addAction(context_import_folder)

        action = context.exec(self.panel_image_list.mapToGlobal(pos))
        if action == context_remove:
            row = self.panel_image_list.indexAt(pos).row()
            if row != -1:
                del self.__image_data[row]
                self.update_image_list()
                self.update_controls()

    def show_preview_context_menu(self, pos: QPoint):
        # Right click -> Add contour
        context_add_contour = QAction(Strings.ADD_CONTOUR, self)
        context_add_contour.triggered.connect(self.start_drawing)

        # Right click -> Group select
        context_group_select = QAction(Strings.GROUP_SELECT, self)
        context_group_select.setShortcuts(Shortcuts.CONTOUR_GROUP_SELECT)
        context_group_select.triggered.connect(self.start_group_selection)

        # Right click - setup
        context = QMenu()
        context.addAction(context_add_contour)
        context.addAction(context_group_select)

        if self.__group_selected_indices:
            context_clear_group_select = QAction(Strings.CONTEXT_CLEAR_GROUP_SELECT, self)
            context_clear_group_select.setShortcuts(Shortcuts.CLEAR_GROUP_SELECT)
            context_clear_group_select.triggered.connect(self.clear_group_selection)

            context.addAction(context_clear_group_select)

        context.exec(self.label_image.mapToGlobal(pos))

    def start_group_selection(self):
        self.update_preview()
        if self.__drawing or self.__drawing:
            self.__drawing = False
            self.button_contour_add.setStyleSheet("")
            self.label_image.setCursor(Qt.ArrowCursor)

        # Toggle group selection mode.
        if self.__group_select_active:
            self.__group_select_active = False
            self.button_group_select.setStyleSheet("")
            self.__group_selected_indices = []
        else:
            self.__group_select_active = True
            highlight = self.palette().color(QPalette.Highlight).name()
            self.button_group_select.setStyleSheet("background-color: " + highlight + ";")

    def clear_group_selection(self):
        self.__group_selected_indices = []
        self.update_preview()

    def import_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, Strings.SELECT_IMAGES, "", Config.IMAGE_EXTENSIONS_FILTER)
        if files:
            self.load_files(files)

    def import_folder(self):
        folder = QFileDialog.getExistingDirectory(self, Strings.SELECT_FOLDER)
        if folder:
            self.load_files([
                os.path.join(root, f)
                for root, _, files in os.walk(folder) for f in files
                if f.lower().endswith(Config.IMAGE_EXTENSIONS)
            ])

    def load_files(self, files):
        self.progress_bar.setVisible(True)

        # Filter out files that are already loaded
        existing_paths = {data["path"] for data in self.__image_data}
        files_to_load = [f for f in files if f not in existing_paths]

        if len(files) != len(files_to_load):
            QMessageBox.warning(self, Strings.ALREADY_IMPORTED_TITLE, Strings.ALREADY_IMPORTED_MESSAGE)

        total = len(files_to_load)
        for i, f in enumerate(files_to_load):
            try:
                data = ImageLogic.load_image(f)
                self.__image_data.append(data)
            except Exception as e:
                QMessageBox.critical(self, Strings.IMAGE_LOAD_ERROR_TITLE, str(e))
                continue
            self.progress_bar.setValue(int(((i + 1) / max(total, 1)) * 100))
            QApplication.processEvents()

        self.progress_bar.setVisible(False)

        self.update_image_list()
        self.update_controls()

    def update_image_list(self):
        self.panel_image_list.clear()

        if not self.__image_data:
            placeholder = QListWidgetItem(Strings.IMAGE_LIST)
            placeholder.setFlags(Qt.NoItemFlags)  # Disable selection
            self.panel_image_list.addItem(placeholder)

        else:
            for idx, data in enumerate(self.__image_data):
                item = QListWidgetItem(os.path.basename(data["path"]))

                if data.get("processing", False):
                    item.setIcon(Icons.create_loading_icon())
                elif data.get("predicted", False):
                    item.setIcon(Icons.create_done_icon())

                self.panel_image_list.addItem(item)

            if self.__image_data and self.panel_image_list.currentRow() == -1:
                self.panel_image_list.setCurrentRow(0)

        self.update_controls()

    def on_image_selected(self, index):
        self.__current_index = index
        self.__current_contour = []
        self.__group_selected_indices = []

        self.update_preview()
        self.update_controls()

    def toggle_confidences(self):
        self._show_confidences = self.menu_toggle_confidences.isChecked()
        self.update_preview()

    def toggle_contours(self):
        """Toggle contours visibility"""
        self._show_contours = self.menu_toggle_contours.isChecked()
        self.update_preview()

    def update_preview(self):
        if self.__current_index < 0 or self.__current_index >= len(self.__image_data):
            self.label_image.setText(Strings.IMAGE_PREVIEW)
            return

        data = self.__image_data[self.__current_index]
        img = ImageLogic.draw_annotations(
            data,
            self.__cross_preview_mode,
            self.__group_selected_indices,
            self.__effective_scale,
            self._show_contours  # Pass show_contours flag to draw_annotations
        )

        # If drawing
        if self.__drawing:
            ImageLogic.draw_contour(
                img,
                self.__current_contour,
                (255, 100, 0),  # TODO color
                self.__effective_scale
            )

        # If group selecting
        if self.__group_select_active and self.__group_selection_rect is not None:
            ImageLogic.draw_dashed_rectangle(
                img,
                (self.__group_selection_rect.left(), self.__group_selection_rect.top()),
                (self.__group_selection_rect.right(), self.__group_selection_rect.bottom()),
                (0, 0, 255),  # TODO color
                self.__effective_scale
            )

        # Draw prediction scores with dynamic scaling and color based on selection
        if (
            self._show_confidences and
            "contours" in data and "scores" in data
        ):
            ImageLogic.draw_prediction_scores(
                img,
                data["contours"],
                data["scores"],
                self.__group_selected_indices,
                self.__effective_scale
            )

        # Render the image
        h, w, _ = img.shape
        pixmap = QPixmap.fromImage(QImage(img.data, w, h, 3 * w, QImage.Format_BGR888))

        w_available = self.panel_image.viewport().width()
        h_available = self.panel_image.viewport().height()
        base_scale = min(w_available / w, h_available / h, 1.0)
        self.__effective_scale = base_scale * self.__zoom_factor

        scaled = pixmap.scaled(
            pixmap.width() * self.__effective_scale,
            pixmap.height() * self.__effective_scale,
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        )
        self.label_image.setPixmap(scaled)
        self.label_image.resize(scaled.size())
        self.label_zoom.setText(f"{self.__effective_scale * 100:.0f}%")

    def zoom(self, zoom_factor):
        h_bar = self.panel_image.horizontalScrollBar()
        v_bar = self.panel_image.verticalScrollBar()
        factor = zoom_factor / self.__zoom_factor

        self.__zoom_factor = zoom_factor
        self.update_preview()

        h_bar.setValue(factor * h_bar.value() + self.panel_image.viewport().width() / 2 * (factor - 1))
        v_bar.setValue(factor * v_bar.value() + self.panel_image.viewport().height() / 2 * (factor - 1))

    def zoom_step_in(self):
        self.zoom(min(self.__zoom_factor * 1.2, 8.0))

    def zoom_step_out(self):
        self.zoom(max(self.__zoom_factor * 0.8, 0.1))

    def prediction_enable_controls(self, enable):
        self.panel_image_list.setEnabled(enable)
        self.button_predict.setVisible(enable)
        self.button_cancel.setVisible(not enable)
        self.menu_start_prediction.setEnabled(enable)
        self.menu_cancel_prediction.setEnabled(not enable)
        self.progress_bar.setVisible(not enable)

    def start_prediction(self):
        indices_to_predict = [i for i, data in enumerate(self.__image_data) if not data.get("predicted", False)]
        if not indices_to_predict:
            return

        self.prediction_enable_controls(False)
        self.__cancel_prediction = False

        total = len(indices_to_predict)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        for idx_num, idx in enumerate(indices_to_predict):
            if self.__cancel_prediction:
                break

            self.__image_data[idx]["processing"] = True
            self.update_image_list()

            contours, scores = PredictionLogic.predict_contours(self.__image_data[idx]["image"], self.__image_data[idx]["path"])
            self.__image_data[idx]["contours"] = contours
            self.__image_data[idx]["scores"] = scores

            self.__image_data[idx]["predicted"] = True
            self.__image_data[idx]["processing"] = False
            self.update_image_list()

            # --- Update preview after each prediction to show confidences ---
            self.update_preview()
            # --- End update preview ---

            # Progress bar increases as files are processed
            self.progress_bar.setValue(int(((idx_num + 1) / total) * 100))
            QApplication.processEvents()

        self.progress_bar.setVisible(False)
        self.prediction_enable_controls(True)
        self.update_preview()
        self.update_controls()

    def cancel_prediction_process(self):
        self.__cancel_prediction = True

    def start_drawing(self):
        # Immediately clear any selected contour when clicking Add Contour.
        self.__group_selected_indices = []
        self.update_preview()

        if self.__drawing:
            # Deselect contour addition
            self.__drawing = False
            self.button_contour_add.setStyleSheet("")
            self.label_image.setCursor(Qt.ArrowCursor)

        else:
            # Cancel group selection if active.
            if self.__group_select_active:
                self.__group_select_active = False
                self.button_group_select.setStyleSheet("")
                self.__group_selection_start = None
                self.__group_selection_rect = None
                self.__group_selected_indices = []

            self.__drawing = True
            self.__current_contour = []
            accent = self.palette().color(QPalette.Highlight).name()
            self.button_contour_add.setStyleSheet("background-color: " + accent + ";")
            self.label_image.setCursor(Qt.CrossCursor)

    def toggle_cross_preview(self):
        self.__cross_preview_mode = self.button_cross_view.isChecked()
        self.__toggle_cross_preview_common()

    def toggle_cross_preview_from_menu(self):
        self.__cross_preview_mode = self.menu_toggle_cross_preview.isChecked()
        self.__toggle_cross_preview_common()

    def __toggle_cross_preview_common(self):
        self.button_cross_view.setChecked(self.__cross_preview_mode)
        self.menu_toggle_cross_preview.setChecked(self.__cross_preview_mode)

        if self.__cross_preview_mode:
            accent = self.palette().color(QPalette.Highlight).name()
            self.button_cross_view.setStyleSheet("background-color: " + accent + ";")
        else:
            self.button_cross_view.setStyleSheet("")

        self.update_preview()

    def preview_mouse_press(self, event: QMouseEvent):
        if not len(self.__image_data) > 0 or self.__current_index == -1:
            return

        pt = self.get_image_coordinates(event)
        # If group selection tool is active, start group selection.
        if self.__group_select_active:
            self.__group_selection_start = pt
            self.__group_selection_rect = QRect(pt, pt)
            return

        # If in contour drawing mode, record the point and update the preview.
        elif self.__drawing:
            self.__current_contour.append((pt.x(), pt.y()))
            self.update_preview()
            return

        # Otherwise, perform manual selection: clear any group selection.
        data = self.__image_data[self.__current_index]
        selected = False
        for i, cnt in enumerate(data["contours"]):
            if cv2.pointPolygonTest(cnt, (pt.x(), pt.y()), False) >= 0:
                self.__group_selected_indices = [i]
                selected = True
                break
        if not selected:
            self.__group_selected_indices = []
        self.update_preview()
        self.update_controls()

        # Prepare for possible panning, but don't start panning yet
        if not self.__drawing and not self.__group_select_active and event.button() == Qt.LeftButton:
            self._pan_maybe = True
            self._pan_start_pos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
            self._pan_start_scroll = (
                self.panel_image.horizontalScrollBar().value(),
                self.panel_image.verticalScrollBar().value()
            )

    def preview_mouse_move(self, event: QMouseEvent):
        # Only start panning if mouse is held and moved enough
        if self._pan_maybe and self._pan_start_pos is not None:
            current_pos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
            dx = current_pos.x() - self._pan_start_pos.x()
            dy = current_pos.y() - self._pan_start_pos.y()
            if abs(dx) > 2 or abs(dy) > 2:  # small threshold to avoid accidental pan
                self._panning = True
                self._pan_maybe = False
                self.label_image.setCursor(Qt.ClosedHandCursor)
        if self._panning and self._pan_start_pos is not None:
            current_pos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
            dx = current_pos.x() - self._pan_start_pos.x()
            dy = current_pos.y() - self._pan_start_pos.y()
            h_bar = self.panel_image.horizontalScrollBar()
            v_bar = self.panel_image.verticalScrollBar()
            h_bar.setValue(self._pan_start_scroll[0] - dx)
            v_bar.setValue(self._pan_start_scroll[1] - dy)
            return
        # --- End panning logic ---

        if self.__group_selection_start is not None:
            pt = self.get_image_coordinates(event)
            self.__group_selection_rect = QRect(self.__group_selection_start, pt).normalized()
            self.update_preview()
            return

        if self.__drawing:
            pt = self.get_image_coordinates(event)
            self.__current_contour.append((pt.x(), pt.y()))
            self.update_preview()

    def preview_mouse_release(self, event: QMouseEvent):
        # End panning logic
        if self._panning or self._pan_maybe:
            self._panning = False
            self._pan_maybe = False
            self.label_image.setCursor(Qt.ArrowCursor)
            self._pan_start_pos = None
            self._pan_start_scroll = None
            return
        # --- End panning logic ---

        # Handle group selection if active.
        if self.__group_selection_rect is not None and self.__group_select_active:
            data = self.__image_data[self.__current_index]
            self.__group_selected_indices = []

            for i, cnt in enumerate(data["contours"]):
                m = cv2.moments(cnt)

                if m["m00"] != 0:
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])
                    if self.__group_selection_rect.contains(QPoint(cx, cy)):
                        self.__group_selected_indices.append(i)

            # Clear temporary group selection variables but keep the tool active.
            self.__group_selection_rect = None
            self.__group_selection_start = None
            self.update_preview()
            self.update_controls()
            return

        # If in drawing mode.
        if self.__drawing:
            # Create an undo command for this operation
            command = AddContourCommand(self.__image_data[self.__current_index], self.__current_contour)
            self.__undo_stack.push(command)

            # Update UI after the command is executed
            self.__current_contour = []
            self.update_preview()

    def get_image_coordinates(self, event: QMouseEvent):
        pixmap_scaled = self.label_image.pixmap()
        if not pixmap_scaled or self.__current_index < 0:
            return QPoint(0, 0)

        disp_size = pixmap_scaled.size()
        offset_x = (self.label_image.width() - disp_size.width()) / 2
        offset_y = (self.label_image.height() - disp_size.height()) / 2

        pos = event.position() if hasattr(event, "position") else event.localPos()

        orig_img = self.__image_data[self.__current_index]["image"]
        w_original, h_original = orig_img.shape[1], orig_img.shape[0]

        scale_factor = disp_size.width() / w_original
        x = (pos.x() - offset_x) / scale_factor
        y = (pos.y() - offset_y) / scale_factor

        return QPoint(round(x), round(y))

    def remove_selected_contour(self):
        if self.__current_index == -1 or not self.__group_selected_indices:
            return

        data = self.__image_data[self.__current_index]

        # Create an undo command for removing contours
        command = RemoveContoursCommand(data, self.__group_selected_indices)
        self.__undo_stack.push(command)

        # Clear selection after removal
        self.__group_selected_indices = []

        # Update UI
        self.update_preview()
        self.update_controls()

    def export_data(self):
        dialog = ExportDialog(self)

        if not dialog.run():
            return

        ExportLogic.export_data(dialog.get_file_name(), dialog.get_selections(), self.__image_data)

    def update_controls(self):
        has_images = len(self.__image_data) > 0 and self.__current_index != -1
        self.menu_zoom_in.setEnabled(has_images)
        self.menu_zoom_out.setEnabled(has_images)
        self.button_zoom_in.setEnabled(has_images)
        self.button_zoom_out.setEnabled(has_images)

        self.button_predict.setEnabled(any(not d.get("predicted", False) for d in self.__image_data))
        self.menu_start_prediction.setEnabled(any(not d.get("predicted", False) for d in self.__image_data))
        self.menu_export.setEnabled(has_images and all(d.get("predicted", False) for d in self.__image_data))

        self.button_contour_add.setEnabled(has_images)
        self.button_contour_remove.setEnabled(has_images and len(self.__group_selected_indices) > 0)
        self.menu_add_contour.setEnabled(has_images)
        self.menu_remove_contour.setEnabled(has_images and len(self.__group_selected_indices) > 0)
        self.button_cross_view.setEnabled(has_images)

        # Disable group selection button if no image is selected.
        self.button_group_select.setEnabled(has_images)

        # Update undo/redo actions
        self.update_undo_actions()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            if self.focusWidget() == self.label_image or not self.panel_image_list.hasFocus():
                if self.__group_selected_indices:
                    self.remove_selected_contour()
                    return

            elif self.panel_image_list.hasFocus():
                row = self.panel_image_list.currentRow()
                if row != -1:
                    del self.__image_data[row]
                    self.update_image_list()
                    self.update_controls()
                    return

        super().keyPressEvent(event)

    def preview_wheel_event(self, event):
        # Zoom in/out with Ctrl + mouse wheel
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_step_in()
            elif delta < 0:
                self.zoom_step_out()
            event.accept()
        else:
            # Pass event to parent for normal scrolling
            super(QLabel, self.label_image).wheelEvent(event)
