from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy,
                               QSplitter, QListWidgetItem, QLabel, QPushButton, QFileDialog, QScrollArea,
                               QProgressBar, QMenu, QMessageBox, QUndoView, QStatusBar, QInputDialog)
from PySide6.QtGui import QMouseEvent, QAction, QActionGroup, QUndoStack
from PySide6.QtCore import Qt, QPoint, QPointF, QRect, QEvent, QSize, QSignalBlocker, QSettings, QTimer
from logic.prediction_logic import PredictionLogic
from logic.export_logic import ExportLogic
from logic.image_logic import ImageLogic
from logic import measurement
from logic.commands import AddContourCommand, RemoveContoursCommand, ReclassifyContoursCommand, ResetAnnotationsCommand
from ui.draggable_image_list import DraggableImageList
from ui.export_dialog import ExportDialog
from ui.contour_properties_dialog import ContourPropertiesDialog
from ui.image_preview import ImagePreview
from ui.prediction_worker import BatchPredictionWorker, InvalidationWorker
from logic.adaptive_prediction import AdaptivePredictor
from config.model import Model
from config.shortcuts import Shortcuts
from config.strings import Strings
from config.general import Config
from config.icons import Icons
import cv2
import os
import math
from time import monotonic


class MainWindow(QMainWindow):
    def __init__(self, settings=None):
        super().__init__()
        self._settings = settings if settings is not None else QSettings("RootTracker", "RootTracker")
        self._default_dpi = measurement.valid_dpi(self._settings.value("default_dpi"))

        # Initialize variables
        self.__image_data = []
        self._preview_data = None
        self.__current_index = -1
        self.__drawing = False
        self.__current_contour = []
        self.__group_selected_indices = []
        self.__group_select_active = False
        self.__group_selection_start = None
        self.__group_selection_rect = None
        self.__selection_press_position = None
        self.__selection_dragged = False
        self.__zoom_scale = None  # Fit automatically until the user zooms.
        self.__effective_scale = 1
        self.__cancel_prediction = False
        self._prediction_thread = None
        self._prediction_pending = []
        self._invalidation_thread = None
        self._predictor = AdaptivePredictor()
        QApplication.instance().aboutToQuit.connect(self._shutdown_prediction)
        self._close_after_prediction = False
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
        self.button_contour_add.setIcon(Icons.create_draw_icon())
        self.button_contour_add.setToolTip(Strings.ADD_CONTOUR)
        self.button_contour_add.setCheckable(True)
        self.button_contour_add.clicked.connect(self.start_drawing)

        # Remove contour button
        self.button_contour_remove = QPushButton()
        self.button_contour_remove.setIcon(Icons.create_remove_icon())
        self.button_contour_remove.setToolTip(Strings.REMOVE_CONTOUR)
        self.button_contour_remove.clicked.connect(self.remove_selected_contour)

        self.button_restore_classification = QPushButton()
        self.button_restore_classification.setIcon(Icons.create_restore_icon())
        self.button_restore_classification.setToolTip(Strings.RESTORE_CLASSIFICATION_TOOLTIP)
        self.button_restore_classification.clicked.connect(self.reset_annotations)

        # Group selection button
        self.button_group_select = QPushButton()
        self.button_group_select.setIcon(Icons.create_group_select_icon())
        self.button_group_select.setToolTip(Strings.GROUP_SELECT)
        self.button_group_select.setCheckable(True)
        self.button_group_select.clicked.connect(self.start_group_selection)

        # Navigation tools are explicit; pan is the initial mode.
        self.button_pan = QPushButton()
        self.button_pan.setIcon(Icons.create_pan_icon())
        self.button_pan.setToolTip(Strings.PAN)
        self.button_pan.setCheckable(True)
        self.button_pan.setChecked(True)
        self.button_pan.clicked.connect(self.start_panning)

        self.execution_status = QLabel()

        # Predict button
        self.button_predict = QPushButton(Strings.PREDICT)
        self.button_predict.setFixedWidth(80)
        self.button_predict.clicked.connect(self.start_prediction)

        # Cancel button
        self.button_cancel = QPushButton(Strings.CANCEL)
        self.button_cancel.setFixedWidth(80)
        self.button_cancel.setVisible(False)
        self.button_cancel.clicked.connect(self.cancel_prediction_process)

        # Zoom out button
        self.button_zoom_out = QPushButton()
        self.button_zoom_out.setIcon(Icons.create_zoom_out_icon())
        self.button_zoom_out.setToolTip(Strings.ZOOM_OUT)
        self.button_zoom_out.setFixedWidth(32)
        self.button_zoom_out.clicked.connect(self.zoom_step_out)

        # Zoom in button
        self.button_zoom_in = QPushButton()
        self.button_zoom_in.setIcon(Icons.create_zoom_in_icon())
        self.button_zoom_in.setToolTip(Strings.ZOOM_IN)
        self.button_zoom_in.setFixedWidth(32)
        self.button_zoom_in.clicked.connect(self.zoom_step_in)

        for button in (self.button_contour_add, self.button_contour_remove,
                       self.button_group_select, self.button_pan,
                       self.button_zoom_out, self.button_zoom_in, self.button_restore_classification):
            button.setProperty("iconButton", True)
            button.setIconSize(QSize(18, 18))
            button.setAccessibleName(button.toolTip())

        # Zoom label
        self.label_zoom = QLabel("100%")
        self.label_measurements = QLabel()
        self.label_measurements.setToolTip(Strings.MEASUREMENT_TOOLTIP)

        # Preview image label
        self.label_image = ImagePreview(Strings.IMAGE_PREVIEW)
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_image.customContextMenuRequested.connect(self.show_preview_context_menu)
        self.label_image.setCursor(Qt.OpenHandCursor)
        self.label_image.mousePressEvent = self.preview_mouse_press
        self.label_image.mouseMoveEvent = self.preview_mouse_move
        self.label_image.mouseReleaseEvent = self.preview_mouse_release

        # Wheel navigation stays anchored to the image preview.
        self.label_image.wheelEvent = self.preview_wheel_event

        # Image list panel (pass self to DraggableImageList)
        self.panel_image_list = DraggableImageList(self)
        self.panel_image_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.panel_image_list.customContextMenuRequested.connect(self.show_list_context_menu)
        self.panel_image_list.currentRowChanged.connect(self.on_image_selected)

        # Image panel
        self.panel_image = QScrollArea()
        self.panel_image.setWidget(self.label_image)
        self.panel_image.setWidgetResizable(False)
        self.panel_image.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.panel_image.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.panel_image.viewport().installEventFilter(self)
        for bar in (self.panel_image.horizontalScrollBar(), self.panel_image.verticalScrollBar()):
            bar.actionTriggered.connect(self._manual_view_interaction)
            bar.sliderMoved.connect(self._manual_view_interaction)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.setFixedHeight(self.button_predict.sizeHint().height())
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)

        self.label_time_remaining = QLabel()
        self.label_time_remaining.setVisible(False)
        self._progress_started = None
        self._progress_deadline = None
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(1000)
        self._progress_timer.timeout.connect(self._refresh_remaining_time)

        # Right toolbar stays at its preferred width outside the splitter.
        panel_tool = QWidget()
        panel_tool.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        panel_tool_layout = QVBoxLayout(panel_tool)
        panel_tool_layout.setContentsMargins(0, 0, 0, 0)
        panel_tool_layout.addWidget(self.button_pan)
        panel_tool_layout.addWidget(self.button_group_select)
        panel_tool_layout.addWidget(self.button_contour_add)
        panel_tool_layout.addWidget(self.button_contour_remove)
        panel_tool_layout.addStretch()
        panel_tool_layout.addWidget(self.button_restore_classification)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.panel_image_list)
        splitter.addWidget(self.panel_image)
        splitter.setStretchFactor(1, 1)

        # Match Root Tracker's status bar: zoom left, progress and actions right.
        status_bar = QStatusBar(self)
        status_bar.setSizeGripEnabled(False)
        status_bar.setContentsMargins(3, 3, 5, 3)
        self.setStatusBar(status_bar)
        status_bar.addWidget(self.button_zoom_out)
        status_bar.addWidget(self.button_zoom_in)
        status_bar.addWidget(self.label_zoom)
        status_bar.addPermanentWidget(self.execution_status)
        status_bar.addWidget(self.label_measurements)
        status_bar.addWidget(QWidget(), 1)
        status_bar.addPermanentWidget(self.label_time_remaining)
        status_bar.addPermanentWidget(self.progress_bar)
        status_bar.addPermanentWidget(self.button_predict)
        status_bar.addPermanentWidget(self.button_cancel)

        central = QWidget()
        main_layout = QHBoxLayout(central)
        main_layout.addWidget(splitter, 1)
        main_layout.addWidget(panel_tool)
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

        self.menu_select_all_contours = QAction(Strings.EDIT_SELECT_ALL_CONTOURS, self)
        self.menu_select_all_contours.setShortcuts(Shortcuts.CONTOUR_SELECT_ALL)
        self.menu_select_all_contours.triggered.connect(self.select_all_contours)

        # Menu -> Edit -> Remove contour
        self.menu_remove_contour = QAction(Strings.EDIT_REMOVE_CONTOUR, self)
        self.menu_remove_contour.setShortcuts(Shortcuts.CONTOUR_DELETE)
        self.menu_remove_contour.triggered.connect(self.remove_selected_contour)

        self._selected_model = Model.DEFAULT_MODEL
        self.model_action_group = QActionGroup(self)
        self.model_action_group.setExclusive(True)
        self.model_actions = {}
        for model_id, label in (("m", Strings.DETECTOR_M), ("s", Strings.DETECTOR_S)):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(model_id == self._selected_model)
            action.setData(model_id)
            action.setToolTip(Strings.DETECTOR_TOOLTIP)
            self.model_action_group.addAction(action)
            self.model_actions[model_id] = action
        self.model_action_group.triggered.connect(self._change_model)

        # Menu -> Model -> Start prediction
        self.menu_start_prediction = QAction(Strings.START_PREDICTION, self)
        self.menu_repredict = QAction(Strings.REPREDICT, self)
        self.menu_repredict.triggered.connect(self.repredict_current)
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
        menu_edit.addAction(self.menu_select_all_contours)

        # Menu setup - Model
        menu_model.addMenu(Strings.SELECT_MODEL).addActions(self.model_action_group.actions())
        dpi_menu = menu_model.addMenu("")
        self.menu_default_dpi = dpi_menu.menuAction()
        self.default_dpi_group = QActionGroup(self)
        self.default_dpi_group.setExclusive(True)
        self.default_dpi_actions = {}
        for dpi in (None, 600, 720, 1200):
            label = Strings.DPI_NONE if dpi is None else Strings.DPI_PRESET.format(value=dpi)
            action = dpi_menu.addAction(label)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, value=dpi: self.set_default_dpi(value))
            self.default_dpi_group.addAction(action)
            self.default_dpi_actions[dpi] = action
        dpi_menu.addSeparator()
        self.custom_default_dpi_action = dpi_menu.addAction(Strings.DPI_CUSTOM)
        self.custom_default_dpi_action.setCheckable(True)
        self.default_dpi_group.addAction(self.custom_default_dpi_action)
        self.custom_default_dpi_action.triggered.connect(self.choose_default_dpi)
        self._update_default_dpi_label()
        menu_model.addSeparator()
        menu_model.addAction(self.menu_start_prediction)
        menu_model.addAction(self.menu_repredict)
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

        self.__undo_stack.indexChanged.connect(self._annotations_changed)

    def _annotations_changed(self):
        self.update_image_list()
        self.update_preview()
        self.update_controls()

    def _can_edit_image(self, data=None):
        return (self._prediction_thread is None and self._invalidation_thread is None
                and 0 <= self.__current_index < len(self.__image_data)
                and bool((data or self.__image_data[self.__current_index]).get("dpi")))

    def reclassify_selected(self, nice):
        if (self._prediction_thread is not None or self._invalidation_thread is not None) or self.__current_index < 0:
            return
        data = self.__image_data[self.__current_index]
        indices = [index for index in self.__group_selected_indices if 0 <= index < len(data.get("measurements", []))]
        if indices:
            self.__undo_stack.push(ReclassifyContoursCommand(data, indices, nice))

    def reset_annotations(self):
        if (self._prediction_thread is not None or self._invalidation_thread is not None) or self.__current_index < 0:
            return
        data = self.__image_data[self.__current_index]
        self.__group_selected_indices = []
        self.__current_contour = []
        self.__group_selection_start = None
        self.__group_selection_rect = None
        self.__undo_stack.push(ResetAnnotationsCommand(data))

    def update_undo_actions(self):
        """Update the state of undo/redo actions based on undo stack state"""
        self.menu_undo.setEnabled((self._prediction_thread is None and self._invalidation_thread is None) and self.__undo_stack.canUndo())
        self.menu_redo.setEnabled((self._prediction_thread is None and self._invalidation_thread is None) and self.__undo_stack.canRedo())
        self.update_preview()  # TODO this is maybe called multiple times

    def show_about(self):
        QMessageBox.information(
            self,
            Strings.ABOUT,
            f"{Config.APP_TITLE}\n{Strings.AUTHOR} Tadeáš Fryčák\n{Strings.VERSION} {Config.APP_VERSION}"
        )

    def _update_default_dpi_label(self):
        self.menu_default_dpi.setText(Strings.DEFAULT_DPI.format(value=self._default_dpi or Strings.DPI_NONE))
        selected = self.default_dpi_actions.get(self._default_dpi, self.custom_default_dpi_action)
        selected.setChecked(True)

    def _resolve_dpi(self, data):
        dpi = data.get("dpi_override") or data.get("detected_dpi") or self._default_dpi
        if dpi != data.get("dpi"):
            data["predicted"] = False
        data["dpi"] = dpi

    def set_default_dpi(self, dpi):
        if self._prediction_thread is not None or self._invalidation_thread is not None:
            return
        if dpi is not None and measurement.valid_dpi(dpi) is None:
            raise ValueError(Strings.DPI_INVALID)
        self._default_dpi = measurement.valid_dpi(dpi)
        self._settings.setValue("default_dpi", self._default_dpi or 0)
        for data in self.__image_data:
            self._resolve_dpi(data)
        self._update_default_dpi_label()
        self.update_image_list()

    def choose_default_dpi(self):
        dpi, ok = QInputDialog.getInt(self, Strings.MENU_MODEL, Strings.DEFAULT_DPI_PROMPT,
                                     self._default_dpi or 600, 1, 100000)
        if ok:
            self.set_default_dpi(dpi)
        self._update_default_dpi_label()

    def set_image_dpi(self, row, dpi):
        if self._prediction_thread is not None or self._invalidation_thread is not None:
            return
        data = self.__image_data[row]
        ImageLogic.set_dpi_override(data, dpi, self._default_dpi)
        self.__undo_stack.clear()
        self.update_image_list()

    def choose_image_dpi(self, row):
        data = self.__image_data[row]
        dpi, ok = QInputDialog.getInt(self, Strings.SET_DPI, Strings.DPI_PROMPT,
                                     data.get("dpi") or 0, 0, 100000)
        if ok:
            try:
                self.set_image_dpi(row, dpi or None)
            except (OSError, ValueError) as error:
                QMessageBox.critical(self, Strings.DPI_SAVE_FAILED, str(error))

    def show_list_context_menu(self, pos: QPoint):
        # Right click -> Delete image
        context_remove = QAction(Strings.CONTEXT_DELETE_IMAGE, self)
        context_remove.setEnabled(self.panel_image_list.indexAt(pos).isValid())

        # Right click -> Import image/s
        context_import_files = QAction(Strings.CONTEXT_IMPORT_FILES, self)
        context_import_files.triggered.connect(self.import_files)

        # Right click -> Import folder
        context_import_folder = QAction(Strings.CONTEXT_IMPORT_FOLDER, self)
        context_import_folder.triggered.connect(self.import_folder)

        # Right click - setup
        context = QMenu()
        context.addAction(context_remove)
        context_dpi = context.addAction(Strings.SET_DPI)
        context_dpi.setEnabled(self.panel_image_list.indexAt(pos).isValid() and self._invalidation_thread is None)
        context.addSeparator()
        context_import = context.addMenu(Strings.IMPORT)
        context_import.addAction(context_import_files)
        context_import.addAction(context_import_folder)

        action = context.exec(self.panel_image_list.mapToGlobal(pos))
        if action == context_dpi:
            self.choose_image_dpi(self.panel_image_list.indexAt(pos).row())
        if action == context_remove:
            row = self.panel_image_list.indexAt(pos).row()
            if row != -1:
                del self.__image_data[row]
                self.update_image_list()
                self.update_controls()

    def _contour_at_position(self, pos):
        if self.label_image.data is None:
            return None
        data = self.__image_data[self.__current_index]
        point = self.label_image.image_point(QPointF(pos))
        nearest, best_distance = None, -6.0 / self.__effective_scale
        for index, contour in enumerate(data["contours"]):
            distance = cv2.pointPolygonTest(contour, (point.x(), point.y()), True)
            if distance >= 0:
                return index
            if distance >= best_distance:
                nearest, best_distance = index, distance
        return nearest

    def show_preview_context_menu(self, pos: QPoint):
        hit = self._contour_at_position(pos)
        if hit is None:
            self.__group_selected_indices = []
        elif hit not in self.__group_selected_indices:
            self.__group_selected_indices = [hit]
        self.update_preview()
        self.update_controls()
        selected = bool(self.__group_selected_indices)
        editable = selected and (self._prediction_thread is None and self._invalidation_thread is None)
        context = QMenu(self)
        add = context.addAction(Strings.ADD_CONTOUR, self.start_drawing)
        add.setEnabled(self.label_image.data is not None and (self._prediction_thread is None and self._invalidation_thread is None))
        context.addSeparator()
        remove = context.addAction(Strings.REMOVE_CONTOUR, self.remove_selected_contour)
        remove.setEnabled(editable)
        records = self.label_image.data.get("measurements", []) if self.label_image.data else []
        classes = [records[i].get("nice") if i < len(records) else None
                   for i in self.__group_selected_indices]
        if any(value is not True for value in classes):
            mark_nice = context.addAction(Strings.MARK_NICE, lambda: self.reclassify_selected(True))
            mark_nice.setEnabled(editable)
        if any(value is not False for value in classes):
            mark_not_nice = context.addAction(Strings.MARK_NOT_NICE, lambda: self.reclassify_selected(False))
            mark_not_nice.setEnabled(editable)
        context.addSeparator()
        properties = context.addAction(Strings.CONTOUR_PROPERTIES, self.show_contour_properties)
        properties.setEnabled(selected)
        context.exec(self.label_image.mapToGlobal(pos))

    def show_contour_properties(self):
        if self.__current_index < 0 or not self.__group_selected_indices:
            return
        data = self.__image_data[self.__current_index]
        ContourPropertiesDialog(data, self.__group_selected_indices, self).exec()

    def _set_preview_tool(self, tool):
        if (self._prediction_thread is not None or self._invalidation_thread is not None) and tool == "draw":
            return
        self.__drawing = tool == "draw"
        self.__group_select_active = tool == "select"
        self.button_pan.setChecked(tool == "pan")
        self.button_group_select.setChecked(self.__group_select_active)
        self.button_contour_add.setChecked(self.__drawing)
        self.__current_contour = []
        self.__group_selected_indices = []
        self.__group_selection_start = None
        self.__group_selection_rect = None
        self._panning = self._pan_maybe = False
        self._pan_start_pos = self._pan_start_scroll = None
        self.label_image.setCursor(Qt.OpenHandCursor if tool == "pan" else Qt.CrossCursor)
        self.update_preview()
        self.update_controls()

    def start_panning(self):
        self._set_preview_tool("pan")

    def start_group_selection(self):
        self._set_preview_tool("select")

    def select_all_contours(self):
        if self.__current_index == -1:
            return
        self.start_group_selection()
        data = self.__image_data[self.__current_index]
        self.__group_selected_indices = list(range(len(data.get("contours", []))))
        self.update_preview()
        self.update_controls()

    def clear_group_selection(self):
        self.__group_selected_indices = []
        self.update_preview()
        self.update_controls()

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

    def _begin_progress(self, process_events=True):
        self._progress_started = monotonic()
        self._progress_deadline = None
        self._progress_timer.start()
        self.progress_bar.setValue(0)
        self.label_time_remaining.setText(Strings.ESTIMATING)
        self.progress_bar.show()
        self.label_time_remaining.show()
        if process_events:
            QApplication.processEvents()

    def _update_progress(self, completed, total, process_events=True):
        self.progress_bar.setValue(int(completed / max(total, 1) * 100))
        if completed and self._progress_started is not None:
            now = monotonic()
            remaining = (now - self._progress_started) / completed * max(0, total - completed)
            self._progress_deadline = now + remaining
            self._refresh_remaining_time()
        if process_events:
            QApplication.processEvents()

    def _refresh_remaining_time(self):
        if self._progress_deadline is None:
            return
        remaining = max(0, math.ceil(self._progress_deadline - monotonic()))
        if remaining >= 60:
            text = Strings.TIME_REMAINING_MINUTES.format(
                minutes=remaining // 60, seconds=remaining % 60)
        else:
            text = Strings.TIME_REMAINING_SECONDS.format(seconds=remaining)
        self.label_time_remaining.setText(text)

    def _end_progress(self):
        self._progress_timer.stop()
        self._progress_deadline = None
        self.progress_bar.hide()
        self.label_time_remaining.hide()
        self.label_time_remaining.clear()
        self._progress_started = None

    def load_files(self, files):
        if self._prediction_thread is not None:
            return
        # Filter out files that are already loaded
        existing_paths = {data["path"] for data in self.__image_data}
        files_to_load = [f for f in files if f not in existing_paths]

        if len(files) != len(files_to_load):
            QMessageBox.warning(self, Strings.ALREADY_IMPORTED_TITLE, Strings.ALREADY_IMPORTED_MESSAGE)

        total = len(files_to_load)
        self._begin_progress()
        try:
            for i, f in enumerate(files_to_load):
                try:
                    data = ImageLogic.load_image(f, load_pixels=False, default_dpi=self._default_dpi)
                    self._resolve_dpi(data)
                    self.__image_data.append(data)
                except Exception as e:
                    QMessageBox.critical(self, Strings.IMAGE_LOAD_ERROR_TITLE, str(e))
                self._update_progress(i + 1, total)
        finally:
            self._end_progress()
            self.update_image_list()
            self.update_controls()

    def update_image_list(self):
        current = self.panel_image_list.currentItem()
        current_path = current.data(Qt.UserRole + 1) if current is not None else None
        row = min(max(self.__current_index, 0), len(self.__image_data) - 1)
        # Rebuilding status rows must not clear and decode the preview again.
        with QSignalBlocker(self.panel_image_list):
            self.panel_image_list.clear()
            if self.__image_data:
                for idx, data in enumerate(self.__image_data):
                    item = QListWidgetItem(os.path.basename(data["path"]))
                    item.setData(Qt.UserRole + 1, data["path"])
                    item.setToolTip(data["path"])
                    if "measurements" in data and data.get("predicted"):
                        records = data["measurements"]
                        nice_count = sum(record.get("nice") is True for record in records)
                        unknown = sum(record.get("nice") is None for record in records)
                        item.setToolTip(Strings.MEASUREMENT_FILE_TOOLTIP.format(path=data['path'], total=len(records), nice=nice_count, unknown=unknown, pipeline=data.get('pipeline', '')))

                    dpi = data.get("dpi")
                    item.setData(Qt.UserRole + 3, not bool(dpi))
                    source = (Strings.DPI_SOURCE_OVERRIDE if data.get("dpi_override") else
                              Strings.DPI_SOURCE_METADATA if data.get("detected_dpi") else Strings.DPI_SOURCE_DEFAULT)
                    item.setToolTip(item.toolTip() + "\n" + (Strings.DPI_TOOLTIP.format(value=dpi, source=source)
                                                            if dpi else Strings.DPI_MISSING))
                    item.setData(Qt.UserRole + 2, data.get("processing", False))
                    item.setData(Qt.UserRole, data.get("predicted", False) and not data.get("processing", False))

                    self.panel_image_list.addItem(item)

            for index, data in enumerate(self.__image_data):
                if data["path"] == current_path:
                    row = index
                    break
            self.panel_image_list.setCurrentRow(row)

        selected = self.__image_data[row] if row >= 0 else None
        if row != self.__current_index or selected is not self._preview_data:
            self.on_image_selected(row)
        self.update_controls()

    def on_image_selected(self, index):
        self.__current_index = index
        if 0 <= index < len(self.__image_data):
            self.__image_data[index].pop("load_error", None)
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

    def _manual_view_interaction(self, *args):
        if self.label_image.data is not None:
            # A pan also leaves automatic fit mode, preserving its current scale.
            self.__zoom_scale = self.__effective_scale

    def _release_preview(self):
        if self._preview_data is not None and self.label_image.data is not None and self.__zoom_scale is not None:
            viewport = self.panel_image.viewport()
            center = self.label_image.image_point(QPointF(
                self.panel_image.horizontalScrollBar().value() + viewport.width() / 2,
                self.panel_image.verticalScrollBar().value() + viewport.height() / 2))
            # Runtime view state contains no pixels and is not written to sidecars.
            self._preview_data["_view_state"] = (self.__zoom_scale, center.x(), center.y())
        # Drop Qt's borrowed buffer and pyramid before releasing the NumPy array.
        self.label_image.clear_image(Strings.IMAGE_PREVIEW)
        if self._preview_data is not None:
            self._preview_data["image"] = None
        self._preview_data = None

    def update_preview(self):
        has_image = 0 <= self.__current_index < len(self.__image_data)
        scroll_policy = Qt.ScrollBarAlwaysOn if has_image else Qt.ScrollBarAlwaysOff
        self.panel_image.setHorizontalScrollBarPolicy(scroll_policy)
        self.panel_image.setVerticalScrollBarPolicy(scroll_policy)
        if self.__current_index < 0 or self.__current_index >= len(self.__image_data):
            self._release_preview()
            self.label_image.resize(self.panel_image.viewport().size())
            self.label_measurements.clear()
            return

        data = self.__image_data[self.__current_index]
        if data is not self._preview_data:
            self._release_preview()
            self._preview_data = data
            state = data.get("_view_state")
            self.__zoom_scale = state[0] if state is not None else None
        if not data.get("load_error"):
            try:
                ImageLogic.ensure_pixels(data)
            except (OSError, ValueError, cv2.error):
                data["load_error"] = Strings.IMAGE_LOAD_ERROR_MESSAGE.format(file_path=data["path"])
        if data.get("load_error"):
            self.label_image.clear_image(data["load_error"])
            self.label_image.resize(self.panel_image.viewport().size())
            self.label_measurements.clear()
            return
        records = data.get("measurements", [])
        if data.get("predicted"):
            nice_count = sum(record.get("nice") is True for record in records)
            self.label_measurements.setText(Strings.MEASUREMENT_SUMMARY.format(total=len(data['contours']), nice=nice_count))
        else:
            self.label_measurements.clear()
        preview = self.label_image
        changed = preview.set_data(data)
        preview.selected = set(self.__group_selected_indices)
        preview.crosses = self.__cross_preview_mode
        preview.show_contours = self._show_contours
        preview.show_scores = self._show_confidences
        preview.drawing = self.__current_contour if self.__drawing else []
        preview.selection_rect = self.__group_selection_rect if self.__group_select_active else None
        viewport = self.panel_image.viewport()
        h, w = data['image'].shape[:2]
        base_scale = min(viewport.width() / w, viewport.height() / h, 1.0)
        self.__effective_scale = base_scale if self.__zoom_scale is None else self.__zoom_scale
        preview.set_view(self.__effective_scale, viewport.size())
        if changed:
            state = data.get("_view_state")
            image_center = QPointF(state[1], state[2]) if state is not None else QPointF(w / 2, h / 2)
            center = preview.widget_point(image_center)
            self.panel_image.horizontalScrollBar().setValue(round(center.x() - viewport.width() / 2))
            self.panel_image.verticalScrollBar().setValue(round(center.y() - viewport.height() / 2))
        self.label_zoom.setText(f"{self.__effective_scale * 100:.0f}%")

    def zoom(self, scale, anchor=None):
        if self.label_image.data is None:
            return
        viewport = self.panel_image.viewport()
        if anchor is None:
            anchor = QPointF(viewport.width() / 2, viewport.height() / 2)
        h_bar = self.panel_image.horizontalScrollBar()
        v_bar = self.panel_image.verticalScrollBar()
        before = self.label_image.image_point(anchor + QPointF(h_bar.value(), v_bar.value()))
        # Limits are image magnification, not multiples of fit-to-window.
        self.__zoom_scale = min(Config.MAX_ZOOM_FACTOR, max(Config.MIN_ZOOM_FACTOR, scale))
        self.update_preview()
        after = self.label_image.widget_point(before)
        h_bar.setValue(round(after.x() - anchor.x()))
        v_bar.setValue(round(after.y() - anchor.y()))

    def eventFilter(self, watched, event):
        if watched is self.panel_image.viewport() and event.type() == QEvent.Resize:
            h_bar = self.panel_image.horizontalScrollBar()
            v_bar = self.panel_image.verticalScrollBar()
            center = None
            if self.label_image.data is not None and event.oldSize().isValid():
                center = self.label_image.image_point(QPointF(
                    h_bar.value() + event.oldSize().width() / 2,
                    v_bar.value() + event.oldSize().height() / 2))
            self.update_preview()
            if center is not None:
                position = self.label_image.widget_point(center)
                h_bar.setValue(round(position.x() - event.size().width() / 2))
                v_bar.setValue(round(position.y() - event.size().height() / 2))
        return super().eventFilter(watched, event)

    def zoom_step_in(self):
        self.zoom(self.__effective_scale * 1.2)

    def zoom_step_out(self):
        self.zoom(self.__effective_scale / 1.2)

    def prediction_enable_controls(self, enable):
        self.panel_image_list.setEnabled(enable)
        self.menu_import_files.setEnabled(enable)
        self.menu_import_folder.setEnabled(enable)
        self.button_predict.setVisible(enable)
        self.button_cancel.setVisible(not enable)
        self.menu_start_prediction.setEnabled(enable)
        self.menu_cancel_prediction.setEnabled(not enable)
        self.progress_bar.setVisible(not enable)

    def _change_model(self, action):
        model_id = action.data()
        if self._prediction_thread is not None or self._invalidation_thread is not None or model_id == self._selected_model:
            self.model_actions[self._selected_model].setChecked(True)
            return
        self._selected_model = model_id
        pending = []
        for data in self.__image_data:
            was_predicted = data.get("predicted", False)
            data["predicted"] = False
            if was_predicted:
                pending.append(dict(data))
        if pending:
            worker = InvalidationWorker(pending, self)
            self._invalidation_thread = worker
            worker.finished.connect(self._invalidation_finished)
            self.model_action_group.setEnabled(False)
            worker.start()
        # Only status changed: retain rows, selection, scroll and decoded pixels.
        for index in range(self.panel_image_list.count()):
            item = self.panel_image_list.item(index)
            item.setData(Qt.UserRole, False)
            item.setToolTip(item.data(Qt.UserRole + 1))
        self.update_controls()

    def _invalidation_finished(self):
        worker = self._invalidation_thread
        errors = worker.errors
        self._invalidation_thread = None
        worker.deleteLater()
        self.model_action_group.setEnabled(True)
        self.update_controls()
        if errors:
            QMessageBox.warning(self, Strings.MODEL_CHANGE_SAVE_FAILED, "\n".join(errors))
        if self._close_after_prediction:
            self.close()

    def start_prediction(self):
        self._start_prediction_batch([data for data in self.__image_data if not data.get("predicted", False)])

    def repredict_current(self):
        if self._prediction_thread is not None or not 0 <= self.__current_index < len(self.__image_data):
            return
        if QMessageBox.question(self, Strings.REPREDICT, Strings.REPREDICT_CONFIRM) != QMessageBox.Yes:
            return
        self._start_prediction_batch([self.__image_data[self.__current_index]], replace=True)

    def _start_prediction_batch(self, records, replace=False):
        records = [data for data in records if data.get("dpi")]
        if self._prediction_thread is not None or self._invalidation_thread is not None or not records:
            return
        self.__cancel_prediction = False
        self._prediction_pending = records
        self._prediction_by_path = {data["path"]: data for data in records}
        self._prediction_total = len(self._prediction_by_path)
        self._prediction_completed = 0
        for data in records:
            data["_prediction_queued"] = True
        worker = BatchPredictionWorker(list(self._prediction_by_path), self._selected_model,
                                       self._predictor, self, replace=replace,
                                       dpi_by_path={d["path"]: d["dpi"] for d in records},
                                       dpi_overrides={d["path"]: d.get("dpi_override") for d in records})
        self._prediction_thread = worker
        worker.image_started.connect(self._prediction_started)
        worker.image_ready.connect(self._prediction_result)
        worker.status_changed.connect(self._prediction_status)
        worker.finished.connect(self._prediction_finished)
        self.model_action_group.setEnabled(False)
        self.prediction_enable_controls(False)
        self.update_controls()
        self.update_image_list()
        self._begin_progress(process_events=False)
        worker.start()

    def _prediction_status(self, device, workers, fallback):
        template = Strings.CPU_FALLBACK_STATUS if fallback else Strings.EXECUTION_STATUS
        self.execution_status.setText(template.format(device=device.upper(), workers=workers))

    def _prediction_started(self, path):
        self._prediction_by_path[path]["processing"] = True
        self.update_image_list()
        self.update_controls()

    def _prediction_result(self, path, result):
        data = self._prediction_by_path[path]
        data["processing"] = False
        data.pop("_prediction_queued", None)
        self.__undo_stack.clear()
        self.__group_selected_indices = []
        data.pop("original_annotations", None)
        data.update(result)
        data["predicted"] = True
        self._prediction_completed += 1

        self.update_image_list()
        self.update_preview()
        self.update_controls()
        self._update_progress(self._prediction_completed, self._prediction_total, process_events=False)

    def _prediction_finished(self):
        worker = self._prediction_thread
        error = worker.error
        for data in self._prediction_pending:
            data["processing"] = False
            data.pop("_prediction_queued", None)
        worker.deleteLater()
        self._prediction_thread = None
        self._prediction_pending = []
        self.model_action_group.setEnabled(True)
        self._end_progress()
        self.prediction_enable_controls(True)
        self.update_image_list()
        self.update_controls()
        if error is not None:
            QMessageBox.critical(self, Strings.PREDICTION_FAILED, error)
        if self._close_after_prediction:
            self.close()

    def cancel_prediction_process(self):
        self.__cancel_prediction = True
        if self._prediction_thread is not None:
            self._prediction_thread.cancel()

    def _shutdown_prediction(self):
        if self._invalidation_thread is not None:
            self._invalidation_thread.wait()
        if self._prediction_thread is not None:
            self._prediction_thread.cancel()
            self._prediction_thread.wait()
        self._predictor.close()

    def closeEvent(self, event):
        if self._invalidation_thread is not None:
            self._close_after_prediction = True
            event.ignore()
            return
        if self._prediction_thread is not None:
            self._close_after_prediction = True
            self.cancel_prediction_process()
            event.ignore()
            return
        self._predictor.close()
        super().closeEvent(event)

    def start_drawing(self):
        self._set_preview_tool("pan" if self.__drawing else "draw")

    def toggle_cross_preview_from_menu(self):
        self.__cross_preview_mode = self.menu_toggle_cross_preview.isChecked()
        self.__toggle_cross_preview_common()

    def __toggle_cross_preview_common(self):
        self.menu_toggle_cross_preview.setChecked(self.__cross_preview_mode)

        self.update_preview()

    def preview_mouse_press(self, event: QMouseEvent):
        if self.label_image.data is None or event.button() != Qt.LeftButton:
            return

        # Alt temporarily pans in selection mode without changing the tool.
        if self.button_pan.isChecked() or (
                self.__group_select_active and event.modifiers() & Qt.AltModifier):
            self._pan_maybe = True
            self._pan_select_click = self.button_pan.isChecked() and not (
                event.modifiers() & ~(Qt.ControlModifier | Qt.ShiftModifier))
            self._pan_start_pos = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else event.globalPos()
            self._pan_start_scroll = (
                self.panel_image.horizontalScrollBar().value(),
                self.panel_image.verticalScrollBar().value()
            )
            return

        pt = self.get_image_coordinates(event)
        # If group selection tool is active, start group selection.
        if self.__group_select_active:
            self.__group_selection_start = pt
            self.__selection_press_position = QPointF(event.position())
            self.__selection_dragged = False
            self.__group_selection_rect = None
            return

        # If in contour drawing mode, record the point and update the preview.
        elif self.__drawing:
            if (self._prediction_thread is not None or self._invalidation_thread is not None):
                return
            h, w = self.__image_data[self.__current_index]["image"].shape[:2]
            if not (0 <= pt.x() < w and 0 <= pt.y() < h):
                return
            self.__current_contour.append((pt.x(), pt.y()))
            self.update_preview()
            return


    def preview_mouse_move(self, event: QMouseEvent):
        if self.label_image.data is None:
            return
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
            self._manual_view_interaction()
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
            movement = event.position() - self.__selection_press_position
            if movement.manhattanLength() >= QApplication.startDragDistance():
                self.__selection_dragged = True
            if not self.__selection_dragged:
                return
            pt = self.get_image_coordinates(event)
            self.__group_selection_rect = QRect(self.__group_selection_start, pt).normalized()
            self.update_preview()
            return

        if self.__drawing and self.__current_contour and (self._prediction_thread is None and self._invalidation_thread is None):
            pt = self.get_image_coordinates(event)
            h, w = self.__image_data[self.__current_index]["image"].shape[:2]
            pt = QPoint(min(w - 1, max(0, pt.x())), min(h - 1, max(0, pt.y())))
            self.__current_contour.append((pt.x(), pt.y()))
            self.update_preview()

    def preview_mouse_release(self, event: QMouseEvent):
        if self.label_image.data is None:
            return
        # End panning logic
        if self._panning or self._pan_maybe:
            if event.button() != Qt.LeftButton:
                return
            position = event.globalPosition().toPoint()
            movement = position - self._pan_start_pos
            select_click = (self._pan_select_click and not self._panning
                            and abs(movement.x()) <= 2 and abs(movement.y()) <= 2
                            and not (event.modifiers() & ~(Qt.ControlModifier | Qt.ShiftModifier)))
            self._panning = False
            self._pan_maybe = False
            self.label_image.setCursor(
                Qt.OpenHandCursor if self.button_pan.isChecked() else Qt.CrossCursor)
            self._pan_start_pos = None
            self._pan_start_scroll = None
            if select_click:
                nearest = self._contour_at_position(event.position())
                if event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier):
                    if nearest is not None:
                        if nearest in self.__group_selected_indices:
                            self.__group_selected_indices.remove(nearest)
                        else:
                            self.__group_selected_indices.append(nearest)
                else:
                    self.__group_selected_indices = [] if nearest is None else [nearest]
                self.update_preview()
                self.update_controls()
            return
        # --- End panning logic ---

        # Handle group selection if active.
        if (event.button() == Qt.LeftButton and self.__group_selection_start is not None
                and self.__group_select_active):
            data = self.__image_data[self.__current_index]
            self.__group_selected_indices = []
            movement = event.position() - self.__selection_press_position
            if self.__selection_dragged or movement.manhattanLength() >= QApplication.startDragDistance():
                rect = QRect(self.__group_selection_start, self.get_image_coordinates(event)).normalized()
                for i, cnt in enumerate(data["contours"]):
                    m = cv2.moments(cnt)
                    if m["m00"] != 0 and rect.contains(QPoint(
                            int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))):
                        self.__group_selected_indices.append(i)
            else:
                # Use screen-space tolerance so tiny contours remain clickable at fit.
                nearest = self._contour_at_position(event.position())
                if nearest is not None:
                    self.__group_selected_indices = [nearest]

            # Clear temporary group selection variables but keep the tool active.
            self.__group_selection_rect = None
            self.__group_selection_start = None
            self.__selection_press_position = None
            self.__selection_dragged = False
            self.update_preview()
            self.update_controls()
            return

        # If in drawing mode.
        if self.__drawing and self.__current_contour and (self._prediction_thread is None and self._invalidation_thread is None):
            # Create an undo command for this operation
            try:
                command = AddContourCommand(self.__image_data[self.__current_index], self.__current_contour)
            except Exception as error:
                self.__current_contour = []
                self.update_preview()
                QMessageBox.critical(self, Strings.CONTOUR_CLASSIFICATION_FAILED, str(error))
                return
            self.__undo_stack.push(command)

            # Update UI after the command is executed
            self.__current_contour = []
            self.update_preview()
            self.update_controls()

    def get_image_coordinates(self, event: QMouseEvent):
        if self.label_image.data is None:
            return QPoint(0, 0)
        return self.label_image.image_point(event.position()).toPoint()

    def remove_selected_contour(self):
        if (self._prediction_thread is not None or self._invalidation_thread is not None):
            return
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
        busy = self._prediction_thread is not None or self._invalidation_thread is not None
        has_images = len(self.__image_data) > 0 and self.__current_index != -1
        can_edit = self._can_edit_image()
        self.menu_zoom_in.setEnabled(has_images)
        self.menu_zoom_out.setEnabled(has_images)
        self.button_zoom_in.setEnabled(has_images)
        self.button_zoom_out.setEnabled(has_images)

        self.button_predict.setEnabled(not busy and any(d.get("dpi") and not d.get("predicted", False) for d in self.__image_data))
        self.menu_default_dpi.setEnabled(not busy)
        self.menu_repredict.setEnabled(not busy and has_images and bool(self.__image_data[self.__current_index].get("dpi")))
        self.menu_start_prediction.setEnabled(not busy and any(d.get("dpi") and not d.get("predicted", False) for d in self.__image_data))
        self.menu_export.setEnabled(can_edit and all(d.get("predicted", False) for d in self.__image_data))

        self.button_contour_add.setEnabled(can_edit)
        self.button_contour_remove.setVisible(bool(self.__group_selected_indices))
        self.button_contour_remove.setEnabled(can_edit and len(self.__group_selected_indices) > 0)
        self.button_restore_classification.setVisible(has_images)
        has_edits = False
        if has_images:
            data = self.__image_data[self.__current_index]
            original = data.get("original_annotations")
            has_edits = (ImageLogic.annotation_snapshot(data) != original if original is not None
                         else any(record.get("status") == "manual" or "nice_override" in record
                                  for record in data.get("measurements", [])))
        self.button_restore_classification.setEnabled(can_edit and has_edits)
        self.menu_add_contour.setEnabled(can_edit)
        self.menu_select_all_contours.setEnabled(has_images)
        self.menu_remove_contour.setEnabled(can_edit and len(self.__group_selected_indices) > 0)
        self.button_pan.setEnabled(has_images)

        # Disable group selection button if no image is selected.
        self.button_group_select.setEnabled(has_images)

        # Update undo/redo actions
        self.update_undo_actions()

    def keyPressEvent(self, event):
        if (self._prediction_thread is not None or self._invalidation_thread is not None):
            return
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
        modifiers = event.modifiers()
        pixels = event.pixelDelta()
        angle = event.angleDelta()
        if modifiers & (Qt.ShiftModifier | Qt.AltModifier):
            # Alt takes precedence if both modifiers are held.
            bar = (self.panel_image.horizontalScrollBar() if modifiers & Qt.AltModifier
                   else self.panel_image.verticalScrollBar())
            distance = pixels.y() or pixels.x()
            if not distance:
                distance = (angle.y() or angle.x()) / 120 * QApplication.wheelScrollLines() * bar.singleStep()
            if distance:
                self._manual_view_interaction()
            bar.setValue(bar.value() - round(distance))
        elif angle.y() or pixels.y():
            # Plain wheel and Ctrl+wheel both zoom about the pointer.
            delta = angle.y() or pixels.y()
            anchor = event.position() - QPointF(
                self.panel_image.horizontalScrollBar().value(),
                self.panel_image.verticalScrollBar().value())
            self.zoom(self.__effective_scale * 1.2 ** (delta / 120), anchor)
        elif pixels.x() or angle.x():
            # Preserve native horizontal trackpad/tilt-wheel panning.
            bar = self.panel_image.horizontalScrollBar()
            distance = pixels.x() or angle.x() / 120 * QApplication.wheelScrollLines() * bar.singleStep()
            if distance:
                self._manual_view_interaction()
            bar.setValue(bar.value() - round(distance))
        event.accept()
