class Strings:
    # Main Window
    WINDOW_TITLE = "Root Worm Detector"
    MEASUREMENT_TOOLTIP = "Green: nice; red: not nice; grey: unclassified. Blue contour outline: selected. Nice threshold: 0.3. Manual overrides do not change model probabilities or areas."
    MARK_NICE = "Mark nice"
    MARK_NOT_NICE = "Mark not nice"
    RESET_ANNOTATIONS = "Reset all edits"
    RESTORE_CLASSIFICATION = "Restore model"
    MARK_NICE_TOOLTIP = "Mark all selected contours as nice (undoable); preserve model probabilities and areas."
    MARK_NOT_NICE_TOOLTIP = "Mark all selected contours as not nice (undoable); preserve model probabilities and areas."
    RESTORE_CLASSIFICATION_TOOLTIP = "Reset all edits on this image: restore original detections and labels, and remove manually added objects (undoable)."
    MEASUREMENT_SUMMARY = "Total: {total} | Nice: {nice}"
    MEASUREMENT_FILE_TOOLTIP = "{path}\nTotal: {total} | Nice: {nice} | Unclassified: {unknown}\n{pipeline}"
    PREDICTION_FAILED = "Prediction failed"
    EXPORT_NICE = "Nice counts and refined nice-female areas (mm²)"
    EXPORT_INDIVIDUAL = "Per-female measurements (Females sheet / companion CSV)"
    EXPORT_CONTOUR_TOTAL = "Total display-contour area (px²; not refined mask area)"
    EXPORT_CONTOUR_AVERAGE = "Average display-contour area (px²; + stdev, stderr, variance)"
    EXPORT_CONTOUR_MEDIAN = "Median display-contour area (px²; + quantiles)"

    # Button texts & tooltips
    ADD_CONTOUR = "Add Contour"
    REMOVE_CONTOUR = "Remove Contour"
    PAN = "Pan image"
    GROUP_SELECT = "Group Select"
    CROSS_PREVIEW_TOOLTIP = "Toggle dot preview mode for contours"
    PREDICT = "Predict"
    CANCEL = "Cancel"
    ESTIMATING = "Estimating…"
    TIME_REMAINING_SECONDS = "{seconds}s remaining"
    TIME_REMAINING_MINUTES = "{minutes}m {seconds}s remaining"
    ZOOM_OUT_SYMBOL = "−"
    ZOOM_IN_SYMBOL = "+"

    # Menus
    MENU_FILE = "File"
    MENU_EDIT = "Edit"
    MENU_MODEL = "Model"
    MENU_VIEW = "View"
    MENU_HELP = "Help"

    # File Menu items
    IMPORT = "Import..."
    IMPORT_FILES = "Import Files"
    IMPORT_FOLDER = "Import Folder"
    EXPORT = "Export"

    # Edit Menu items
    EDIT_ADD_CONTOUR = "Add Contour"
    CONTOUR_CLASSIFICATION_FAILED = "Could not classify the drawn contour"
    CONTOUR_CLASSIFICATION_INVALID_SHAPE = "The drawn contour is too small or outside the image to classify."
    CONTOUR_CLASSIFICATION_INVALID_FEATURES = "The contour could not be classified because its measured features are invalid."
    CONTOUR_CLASSIFICATION_INVALID_PROBABILITY = "The classifier returned an invalid probability."
    EDIT_SELECT_ALL_CONTOURS = "Select All Contours"
    EDIT_REMOVE_CONTOUR = "Remove Contour"
    EDIT_UNDO = "Undo"
    EDIT_REDO = "Redo"

    # Model Menu items
    START_PREDICTION = "Start Prediction"
    CANCEL_PREDICTION = "Cancel Prediction"

    # View Menu items
    PREVIEW_NAVIGATION_TOOLTIP = "Scroll: zoom at cursor • Shift+scroll: pan vertically • Alt+scroll: pan horizontally • Drag: pan"
    ZOOM_IN = "Zoom In"
    ZOOM_OUT = "Zoom Out"
    SHOW_CONFIDENCES = "Show Confidences"
    SHOW_CONTOURS = "Show Contours"
    CROSS_PREVIEW = "Cross Preview"

    # Help Menu items
    ABOUT = "About"
    VERSION = "Version:"
    AUTHOR = "Author:"

    # Context Menu items
    CONTEXT_DELETE_IMAGE = "Delete Image"
    CONTEXT_IMPORT_FILES = "Import Files"
    CONTEXT_IMPORT_FOLDER = "Import Folder"
    CONTEXT_CLEAR_GROUP_SELECT = "Clear Group Selection"
    CONTOUR_PROPERTIES = "Properties…"
    CONTOUR_PROPERTIES_TITLE = "Contour properties"
    PROPERTY = "Property"
    CONTOUR_NUMBER = "Contour {number}"
    DETECTOR_CONFIDENCE = "Detector confidence"
    MODEL_NICE_PROBABILITY = "Model probability: nice"
    CURRENT_CLASSIFICATION = "Current classification"
    MODEL_CLASSIFICATION = "Original model classification"
    MANUAL_OVERRIDE = "Manual override"
    REFINED_AREA = "Refined area (mm²)"
    CLASS_NICE = "Nice"
    CLASS_NOT_NICE = "Not nice"
    CLASS_UNKNOWN = "Unknown"
    NOT_AVAILABLE = "Not available"
    NO_OVERRIDE = "None"

    # File dialog
    SELECT_IMAGES = "Select Images"
    SELECT_FOLDER = "Select Folder"

    # Placeholders
    IMAGE_LIST = "Image List"
    IMAGE_PREVIEW = "Image Preview"

    # Error/Warning messages
    IMAGE_LOAD_ERROR_TITLE = "Image Load Error"
    ALREADY_IMPORTED_TITLE = "Already Imported"
    ALREADY_IMPORTED_MESSAGE = "Some files are already imported and were skipped."
    IMAGE_LOAD_ERROR_MESSAGE = "Failed to load image: {file_path}"

    MODEL_CHANGE_SAVE_FAILED = "Could not save prediction invalidation"
    SELECT_MODEL = "Select model"
    DETECTOR_LABEL = "Detector"
    DETECTOR_M = "M — medium (default)"
    DETECTOR_S = "S — small, faster"
    DETECTOR_TOOLTIP = "Changing the detector marks all loaded images as needing prediction again."
    REPREDICT = "Re-predict current image…"
    REPREDICT_CONFIRM = "Replace this image's annotations using the selected detector? A backup of the saved annotations will be kept."
    EXECUTION_STATUS = "{device} · {workers} image worker(s)"
    CPU_FALLBACK_STATUS = "CPU fallback · {workers} image worker(s)"
