"""Real M inference, sidecar reload and selector preservation on a disposable copy."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from PySide6.QtWidgets import QApplication
    from logic.image_logic import ImageLogic
    from ui.main_window import MainWindow
    import pandas as pd
    app = QApplication([])
    plate = pd.read_csv(ROOT/'train/runs/5wpi_nice_area/plates_5wpi.csv').sort_values('detected_count').iloc[89]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/plate.file
        shutil.copyfile(ROOT/'data-test'/plate.folder/plate.file, path)
        window = MainWindow()
        try:
            window.load_files([str(path)])
            assert window.model_selector.currentData() == 'm'
            window.start_prediction()
            deadline = time.monotonic()+180
            while window._prediction_thread is not None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.01)
            assert window._prediction_thread is None, 'Prediction timed out'
            data = window._MainWindow__image_data[0]
            assert data['predicted'] and data['model_id'] == 'm'
            sidecar = Path(str(path)+'_contours.json')
            saved = sidecar.read_bytes()
            reloaded = ImageLogic.load_image(str(path))
            assert reloaded['predicted'] and reloaded['model_id'] == 'm'
            assert reloaded['measurements'] == data['measurements']
            window.model_selector.setCurrentIndex(window.model_selector.findData('s'))
            assert sidecar.read_bytes() == saved and data['model_id'] == 'm'
            from logic.export_logic import ExportLogic
            export = Path(folder)/'results.xlsx'
            ExportLogic.export_data(str(export), {'individual': True}, [data])
            exported = pd.read_excel(export, sheet_name='Images')
            assert exported.iloc[0]['Detector Model'] == 'm'
            assert exported.iloc[0]['Pipeline'] == data['pipeline']
            report = dict(model='m', detected=len(data['scores']), reload=True, selector_preserves_results=True, export=True)
            (ROOT/'outputs/adaptive_inference_benchmark/gui_smoke.json').write_text(json.dumps(report, indent=2))
            print(report)
        finally:
            window.close()
            window._shutdown_prediction()


if __name__ == '__main__':
    main()
