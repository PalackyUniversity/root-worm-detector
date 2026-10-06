import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

class AdaptiveGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def test_selector_defaults_m_and_switch_does_not_clear_results(self):
        window=MainWindow()
        try:
            self.assertEqual(window.model_selector.currentData(),'m')
            data=dict(path='existing.tif',predicted=True,contours=[],measurements=[],scores=[],pipeline='old')
            window._MainWindow__image_data=[data]
            window.model_selector.setCurrentIndex(window.model_selector.findData('s'))
            self.assertTrue(data['predicted']);self.assertEqual(data['pipeline'],'old')
        finally:
                window.close();window.deleteLater();self.app.processEvents()

    def test_out_of_order_results_keep_image_identity(self):
        import tempfile,time
        from pathlib import Path
        from PIL import Image
        class ReversePredictor:
            def run(self,paths,model,**hooks):
                for path in paths:hooks['on_started'](path)
                for path in reversed(paths):
                    hooks['on_result'](path,dict(contours=[],scores=[],measurements=[],predicted=True,model_id=model,marker=Path(path).stem))
            def close(self):pass
        with tempfile.TemporaryDirectory() as folder:
            paths=[]
            for name in ['a','b']:
                path=Path(folder)/(name+'.png');Image.new('RGB',(20,20)).save(path);paths.append(str(path))
            window=MainWindow();window._predictor=ReversePredictor();window.load_files(paths)
            try:
                window.start_prediction()
                self.assertFalse(window.model_selector.isEnabled())
                deadline=time.monotonic()+5
                while window._prediction_thread is not None and time.monotonic()<deadline:
                    self.app.processEvents();time.sleep(.005)
                self.assertIsNone(window._prediction_thread)
                self.assertEqual([d['marker'] for d in window._MainWindow__image_data],['a','b'])
                self.assertTrue(window.model_selector.isEnabled())
            finally:
                window.close();window.deleteLater();self.app.processEvents()

    def test_reprediction_keeps_backup_of_old_annotations(self):
        import tempfile
        from pathlib import Path
        from ui.prediction_worker import BatchPredictionWorker
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'image.tif');sidecar=Path(path+'_contours.json')
            sidecar.write_text('{"manual":"original"}')
            worker=BatchPredictionWorker([path],'m',None,replace=True)
            worker._save_result(path,dict(contours=[],scores=[],measurements=[],predicted=True))
            backups=list(Path(directory).glob('*.before-repredict-*.bak'))
            self.assertEqual(len(backups),1);self.assertEqual(backups[0].read_text(),'{"manual":"original"}')
