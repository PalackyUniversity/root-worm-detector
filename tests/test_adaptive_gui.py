import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

class AdaptiveGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def test_menu_model_change_invalidates_and_persists_all_predictions(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QComboBox
        from logic.image_logic import ImageLogic
        from config.strings import Strings
        with tempfile.TemporaryDirectory() as folder:
            paths=[]
            for name in ('a', 'b'):
                path=Path(folder)/(name+'.png')
                Image.new('RGB',(20,20)).save(path)
                paths.append(str(path))
            window=MainWindow()
            try:
                window.load_files(paths)
                records=window._MainWindow__image_data
                for data in records:
                    data['predicted']=True
                    ImageLogic.save_image_data(data)
                window.update_image_list()
                self.assertTrue(window.model_actions['m'].isChecked())
                self.assertFalse(window.findChildren(QComboBox))
                menu=next(a.menu() for a in window.menuBar().actions() if a.text()==Strings.MENU_MODEL)
                submenu=next(a.menu() for a in menu.actions() if a.text()==Strings.SELECT_MODEL)
                self.assertIn(window.model_actions['s'],submenu.actions())
                window.model_actions['m'].trigger()
                self.assertTrue(all(d['predicted'] for d in records))
                window.model_actions['s'].trigger()
                self.assertTrue(window.model_actions['s'].isChecked())
                self.assertFalse(window.model_actions['m'].isChecked())
                self.assertTrue(all(not d['predicted'] for d in records))
                self.assertTrue(all(not window.panel_image_list.item(i).data(Qt.UserRole) for i in range(2)))
                self.wait_for_invalidation(window)
                self.assertTrue(all(not ImageLogic.load_image(p)['predicted'] for p in paths))
                self.assertTrue(window.button_predict.isEnabled())
                self.assertFalse(window.menu_export.isEnabled())
            finally:
                window.close();window.deleteLater();self.app.processEvents()

    def wait_for_invalidation(self, window):
        import time
        deadline=time.monotonic()+5
        while window._invalidation_thread is not None and time.monotonic()<deadline:
            self.app.processEvents();time.sleep(.005)
        self.assertIsNone(window._invalidation_thread)

    def test_model_switch_does_not_wait_for_sidecar_io(self):
        import threading,time
        from unittest.mock import patch
        from PySide6.QtCore import QTimer
        from logic.image_logic import ImageLogic
        window=MainWindow()
        release=threading.Event()
        entered=threading.Event()
        threads=[];ticks=[]
        window._MainWindow__image_data=[dict(path='slow.tif', predicted=True, contours=[], scores=[], measurements=[])]
        def slow_save(data):
            threads.append(threading.get_ident());entered.set();release.wait(.5)
        try:
            with patch.object(ImageLogic,'save_image_data',side_effect=slow_save):
                start=time.monotonic()
                window.model_actions['s'].trigger()
                elapsed=time.monotonic()-start
                self.assertLess(elapsed,.15)
                self.assertTrue(entered.wait(1))
                self.assertNotEqual(threads[0],threading.get_ident())
                QTimer.singleShot(0,lambda:ticks.append(True))
                self.app.processEvents()
                self.assertEqual(ticks,[True])
                self.assertFalse(window.button_predict.isEnabled())
                self.assertFalse(window._can_edit_image())
                window.close()
                self.assertTrue(window._close_after_prediction)
                self.assertIsNotNone(window._invalidation_thread)
                release.set()
                self.wait_for_invalidation(window)
                self.assertTrue(window.button_predict.isEnabled())
        finally:
            release.set()
            if getattr(window,'_invalidation_thread',None) is not None:
                self.wait_for_invalidation(window)
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
                self.assertFalse(window.model_action_group.isEnabled())
                deadline=time.monotonic()+5
                while window._prediction_thread is not None and time.monotonic()<deadline:
                    self.app.processEvents();time.sleep(.005)
                self.assertIsNone(window._prediction_thread)
                self.assertEqual([d['marker'] for d in window._MainWindow__image_data],['a','b'])
                self.assertTrue(window.model_action_group.isEnabled())
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
