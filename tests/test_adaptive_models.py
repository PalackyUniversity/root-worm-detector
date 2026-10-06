import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from config.model import Model
from logic.image_logic import ImageLogic

class ModelTests(unittest.TestCase):
    def test_default_medium_and_distinct_pipeline_identity(self):
        from logic.model_registry import model_spec
        self.assertEqual(Model.DEFAULT_MODEL, 'm')
        self.assertEqual(model_spec('s')['pipeline'], Model.PIPELINE_ID)
        self.assertNotEqual(model_spec('m')['pipeline'], model_spec('s')['pipeline'])
        self.assertEqual(model_spec('m')['confidence'], .35)
        with self.assertRaises(ValueError):model_spec('unknown')

    def test_both_models_sidecars_survive_default_change(self):
        from logic.model_registry import model_spec
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plate.png';Image.new('RGB',(20,20)).save(path, dpi=(600, 600))
            for name in ['s','m']:
                spec=model_spec(name)
                data=dict(path=str(path),contours=[],scores=[],measurements=[],predicted=True,pipeline=spec['pipeline'],model_id=name)
                ImageLogic.save_image_data(data)
                loaded=ImageLogic.load_image(str(path))
                self.assertTrue(loaded['predicted']);self.assertEqual(loaded['model_id'],name)

    def test_decode_shared_only_for_unrotated_inputs(self):
        from logic.measurement import read_prediction_images
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'a.tif'
            pixels=np.zeros((12,20,3),np.uint8);pixels[:,:,0]=130
            Image.fromarray(pixels).save(path, dpi=(600, 600))
            detection,bgr=read_prediction_images(path)
            self.assertEqual(detection.size,(20,12))
            np.testing.assert_array_equal(bgr[:,:,2],pixels[:,:,0])
            exif=Image.Exif();exif[274]=6
            Image.fromarray(pixels).save(path,exif=exif)
            detection,bgr=read_prediction_images(path)
            self.assertIsInstance(detection,str) # retain existing SAHI orientation handling

    def test_conflicting_provenance_is_not_accepted_as_current(self):
        import json
        from logic.model_registry import model_spec
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plate.png';Image.new('RGB',(20,20)).save(path, dpi=(600, 600))
            data=dict(path=str(path),contours=[],scores=[],measurements=[],predicted=True,pipeline=model_spec('m')['pipeline'],model_id='m')
            ImageLogic.save_image_data(data);sidecar=Path(str(path)+'_contours.json')
            original=json.loads(sidecar.read_text())
            for field,value in [('confidence',.1),('models',model_spec('s')['hashes']),('model_id','s')]:
                payload=json.loads(json.dumps(original));payload.setdefault('provenance',{})[field]=value
                sidecar.write_text(json.dumps(payload))
                self.assertFalse(ImageLogic.load_image(str(path))['predicted'],field)
