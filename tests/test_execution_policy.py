import unittest
from unittest.mock import patch

class PolicyTests(unittest.TestCase):
    def test_gpu_limit_uses_free_memory_with_headroom(self):
        from logic.execution_policy import worker_limit
        self.assertEqual(worker_limit('cuda:0',8,8*1024,700,16000,1000),1)
        self.assertLessEqual(worker_limit('cuda:0',8,8*1024,3000,16000,1000),2)
        self.assertGreater(worker_limit('cuda:0',8,8*1024,12000,16000,1000),2)

    def test_cpu_respects_cores_and_ram(self):
        from logic.execution_policy import worker_limit
        self.assertEqual(worker_limit('cpu',1,8000,None,0,500),1)
        self.assertEqual(worker_limit('cpu',8,600,None,0,1000),1)
        self.assertLessEqual(worker_limit('cpu',8,16000,None,0,1000),7)

    def test_errors_classified_narrowly(self):
        from logic.execution_policy import recoverable_accelerator_error
        self.assertTrue(recoverable_accelerator_error(RuntimeError('CUDA out of memory')))
        self.assertTrue(recoverable_accelerator_error(RuntimeError('no kernel image is available for execution on the device')))
        self.assertFalse(recoverable_accelerator_error(ValueError('Model checksum mismatch')))
        self.assertFalse(recoverable_accelerator_error(FileNotFoundError('image.tif')))
        self.assertFalse(recoverable_accelerator_error(RuntimeError('invalid feature dimensions')))

    def test_cpu_fallback_when_cuda_probe_fails(self):
        from logic.execution_policy import available_device
        with patch('torch.cuda.is_available',side_effect=RuntimeError('driver failed')), patch('torch.backends.mps.is_available',return_value=False):
            self.assertEqual(available_device(),'cpu')

    def test_cpu_allocator_errors_are_recoverable_memory_pressure(self):
        from logic.execution_policy import recoverable_memory_error
        self.assertTrue(recoverable_memory_error(RuntimeError("DefaultCPUAllocator: can't allocate memory")))
        self.assertFalse(recoverable_memory_error(RuntimeError('invalid dimensions')))
