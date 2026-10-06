import unittest
import threading
from concurrent.futures import Future
from unittest.mock import patch

class InlineExecutor:
    def __init__(self, fn):self.fn=fn;self.closed=False
    def submit(self,fn,path,*args):
        f=Future()
        try:f.set_result(self.fn(path,*args))
        except Exception as e:f.set_exception(e)
        return f
    def shutdown(self,**kwargs):self.closed=True

def success(path,*args):
    return dict(result=dict(path=path),seconds=1,peak_mib=100,rss_mib=100)

class SchedulerTests(unittest.TestCase):
    def engine(self, execute=success, device='cpu'):
        from logic.adaptive_prediction import AdaptivePredictor
        return AdaptivePredictor(device=device, executor_factory=lambda:InlineExecutor(execute))

    def test_every_path_reported_once_and_model_forwarded(self):
        calls=[];done=[]
        def execute(path,model,device,threads):
            calls.append((path,model,device,threads));return success(path)
        engine=self.engine(execute)
        engine.run(['a','b','c'],'m',on_result=lambda p,r:done.append(p))
        self.assertEqual(sorted(done),['a','b','c'])
        self.assertTrue(all(x[1]=='m' for x in calls));engine.close()

    def test_cancel_does_not_dispatch_remaining_images(self):
        cancel=threading.Event();done=[];engine=self.engine();engine.max_workers=1
        def result(path,r):done.append(path);cancel.set()
        engine.run(['a','b','c'],'s',cancel=cancel,on_result=result)
        self.assertEqual(done,['a']);engine.close()

    def test_single_gpu_failure_retries_cpu_same_model(self):
        seen=[]
        def execute(path,model,device,threads):
            seen.append((model,device))
            if device!='cpu':return dict(error='CUDA out of memory',recoverable=True)
            return success(path)
        engine=self.engine(execute,'cuda:0');done=[]
        engine.run(['a'],'m',on_result=lambda p,r:done.append(p))
        self.assertEqual(seen,[('m','cuda:0'),('m','cpu')]);self.assertEqual(done,['a']);engine.close()

    def test_ordinary_errors_never_silently_fall_back(self):
        def execute(*args):return dict(error='Model checksum mismatch',recoverable=False)
        engine=self.engine(execute,'cuda:0')
        with self.assertRaisesRegex(RuntimeError,'checksum'):engine.run(['a'],'s')
        self.assertEqual(engine.device,'cuda:0');engine.close()

    def test_short_tail_does_not_reset_learned_parallelism(self):
        engine=self.engine();engine._model='m';engine._target=3
        with patch('logic.adaptive_prediction.resources',return_value=dict(cores=16,ram_mib=32000,free_gpu_mib=None,total_gpu_mib=0)):
            engine.run(['a','b'],'m')
        self.assertEqual(engine._target,3);engine.close()

    def test_concurrent_oom_drains_successes_then_reduces_without_duplicates(self):
        calls=[];done=[];statuses=[]
        def execute(path,model,device,threads):
            calls.append((path,device))
            if path=='b' and calls.count(('b','cuda:0'))==1:
                return dict(error='CUDA out of memory',recoverable=True)
            return success(path)
        engine=self.engine(execute,'cuda:0')
        with patch('logic.adaptive_prediction.resources',return_value=dict(cores=8,ram_mib=32000,free_gpu_mib=14000,total_gpu_mib=16000)):
            engine.run(['a','b','c','d'],'m',on_result=lambda p,r:done.append(p),on_status=lambda d,w,f:statuses.append(w))
        self.assertEqual(sorted(done),['a','b','c','d']);self.assertEqual(calls.count(('c','cuda:0')),1)
        self.assertEqual(calls.count(('b','cuda:0')),2);self.assertNotIn(('b','cpu'),calls)
        self.assertIn(2,statuses);self.assertEqual(statuses[-1],1);engine.close()

    def test_cancel_drains_only_dispatched_images(self):
        cancel=threading.Event();done=[];engine=self.engine();engine._model='m';engine._target=2
        def finished(path,result):
            self.assertEqual(result['path'],path);done.append(path);cancel.set()
        with patch('logic.adaptive_prediction.resources',return_value=dict(cores=16,ram_mib=32000,free_gpu_mib=None,total_gpu_mib=0)):
            engine.run(['a','b','c','d'],'m',cancel=cancel,on_result=finished)
        self.assertEqual(sorted(done),['a','b']);engine.close()

    def test_tail_oom_releases_idle_gpu_workers_before_cpu_retry(self):
        calls=[]
        def execute(path,model,device,threads):
            calls.append(device)
            if len(calls)==1:return dict(error='CUDA out of memory',recoverable=True)
            return success(path)
        engine=self.engine(execute,'cuda:0');engine._model='m';engine._target=3;engine._resize(3)
        with patch('logic.adaptive_prediction.resources',return_value=dict(cores=8,ram_mib=32000,free_gpu_mib=14000,total_gpu_mib=16000)):
            engine.run(['tail'],'m')
        self.assertEqual(calls,['cuda:0','cuda:0']);engine.close()

    def test_submit_failure_drains_other_results_and_retries_failed_path(self):
        from concurrent.futures.process import BrokenProcessPool
        class BrokenExecutor(InlineExecutor):
            def submit(self,*args):raise BrokenProcessPool('worker died while idle')
        engine=self.engine(device='cuda:0');engine._model='m';engine._target=2
        engine._pools=[InlineExecutor(success),BrokenExecutor(success)];done=[]
        with patch('logic.adaptive_prediction.resources',return_value=dict(cores=8,ram_mib=32000,free_gpu_mib=14000,total_gpu_mib=16000)):
            engine.run(['a','b'],'m',on_result=lambda p,r:done.append(p))
        self.assertEqual(sorted(done),['a','b']);self.assertEqual(len(done),2);engine.close()

    def test_changing_model_retries_requested_gpu_after_fallback(self):
        calls=[]
        def execute(path,model,device,threads):
            calls.append((model,device))
            if model=='m' and device!='cpu':return dict(error='CUDA out of memory',recoverable=True)
            return success(path)
        engine=self.engine(execute,'cuda:0')
        engine.run(['a'],'m');engine.run(['b'],'s')
        self.assertEqual(calls,[('m','cuda:0'),('m','cpu'),('s','cuda:0')]);engine.close()
