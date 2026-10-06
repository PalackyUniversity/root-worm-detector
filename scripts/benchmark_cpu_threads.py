"""Compare CPU thread budgets without altering detector batching or precision."""
import json
import pickle
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from logic.inference_pipeline import InferencePipeline
from scripts.benchmark_pipeline import compare, differences
import pandas as pd

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, nargs='+', default=[1,2,4])
    parser.add_argument('--models', nargs='+', default=['s','m'])
    parser.add_argument('--name', default='cpu_threads')
    args=parser.parse_args()
    out=ROOT/'outputs/adaptive_inference_benchmark';rows=[]
    plate=pd.read_csv(ROOT/'train/runs/5wpi_nice_area/plates_5wpi.csv').sort_values('detected_count').iloc[89]
    path=ROOT/'data-test'/plate.folder/plate.file
    for model in args.models:
        with (out/'before'/f'cpu_{model}_{plate.file}.pkl').open('rb') as f:reference=pickle.load(f)
        for threads in args.threads:
            pipeline=InferencePipeline('cpu',model_id=model,cpu_threads=threads)
            start=time.perf_counter();result=pipeline.predict(path);elapsed=time.perf_counter()-start
            rows.append(dict(model=model,threads=threads,seconds=elapsed,parity=compare(reference,result),differences=differences(reference,result)))
            (out/(args.name+'.json')).write_text(json.dumps(rows,indent=2));print(model,threads,round(elapsed,3),rows[-1]['parity'],flush=True)
            del pipeline
if __name__=='__main__':main()
