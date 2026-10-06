"""Benchmark the production scheduler; preserves all original image sidecars."""
import argparse
import json
import pickle
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from logic.adaptive_prediction import AdaptivePredictor
from scripts.benchmark_pipeline import compare,differences


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--models',nargs='+',default=['s','m'])
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--workers',type=int)
    p.add_argument('--limit',type=int,default=3)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/adaptive_inference_benchmark/integrated_gpu')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    plates=pd.read_csv(ROOT/'train/runs/5wpi_nice_area/plates_5wpi.csv').sort_values('detected_count')
    if args.limit==3:plates=plates.iloc[[0,89,149]]
    else:plates=plates.iloc[__import__('numpy').linspace(0,len(plates)-1,args.limit).astype(int)]
    paths=[str(ROOT/'data-test'/r.folder/r.file) for r in plates.itertuples()]
    all_rows=[]
    for model in args.models:
        engine=AdaptivePredictor(device=args.device,max_workers=args.workers,cpu_threads=args.threads)
        try:
            for repeat in range(args.repeats):
                records=[];statuses=[]
                def finished(path,result):
                    prefix='cpu' if args.device=='cpu' else 'gpu'
                    reference_path=ROOT/'outputs/adaptive_inference_benchmark/before'/f'{prefix}_{model}_{Path(path).name}.pkl'
                    entry=dict(file=Path(path).name,count=len(result['scores']))
                    if reference_path.exists():
                        with reference_path.open('rb') as f:reference=pickle.load(f)
                        entry.update(parity=compare(reference,result),differences=differences(reference,result))
                    with (args.output/f'{model}_{Path(path).name}.pkl').open('wb') as f:pickle.dump(result,f)
                    records.append(entry)
                start=time.perf_counter()
                engine.run(paths,model,on_result=finished,on_status=lambda d,w,f:statuses.append(dict(device=d,workers=w,fallback=f)))
                elapsed=time.perf_counter()-start
                row=dict(model=model,device=args.device,threads=args.threads,repeat=repeat,seconds=elapsed,seconds_per_image=elapsed/len(paths),statuses=statuses,images=records)
                all_rows.append(row);(args.output/'measurements.json').write_text(json.dumps(all_rows,indent=2))
                print(model,args.device,repeat,round(elapsed,3),statuses,flush=True)
        finally:engine.close()
if __name__=='__main__':main()
