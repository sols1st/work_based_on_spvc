"""Rerun current robust certificates, without training or modifying old files."""
import argparse
import json
import os
from pathlib import Path
from .tasks import load_task,NAMES


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=NAMES,required=True)
    p.add_argument('--budget',type=float,default=300.)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import torch
    torch.set_num_threads(1)
    task=load_task(args.task)
    report=task.current_verify(args.budget)
    report['protocol']=dict(task=task.name,budget_seconds=args.budget,threads=1,metadata=task.metadata,
                            unsafe_threshold=task.metadata['unsafe_barrier_threshold'])
    report['fresh_checks']=True
    report.setdefault('status','pass' if report.get('passed') else 'unknown')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(task=task.name,status=report['status'],elapsed_seconds=report['elapsed_seconds'],output=str(args.output))),flush=True)


if __name__=='__main__':main()
