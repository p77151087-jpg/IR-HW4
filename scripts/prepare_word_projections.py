"""Prepare PCA and t-SNE views ahead of a demo, using verified saved runs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ir_hw4.experiment_ui import REFERENCE_RUNS
from ir_hw4.experiments import ARCHITECTURES, read_experiment
from ir_hw4.io import sha256
from ir_hw4.word_projection import (
    PROJECTION_METHODS, get_vocabulary_projection, projection_cache_path, read_cached_projection,
)


def prepare_model(name, architecture, methods):
    start = perf_counter()
    run, _, _, _ = read_experiment(name)
    signature = sha256(run / "manifest.json")
    vectors = np.load(run / architecture / "vectors.npy", allow_pickle=False)
    for method in methods:
        get_vocabulary_projection(signature, architecture, method, vectors)
        path = projection_cache_path(signature, architecture, method)
        if read_cached_projection(path, vectors, method) is None:
            raise RuntimeError(f"Could not prepare {method} for {name} {architecture}")
    return f"{name} {architecture}: {', '.join(methods)} ready ({perf_counter() - start:.1f}s)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", help="Names of completed model runs")
    parser.add_argument("--all-reference", action="store_true", help="Prepare all 18 reference settings")
    parser.add_argument("--method", choices=("all", *PROJECTION_METHODS), default="all")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    if args.all_reference == bool(args.runs):
        parser.error("Specify run names or --all-reference")
    names = REFERENCE_RUNS if args.all_reference else dict.fromkeys(args.runs)
    methods = PROJECTION_METHODS if args.method == "all" else (args.method,)
    tasks = [(name, architecture, methods) for name in names for architecture in ARCHITECTURES]
    if args.workers == 1:
        for task in tasks:
            print(prepare_model(*task), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(prepare_model, *task) for task in tasks]
            for future in as_completed(futures):
                print(future.result(), flush=True)


if __name__ == "__main__":
    main()
