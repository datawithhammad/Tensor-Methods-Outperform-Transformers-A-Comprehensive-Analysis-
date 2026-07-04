"""Run the full experiment grid: 7 models x 3 datasets on CPU.

Each finished (dataset, model) run is saved immediately to
results/<dataset>_<model>.json so the suite is crash-resilient and
re-runnable (existing results are skipped unless --force).

Epochs/batch sizes are scaled per dataset so the whole suite fits a
4-core CPU budget; steps_per_epoch subsamples training batches per epoch
on the larger graphs. All of this is recorded in the result files.
"""
import argparse
import functools
import json
import os
import sys
import time

from common import RESULTS_DIR, set_seed
from train_eval import KGData, run_one

DATASETS = ["UMLS", "DB15K", "FB15K"]
MODELS = ["TransE", "DistMult", "ComplEx", "Tensor-CP", "Tensor-Tucker", "GNN", "LP-BERT"]

# ---- per-dataset, per-model training configuration ------------------------ #
CONFIGS = {
    "UMLS": {
        "TransE":        dict(epochs=300, batch_size=512, neg_k=10, lr=1e-3, val_every=30),
        "DistMult":      dict(epochs=300, batch_size=512, neg_k=10, lr=1e-3, val_every=30),
        "ComplEx":       dict(epochs=300, batch_size=512, neg_k=10, lr=1e-3, val_every=30),
        "Tensor-CP":     dict(epochs=200, batch_size=128, lr=1e-3, lr_decay=0.995, val_every=20),
        "Tensor-Tucker": dict(epochs=200, batch_size=128, lr=1e-3, lr_decay=0.995, val_every=20),
        "GNN":           dict(epochs=150, batch_size=1024, neg_k=10, lr=5e-3, val_every=15,
                              steps_per_epoch=5),
        "LP-BERT":       dict(epochs=150, batch_size=128, lr=5e-4, val_every=15),
    },
    "DB15K": {
        "TransE":        dict(epochs=80, batch_size=1024, neg_k=10, lr=1e-3, val_every=10, val_sample=1500),
        "DistMult":      dict(epochs=80, batch_size=1024, neg_k=10, lr=1e-3, val_every=10, val_sample=1500),
        "ComplEx":       dict(epochs=80, batch_size=1024, neg_k=10, lr=1e-3, val_every=10, val_sample=1500),
        "Tensor-CP":     dict(epochs=15, batch_size=512, lr=1e-3, lr_decay=0.995, val_every=3, val_sample=1500),
        "Tensor-Tucker": dict(epochs=15, batch_size=512, lr=1e-3, lr_decay=0.995, val_every=3, val_sample=1500),
        "GNN":           dict(epochs=25, batch_size=8192, neg_k=5, lr=5e-3, val_every=5,
                              steps_per_epoch=20, val_sample=1500),
        "LP-BERT":       dict(epochs=12, batch_size=512, lr=5e-4, val_every=3, val_sample=1500),
    },
    "FB15K": {
        "TransE":        dict(epochs=30, batch_size=2048, neg_k=10, lr=1e-3, val_every=5, val_sample=1000),
        "DistMult":      dict(epochs=30, batch_size=2048, neg_k=10, lr=1e-3, val_every=5, val_sample=1000),
        "ComplEx":       dict(epochs=30, batch_size=2048, neg_k=10, lr=1e-3, val_every=5, val_sample=1000),
        "Tensor-CP":     dict(epochs=10, batch_size=1024, lr=1e-3, lr_decay=0.995, val_every=2,
                              steps_per_epoch=400, val_sample=1000),
        "Tensor-Tucker": dict(epochs=10, batch_size=1024, lr=1e-3, lr_decay=0.995, val_every=2,
                              steps_per_epoch=400, val_sample=1000),
        "GNN":           dict(epochs=12, batch_size=8192, neg_k=5, lr=5e-3, val_every=3,
                              steps_per_epoch=25, val_sample=1000),
        "LP-BERT":       dict(epochs=8, batch_size=1024, lr=5e-4, val_every=2,
                              steps_per_epoch=300, val_sample=1000),
    },
}

MODEL_KWARGS = {
    "TransE": dict(dim=100),
    "DistMult": dict(dim=100),
    "ComplEx": dict(dim=100),
    "Tensor-CP": dict(dim=100),
    "Tensor-Tucker": dict(de=100, dr=50),
    "GNN": dict(dim=64, n_bases=8),
    "LP-BERT": dict(dim=128, n_layers=2, n_heads=4, ff=256),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    log = functools.partial(print, flush=True)
    failures = []

    for ds in args.datasets:
        log(f"\n===== {ds} =====")
        data = KGData(ds)
        log(f"[{ds}] {data.n_ent} entities, {data.n_rel} relations, "
            f"{data.train.shape[0]}/{data.valid.shape[0]}/{data.test.shape[0]} triples")
        for model_name in args.models:
            out_path = os.path.join(RESULTS_DIR, f"{ds}_{model_name}.json")
            if os.path.exists(out_path) and not args.force:
                log(f"[{ds}/{model_name}] result exists, skipping")
                continue
            cfg = dict(CONFIGS[ds][model_name])
            cfg["model_kwargs"] = MODEL_KWARGS[model_name]
            try:
                t0 = time.perf_counter()
                result = run_one(model_name, data, cfg, log=log)
                result["wall_time_sec"] = round(time.perf_counter() - t0, 2)
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
            except Exception as e:
                import traceback
                traceback.print_exc()
                failures.append((ds, model_name, str(e)))
                log(f"[{ds}/{model_name}] FAILED: {e}")

    log("\n===== DONE =====")
    if failures:
        log(f"Failures: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    set_seed()
    main()
