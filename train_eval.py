"""Training loops (negative sampling, 1-N, GNN) and filtered evaluation."""
import copy
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from common import DATA_DIR, RunTracker, set_seed


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class KGData:
    def __init__(self, name):
        self.name = name
        splits = {}
        for split in ("train", "valid", "test"):
            path = os.path.join(DATA_DIR, name, f"{split}.txt")
            with open(path, encoding="utf-8") as f:
                splits[split] = [tuple(l.rstrip("\n").split("\t")) for l in f]
        ents = sorted({e for tr in splits.values() for h, _, t in tr for e in (h, t)})
        rels = sorted({r for tr in splits.values() for _, r, _ in tr})
        self.ent2id = {e: i for i, e in enumerate(ents)}
        self.rel2id = {r: i for i, r in enumerate(rels)}
        self.n_ent, self.n_rel = len(ents), len(rels)

        def to_tensor(triples):
            arr = np.array(
                [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h, r, t in triples],
                dtype=np.int64,
            )
            return torch.from_numpy(arr)

        self.train = to_tensor(splits["train"])
        self.valid = to_tensor(splits["valid"])
        self.test = to_tensor(splits["test"])

        # filters over all splits: (h, r) -> tails ; (t, r) -> heads
        self.tails_of = defaultdict(list)
        self.heads_of = defaultdict(list)
        for split in (self.train, self.valid, self.test):
            for h, r, t in split.tolist():
                self.tails_of[(h, r)].append(t)
                self.heads_of[(t, r)].append(h)

        # 1-vs-All training samples with reciprocal relations:
        # rows are (input_entity, relation, target_entity)
        fwd = self.train
        rev = torch.stack([fwd[:, 2], fwd[:, 1] + self.n_rel, fwd[:, 0]], dim=1)
        self.samples_1n = torch.cat([fwd, rev])


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def uniform_negatives(pos, n_ent, k, rng):
    """Uniform corruption: replace head or tail with a random entity."""
    b = pos.shape[0]
    neg = pos.repeat_interleave(k, dim=0).clone()
    corrupt_head = torch.from_numpy(rng.random(b * k) < 0.5)
    rand_ent = torch.from_numpy(rng.integers(0, n_ent, b * k))
    neg[corrupt_head, 0] = rand_ent[corrupt_head]
    neg[~corrupt_head, 2] = rand_ent[~corrupt_head]
    return neg


def train_ns(model, data, cfg, tracker, log):
    """Negative-sampling training (TransE, DistMult, ComplEx, GNN)."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(42)
    is_gnn = model.family == "gnn"
    n = data.train.shape[0]
    bs, k = cfg["batch_size"], cfg["neg_k"]
    steps_per_epoch = cfg.get("steps_per_epoch") or max(1, n // bs)
    best = {"mrr": -1.0, "state": None}

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(n)
        losses = []
        for step in range(steps_per_epoch):
            idx = perm[(step * bs) % n : (step * bs) % n + bs]
            pos = data.train[idx]
            neg = uniform_negatives(pos, data.n_ent, k, rng)
            loss = model.loss(
                (pos[:, 0], pos[:, 1], pos[:, 2]), (neg[:, 0], neg[:, 1], neg[:, 2])
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            if hasattr(model, "normalize") and not is_gnn:
                model.normalize()
            losses.append(loss.item())
        tracker.sample()
        if epoch % cfg["val_every"] == 0 or epoch == cfg["epochs"]:
            mrr = quick_val_mrr(model, data, cfg)
            log(f"  epoch {epoch}/{cfg['epochs']} loss={np.mean(losses):.4f} val_mrr={mrr:.4f}")
            if mrr > best["mrr"]:
                best = {"mrr": mrr, "state": copy.deepcopy(model.state_dict())}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model


def train_1n(model, data, cfg, tracker, log):
    """1-vs-All softmax training (Tensor-CP, Tensor-Tucker, LP-BERT).

    Full cross-entropy over all entities per (input, relation) query —
    much stronger gradient signal per step than 1-N mean-BCE, which needs
    hundreds of epochs at large entity counts (Lacroix et al., 2018).
    """
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg.get("lr_decay", 1.0))
    n_samples = data.samples_1n.shape[0]
    bs, ls = cfg["batch_size"], cfg.get("label_smoothing", 0.1)
    steps = cfg.get("steps_per_epoch")
    epoch_samples = min(n_samples, steps * bs) if steps else n_samples
    best = {"mrr": -1.0, "state": None}

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(n_samples)  # random subset each epoch if capped
        losses = []
        for start in range(0, epoch_samples, bs):
            batch = data.samples_1n[perm[start : start + bs]]
            logits = model.forward_1n(batch[:, 0], batch[:, 1])
            loss = F.cross_entropy(logits, batch[:, 2], label_smoothing=ls)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        tracker.sample()
        if epoch % cfg["val_every"] == 0 or epoch == cfg["epochs"]:
            mrr = quick_val_mrr(model, data, cfg)
            log(f"  epoch {epoch}/{cfg['epochs']} loss={np.mean(losses):.4f} val_mrr={mrr:.4f}")
            if mrr > best["mrr"]:
                best = {"mrr": mrr, "state": copy.deepcopy(model.state_dict())}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model


# --------------------------------------------------------------------------- #
# Evaluation: filtered ranking (both directions), MRR + Hits@{1,3,10}
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, data, triples, batch_size=256, max_triples=None):
    model.eval()
    if model.family == "gnn":
        model.cache_encoding()
    if max_triples is not None and triples.shape[0] > max_triples:
        g = torch.Generator().manual_seed(42)
        triples = triples[torch.randperm(triples.shape[0], generator=g)[:max_triples]]

    ranks = []
    for start in range(0, triples.shape[0], batch_size):
        batch = triples[start : start + batch_size]
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

        for direction in ("tail", "head"):
            if direction == "tail":
                scores = model.score_tails(h, r)
                target_idx, filt = t, data.tails_of
                keys = list(zip(h.tolist(), r.tolist()))
            else:
                scores = model.score_heads(t, r)
                target_idx, filt = h, data.heads_of
                keys = list(zip(t.tolist(), r.tolist()))
            target = scores.gather(1, target_idx.view(-1, 1)).squeeze(1)
            for i, key in enumerate(keys):
                scores[i, filt[key]] = -1e9
            rank = (scores > target.view(-1, 1)).sum(dim=1) + 1
            ranks.append(rank)

    ranks = torch.cat(ranks).float()
    return {
        "mrr": (1.0 / ranks).mean().item(),
        "hits1": (ranks <= 1).float().mean().item(),
        "hits3": (ranks <= 3).float().mean().item(),
        "hits10": (ranks <= 10).float().mean().item(),
        "n_eval_triples": ranks.shape[0] // 2,
    }


def quick_val_mrr(model, data, cfg):
    return evaluate(model, data, data.valid, max_triples=cfg.get("val_sample", 2000))["mrr"]


# --------------------------------------------------------------------------- #
# Orchestration for a single (model, dataset) run
# --------------------------------------------------------------------------- #
def run_one(model_name, data, cfg, log=print):
    from models import MODEL_REGISTRY

    set_seed()
    category, cls = MODEL_REGISTRY[model_name]
    if model_name == "GNN":
        edges = (data.train[:, 0], data.train[:, 1], data.train[:, 2])
        model = cls(data.n_ent, data.n_rel, edges, **cfg.get("model_kwargs", {}))
    else:
        model = cls(data.n_ent, data.n_rel, **cfg.get("model_kwargs", {}))
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[{data.name}/{model_name}] {n_params:,} params, cfg={ {k: v for k, v in cfg.items() if k != 'model_kwargs'} }")

    tracker = RunTracker()
    trainer = train_1n if model.family == "1n" else train_ns
    trainer(model, data, cfg, tracker, log)
    train_time = tracker.elapsed()

    t0 = time.perf_counter()
    metrics = evaluate(model, data, data.test)
    inference_time = time.perf_counter() - t0
    tracker.sample()

    result = {
        "dataset": data.name,
        "model": model_name,
        "category": category,
        **metrics,
        "train_time_sec": round(train_time, 2),
        "inference_time_sec": round(inference_time, 2),
        "params": n_params,
        "peak_mem_mb": round(tracker.peak_mb, 1),
        "epochs": cfg["epochs"],
    }
    log(f"[{data.name}/{model_name}] TEST mrr={metrics['mrr']:.4f} "
        f"h@1={metrics['hits1']:.4f} h@10={metrics['hits10']:.4f} "
        f"train={train_time:.0f}s infer={inference_time:.0f}s")
    return result
