"""Preprocess the raw triples per the paper's protocol (Section III.B):

1. Pool all raw triples and deduplicate.
2. Iteratively filter rare entities/relations (< 5 occurrences).
3. Random 80/10/10 train/validation/test split (seed 42).
4. Move validation/test triples whose entity or relation is unseen in
   train back into train (required for transductive link prediction).

Outputs per dataset: train.txt / valid.txt / test.txt (TSV) + stats.json.
"""
import json
import os
import random
from collections import Counter

from common import DATA_DIR, SEED

MIN_OCCURRENCES = 5


def load_raw(name):
    path = os.path.join(DATA_DIR, name, "raw_triples.txt")
    triples = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) == 3:
                triples.add(tuple(cols))
    return list(triples)


def filter_rare(triples):
    while True:
        ent_count = Counter()
        rel_count = Counter()
        for h, r, t in triples:
            ent_count[h] += 1
            ent_count[t] += 1
            rel_count[r] += 1
        kept = [
            (h, r, t)
            for h, r, t in triples
            if ent_count[h] >= MIN_OCCURRENCES
            and ent_count[t] >= MIN_OCCURRENCES
            and rel_count[r] >= MIN_OCCURRENCES
        ]
        if len(kept) == len(triples):
            return kept
        triples = kept


def split_triples(triples):
    rng = random.Random(SEED)
    triples = sorted(triples)
    rng.shuffle(triples)
    n = len(triples)
    n_train, n_valid = int(0.8 * n), int(0.1 * n)
    train = triples[:n_train]
    valid = triples[n_train : n_train + n_valid]
    test = triples[n_train + n_valid :]

    # transductive fix: eval triples must only mention train entities/relations
    while True:
        ents = {e for h, _, t in train for e in (h, t)}
        rels = {r for _, r, _ in train}
        moved = False
        for split in (valid, test):
            keep = []
            for h, r, t in split:
                if h in ents and t in ents and r in rels:
                    keep.append((h, r, t))
                else:
                    train.append((h, r, t))
                    moved = True
            split[:] = keep
        if not moved:
            return train, valid, test


def preprocess(name):
    raw = load_raw(name)
    filtered = filter_rare(raw)
    train, valid, test = split_triples(filtered)

    out_dir = os.path.join(DATA_DIR, name)
    for split_name, split in [("train", train), ("valid", valid), ("test", test)]:
        with open(os.path.join(out_dir, f"{split_name}.txt"), "w", encoding="utf-8") as f:
            f.writelines(f"{h}\t{r}\t{t}\n" for h, r, t in split)

    ents = {e for h, _, t in train for e in (h, t)}
    rels = {r for _, r, _ in train}
    stats = {
        "raw_triples": len(raw),
        "after_filtering": len(filtered),
        "entities": len(ents),
        "relations": len(rels),
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[{name}] {stats}")


if __name__ == "__main__":
    for name in ["UMLS", "FB15K", "DB15K"]:
        preprocess(name)
