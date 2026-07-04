"""Download the three benchmark knowledge graphs used in the paper.

- UMLS  : biomedical KG (Kok & Domingos splits, standard benchmark)
- FB15K : Freebase subset (Bordes et al., 2013)
- DB15K : DBpedia subset from the MMKG project (Liu et al., 2019)

Each dataset ends up in data/<name>/raw_triples.txt as tab-separated
(head, relation, tail) lines. Splitting/filtering happens later in
preprocess.py so that all three datasets get the identical 80/10/10
protocol described in the paper.
"""
import os
import sys
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Multiple mirrors per file so a dead link does not kill the pipeline.
SOURCES = {
    "UMLS": [
        # villmow/datasets_knowledge_embedding mirror of the standard splits
        [
            "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/UMLS/train.txt",
            "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/UMLS/valid.txt",
            "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/UMLS/test.txt",
        ],
        [
            "https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/UMLS/train.txt",
            "https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/UMLS/valid.txt",
            "https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/UMLS/test.txt",
        ],
    ],
    "FB15K": [
        [
            "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15K/train.txt",
            "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15K/valid.txt",
            "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15K/test.txt",
        ],
        [
            "https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/FB15k/valid.txt",
            "https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/FB15k/test.txt",
        ],
    ],
    "DB15K": [
        [
            "https://raw.githubusercontent.com/mniepert/mmkb/master/DB15K/DB15K_EntityTriples.txt",
        ],
        [
            "https://raw.githubusercontent.com/nle-ml/mmkb/master/DB15K/DB15K_EntityTriples.txt",
        ],
    ],
}


def fetch(url):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.text


def download_dataset(name, mirrors):
    out_dir = os.path.join(DATA_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "raw_triples.txt")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[{name}] already downloaded, skipping")
        return True
    for mirror in mirrors:
        try:
            parts = [fetch(u) for u in mirror]
            text = "\n".join(p.strip("\n") for p in parts)
            # DB15K lines end with " ." (N-Triples style); normalise to TSV
            lines = []
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.endswith(" ."):
                    line = line[:-2].strip()
                cols = line.split("\t")
                if len(cols) != 3:
                    cols = line.split()
                if len(cols) != 3:
                    continue
                lines.append("\t".join(cols))
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            print(f"[{name}] downloaded {len(lines)} triples from {mirror[0].split('/')[3]}")
            return True
        except Exception as e:
            print(f"[{name}] mirror failed ({e}); trying next")
    print(f"[{name}] ALL MIRRORS FAILED")
    return False


if __name__ == "__main__":
    ok = all(download_dataset(n, m) for n, m in SOURCES.items())
    sys.exit(0 if ok else 1)
