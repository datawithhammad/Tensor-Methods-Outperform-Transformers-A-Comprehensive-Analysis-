# Tensor Methods Outperform Transformers: A Comprehensive Comparative Analysis of Knowledge Graph Link Prediction Methods

Official code and experimental results for the ICCTA 2025 paper by
**Manal Helal** and **Hammad Khawaja** (School of Physics, Engineering and
Computer Science, University of Hertfordshire).

Seven link-prediction methods spanning four paradigms — classical embeddings,
tensor factorisation, graph neural networks, and transformers — are evaluated
on three benchmark knowledge graphs (UMLS, FB15K, DB15K) under a single
standardised protocol, with accuracy, efficiency, and memory measured for
every run.

## Results at a glance

Filtered MRR on the full test sets (both prediction directions):

| Model | Category | UMLS | FB15K | DB15K | Mean |
|---|---|---|---|---|---|
| **Tensor-Tucker** | Tensor | 0.920 | 0.449 | **0.355** | **0.575** |
| LP-BERT | Transformer | **0.930** | 0.365 | 0.311 | 0.535 |
| ComplEx | Classical | 0.889 | **0.484** | 0.207 | 0.527 |
| Tensor-CP | Tensor | 0.918 | 0.383 | 0.262 | 0.521 |
| DistMult | Classical | 0.687 | 0.362 | 0.170 | 0.406 |
| TransE | Classical | 0.602 | 0.212 | 0.163 | 0.326 |
| GNN (R-GCN) | GNN | 0.753 | 0.141 | 0.068 | 0.320 |

Full metrics (Hits@1/3/10, training and inference time, parameter counts,
peak memory) are in [`results/RESULTS.md`](results/RESULTS.md) and
[`results/results_table.csv`](results/results_table.csv); the paper's six
figures are in [`figures/`](figures).

## Repository layout

| File | Purpose |
|---|---|
| `download_data.py` | Downloads UMLS, FB15K, DB15K (with mirror fallbacks) |
| `preprocess.py` | Dedupe, filter rare entities/relations (<5 occurrences), 80/10/10 split, transductive fix |
| `models.py` | The 7 models: TransE, DistMult, ComplEx, Tensor-CP, Tensor-Tucker (TuckER), R-GCN GNN, LP-BERT-style transformer |
| `train_eval.py` | Training loops (uniform negative sampling / 1-vs-All softmax) + filtered MRR & Hits@{1,3,10} evaluation |
| `run_experiments.py` | Runs the full 7×3 grid; one JSON per run in `results/` (crash-resilient, resumable) |
| `make_figures.py` | Generates the paper's 6 figures + results tables |
| `smoke_test.py` | 2-epoch sanity check of all models |

## Reproducing the results

```bash
pip install -r requirements.txt
python download_data.py
python preprocess.py
python run_experiments.py            # full grid (several hours on a 4-core CPU)
python make_figures.py
```

Random seed is fixed (42) throughout — downloads, splits, and training are
reproducible end to end. Subsets can be run with, e.g.:

```bash
python run_experiments.py --datasets UMLS FB15K --models Tensor-Tucker LP-BERT
```

## Method notes

- **Evaluation** is standard filtered ranking over all entities, corrupting
  both heads and tails, on the full test split. MRR and Hits@K are averaged
  over both directions.
- **Classical models + GNN** train with uniform corruption negative sampling.
  **Tensor models and the transformer** train with 1-vs-All softmax
  cross-entropy over all entities and reciprocal relations (head prediction
  for `(?, r, t)` is tail prediction for `(t, r⁻¹, ?)`), as in Lacroix et
  al. (2018) — the common 1-N mean-BCE recipe needs hundreds of epochs to
  converge at 10k+ entities, far outside a CPU budget.
- **Tensor-Tucker** is TuckER (Balažević et al., 2019) with batch-norm and
  dropout. **GNN** is an R-GCN encoder (basis decomposition, sparse message
  passing over train + inverse edges) with a DistMult decoder.
- **LP-BERT** is a BERT-architecture transformer encoder over
  `[CLS, head, relation]` token sequences trained from scratch — no
  pretrained language model, keeping every method on identical training data.
- **CPU budget**: epochs and per-epoch step caps are scaled per dataset (see
  `CONFIGS` in `run_experiments.py`); every result JSON records the exact
  configuration used. The best checkpoint is selected by validation MRR.

## Datasets (after preprocessing)

| Dataset | Domain | Entities | Relations | Train | Valid | Test |
|---|---|---|---|---|---|---|
| UMLS | Biomedical | 134 | 40 | 5,208 | 651 | 651 |
| FB15K | General (Freebase) | 14,361 | 1,025 | 471,983 | 58,997 | 58,999 |
| DB15K | General (DBpedia/MMKG) | 10,007 | 181 | 64,851 | 8,106 | 8,107 |

## Citation

```bibtex
@inproceedings{helal2025tensor,
  title     = {Tensor Methods Outperform Transformers: A Comprehensive
               Comparative Analysis of Knowledge Graph Link Prediction Methods},
  author    = {Helal, Manal and Khawaja, Hammad},
  booktitle = {Proceedings of the International Conference on Computer
               Theory and Applications (ICCTA)},
  year      = {2025}
}
```

## License

Released under the [MIT License](LICENSE).
