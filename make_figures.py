"""Generate the paper's six figures plus results tables from results/*.json.

Fig 1  MRR performance comparison across all methods and datasets (bars)
Fig 2  Heatmap of MRR across models and datasets
Fig 3  Overall model ranking based on mean MRR
Fig 4  Training time comparison (log scale)
Fig 5  Performance-efficiency trade-off (MRR vs training time)
Fig 6  Multi-dimensional radar plot

Also writes results/results_table.csv and results/RESULTS.md.
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES_DIR, RESULTS_DIR

MODEL_ORDER = ["TransE", "DistMult", "ComplEx", "Tensor-CP", "Tensor-Tucker", "GNN", "LP-BERT"]
DATASET_ORDER = ["UMLS", "FB15K", "DB15K"]
CATEGORY_COLORS = {
    "Classical": "#4C72B0",
    "Tensor": "#C44E52",
    "GNN": "#55A868",
    "Transformer": "#8172B2",
}

plt.rcParams.update({"font.size": 10, "figure.dpi": 120, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})


def load_results():
    rows = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "*_*.json")):
        with open(path) as f:
            rows.append(json.load(f))
    df = pd.DataFrame(rows)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df["dataset"] = pd.Categorical(df["dataset"], DATASET_ORDER, ordered=True)
    return df.sort_values(["dataset", "model"]).reset_index(drop=True)


def colors_for(models, df):
    cat = df.drop_duplicates("model").set_index("model")["category"]
    return [CATEGORY_COLORS[cat[m]] for m in models]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"saved {name}")


def fig1_bars(df):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    x = np.arange(len(models))
    width = 0.25
    hatches = ["", "//", ".."]
    for i, ds in enumerate(DATASET_ORDER):
        sub = df[df["dataset"] == ds].set_index("model")["mrr"]
        vals = [sub.get(m, np.nan) for m in models]
        ax.bar(x + (i - 1) * width, vals, width, label=ds,
               color=colors_for(models, df), alpha=1 - 0.28 * i, hatch=hatches[i],
               edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("MRR (filtered)")
    ax.set_title("MRR performance comparison across all methods and datasets")
    ax.legend(title="Dataset")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig1_mrr_comparison")


def fig2_heatmap(df):
    pivot = df.pivot_table(index="model", columns="dataset", values="mrr", observed=True)
    pivot = pivot.reindex(index=MODEL_ORDER, columns=DATASET_ORDER)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < np.nanmax(pivot.values) * 0.6 else "black",
                        fontsize=9)
    fig.colorbar(im, label="MRR (filtered)")
    ax.set_title("MRR heatmap across models and datasets")
    save(fig, "fig2_mrr_heatmap")


def fig3_ranking(df):
    mean_mrr = df.groupby("model", observed=True)["mrr"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(mean_mrr.index.astype(str), mean_mrr.values,
            color=colors_for(mean_mrr.index, df))
    for i, v in enumerate(mean_mrr.values):
        ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Mean MRR across datasets")
    ax.set_title("Overall model ranking based on mean MRR")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "fig3_overall_ranking")


def fig4_train_time(df):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    x = np.arange(len(models))
    width = 0.25
    for i, ds in enumerate(DATASET_ORDER):
        sub = df[df["dataset"] == ds].set_index("model")["train_time_sec"]
        vals = [sub.get(m, np.nan) for m in models]
        ax.bar(x + (i - 1) * width, vals, width, label=ds, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("Training time (s, log scale)")
    ax.set_title("Training time comparison across all evaluated methods")
    ax.legend(title="Dataset")
    ax.grid(axis="y", alpha=0.3, which="both")
    save(fig, "fig4_training_time")


def fig5_tradeoff(df):
    agg = df.groupby("model", observed=True).agg(
        mrr=("mrr", "mean"), time=("train_time_sec", "mean"),
        category=("category", "first")).dropna()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for _, row in agg.iterrows():
        ax.scatter(row["time"], row["mrr"], s=110,
                   color=CATEGORY_COLORS[row["category"]], zorder=3)
    for name, row in agg.iterrows():
        ax.annotate(str(name), (row["time"], row["mrr"]),
                    xytext=(6, 5), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Mean training time (s, log scale)")
    ax.set_ylabel("Mean MRR")
    ax.set_title("Performance-efficiency trade-off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k)
               for k, c in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, title="Category")
    ax.grid(alpha=0.3, which="both")
    save(fig, "fig5_tradeoff")


def fig6_radar(df):
    agg = df.groupby("model", observed=True).agg(
        mrr=("mrr", "mean"), hits10=("hits10", "mean"),
        train_time=("train_time_sec", "mean"),
        infer_time=("inference_time_sec", "mean"),
        mem=("peak_mem_mb", "mean")).dropna()

    def norm(s, invert=False):
        s = np.log10(s + 1) if invert else s
        rng = s.max() - s.min()
        z = (s - s.min()) / (rng if rng > 0 else 1)
        return 1 - z if invert else z

    dims = pd.DataFrame({
        "Accuracy\n(MRR)": norm(agg["mrr"]),
        "Hits@10": norm(agg["hits10"]),
        "Training\nefficiency": norm(agg["train_time"], invert=True),
        "Inference\nefficiency": norm(agg["infer_time"], invert=True),
        "Memory\nefficiency": norm(agg["mem"], invert=True),
    })
    angles = np.linspace(0, 2 * np.pi, len(dims.columns), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"projection": "polar"})
    cat = df.drop_duplicates("model").set_index("model")["category"]
    for model in dims.index:
        vals = dims.loc[model].tolist() + [dims.loc[model].iloc[0]]
        ax.plot(angles, vals, label=str(model), color=None, lw=1.6,
                ls="--" if cat[model] == "Classical" else "-")
        ax.fill(angles, vals, alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims.columns, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.set_title("Multi-dimensional comparison (normalised, higher is better)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    save(fig, "fig6_radar")


def write_tables(df):
    cols = ["dataset", "model", "category", "mrr", "hits1", "hits3", "hits10",
            "train_time_sec", "inference_time_sec", "params", "peak_mem_mb", "epochs"]
    table = df[cols].copy()
    table.to_csv(os.path.join(RESULTS_DIR, "results_table.csv"), index=False)

    lines = ["# Experimental Results", "",
             "Filtered MRR / Hits@K, both prediction directions, full test sets.", ""]
    for ds in DATASET_ORDER:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        lines += [f"## {ds}", "",
                  "| Model | MRR | Hits@1 | Hits@3 | Hits@10 | Train (s) | Infer (s) | Params | Peak MB |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['model']} | {r['mrr']:.4f} | {r['hits1']:.4f} | {r['hits3']:.4f} "
                f"| {r['hits10']:.4f} | {r['train_time_sec']:.0f} | {r['inference_time_sec']:.0f} "
                f"| {r['params']:,} | {r['peak_mem_mb']:.0f} |")
        lines.append("")
    mean_mrr = df.groupby("model", observed=True)["mrr"].mean().sort_values(ascending=False)
    lines += ["## Overall ranking (mean MRR)", ""]
    lines += [f"{i+1}. **{m}** — {v:.4f}" for i, (m, v) in enumerate(mean_mrr.items())]
    with open(os.path.join(RESULTS_DIR, "RESULTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("saved results_table.csv and RESULTS.md")


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = load_results()
    if df.empty:
        raise SystemExit("no results found — run run_experiments.py first")
    print(f"loaded {len(df)} results")
    fig1_bars(df)
    fig2_heatmap(df)
    fig3_ranking(df)
    fig4_train_time(df)
    fig5_tradeoff(df)
    fig6_radar(df)
    write_tables(df)
