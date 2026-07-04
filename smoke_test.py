"""Quick correctness check: run every model for 2 epochs on UMLS."""
from common import set_seed
from run_experiments import MODEL_KWARGS, MODELS
from train_eval import KGData, run_one

set_seed()
data = KGData("UMLS")
print(f"UMLS: {data.n_ent} entities, {data.n_rel} relations")

for model_name in MODELS:
    cfg = dict(epochs=2, batch_size=256, neg_k=5, lr=1e-3, val_every=1,
               steps_per_epoch=5, model_kwargs=MODEL_KWARGS[model_name])
    run_one(model_name, data, cfg)
print("SMOKE TEST PASSED")
