import json
import random
from pathlib import Path

rng = random.Random(314159)

N_TRIALS = 24
OUT = Path("hpo_trials.jsonl")

losses = [
    "weighted_logistic",
    "logistic",
    "hinge",
    "margin_ranking",
]

epsilons = [0.05, 0.1, 0.2, 0.3, 0.5]
scales = [5.0, 10.0, 20.0, 30.0, 40.0]
learning_rates = [5e-6, 1e-5, 2e-5, 3e-5, 1e-4]
warmup_ratios = [0.03, 0.06, 0.1]
weight_decays = [0.0, 0.01, 0.05]
loss_normalizations = ["items", "batch"]

batch_options = [
    {"per_device_train_batch_size": 4, "gradient_accumulation_steps": 4},
    {"per_device_train_batch_size": 8, "gradient_accumulation_steps": 2},
    {"per_device_train_batch_size": 16, "gradient_accumulation_steps": 1},
]

configs = []
seen = set()

while len(configs) < N_TRIALS:
    loss = rng.choice(losses)

    cfg = {
        "trial_id": len(configs),
        "loss": loss,
        "epsilon": rng.choice(epsilons),
        "scale": rng.choice(scales),
        "learning_rate": rng.choice(learning_rates),
        "warmup_ratio": rng.choice(warmup_ratios),
        "weight_decay": rng.choice(weight_decays),
        "loss_normalization": rng.choice(loss_normalizations),
        "num_train_epochs": 1,
    }

    cfg.update(rng.choice(batch_options))

    key = tuple(sorted(cfg.items()))
    if key in seen:
        continue

    seen.add(key)

    cfg["trial_name"] = (
        f"trial_{cfg['trial_id']:03d}"
        f"_loss-{cfg['loss']}"
        f"_eps-{cfg['epsilon']}"
        f"_scale-{cfg['scale']}"
        f"_lr-{cfg['learning_rate']}"
        f"_bs-{cfg['per_device_train_batch_size']}"
        f"_ga-{cfg['gradient_accumulation_steps']}"
    )

    configs.append(cfg)

with OUT.open("w") as f:
    for cfg in configs:
        f.write(json.dumps(cfg) + "\n")

print(f"Wrote {len(configs)} configs to {OUT}")