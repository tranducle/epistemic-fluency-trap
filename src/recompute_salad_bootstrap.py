from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ARTIFACT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

from core import cluster_bootstrap_ci

result_path = ARTIFACT_ROOT / "results" / "salad_results.json"
pred_path = ARTIFACT_ROOT / "experiment" / "outputs" / "salad" / "primary_predictions.npz"

data = np.load(pred_path, allow_pickle=True)
result = json.loads(result_path.read_text(encoding="utf-8"))
threshold = float(result["tuning"]["PITSA_FullTrain"]["threshold"])
bootstrap = cluster_bootstrap_ci(
    data["label"],
    data["PITSA_FullTrain_clean_scores"],
    threshold,
    data["cluster_id"],
    1000,
    20261014,
)
result["pitsa_full_cluster_bootstrap"] = bootstrap
result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(bootstrap, indent=2, sort_keys=True))
