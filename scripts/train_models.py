"""Train all disease pipelines and produce measured evaluation artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disease_prediction.training import train_all  # noqa: E402


if __name__ == "__main__":
    results = train_all(ROOT)
    for key, item in results.items():
        metrics = item["holdout_metrics"]
        print(
            f"{key}: selected={item['selected_model']}; "
            f"recall={metrics['recall_sensitivity']:.3f}; "
            f"precision={metrics['precision']:.3f}; "
            f"ROC-AUC={metrics['roc_auc']:.3f}"
        )
