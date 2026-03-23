from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import f_oneway

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except Exception:  # pragma: no cover
    pairwise_tukeyhsd = None


def run_anova(results_by_group: dict[str, dict[str, list[float] | float]]) -> dict[str, dict[str, Any]]:
    if not results_by_group:
        return {}

    metrics = list(next(iter(results_by_group.values())).keys())
    out: dict[str, dict[str, Any]] = {}

    for metric in metrics:
        groups: list[np.ndarray] = []
        labels: list[str] = []

        for group_name, metric_map in results_by_group.items():
            values = metric_map.get(metric, [])
            arr = np.asarray(values, dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            groups.append(arr)
            labels.extend([group_name] * int(arr.size))

        if len(groups) < 2:
            out[metric] = {"note": "Not enough valid groups for ANOVA"}
            continue

        f_stat, p_val = f_oneway(*groups)
        result: dict[str, Any] = {"ANOVA_F": float(f_stat), "ANOVA_p": float(p_val)}
        if p_val < 0.05 and pairwise_tukeyhsd is not None:
            all_values = np.concatenate(groups)
            tukey = pairwise_tukeyhsd(all_values, labels, alpha=0.05)
            result["Tukey"] = tukey.summary().as_text()
        out[metric] = result

    return out


__all__ = ["run_anova"]
