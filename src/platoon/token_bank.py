from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans

from src.schema import CFNAMES as CF
from src.exps.datahandle.stylebuilder import build_style_tokens_from_datapack


@dataclass
class TokenBank:
    pools: dict[str, torch.Tensor]
    fallback_label: str
    fallback_to_notruck: bool = True

    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.pools.keys()))

    def resolve_label(self, label: str) -> str:
        if label in self.pools:
            return label
        if self.fallback_to_notruck and label.startswith("T"):
            suffix = label[1:]
            candidate = f"P{suffix}"
            if candidate in self.pools:
                return candidate
            p_labels = [x for x in self.pools if x.startswith("P")]
            if p_labels:
                return sorted(p_labels)[0]
        return self.fallback_label

    def sample(
        self,
        label: str,
        device: torch.device,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        real_label = self.resolve_label(label)
        pool = self.pools[real_label]
        idx = int(rng.integers(0, pool.shape[0]))
        return pool[idx].to(device)


def _cluster_to_pools(
    tokens: np.ndarray,
    prefix: str,
    num_clusters: int,
    random_seed: int,
) -> dict[str, np.ndarray]:
    if tokens.size == 0:
        return {}

    n = tokens.shape[0]
    k = max(1, min(int(num_clusters), n))

    if k == 1:
        labels = np.zeros(n, dtype=int)
    else:
        model = KMeans(n_clusters=k, random_state=random_seed, n_init="auto")
        labels = model.fit_predict(tokens)

    pools: dict[str, list[np.ndarray]] = {}
    for idx, cluster_id in enumerate(labels.tolist()):
        label = f"{prefix}{int(cluster_id)}"
        pools.setdefault(label, []).append(tokens[idx])

    return {key: np.stack(values, axis=0) for key, values in pools.items()}


def build_style_token_bank(
    d_style,
    style_model,
    style_scalers: dict[str, Any],
    simulation_config: dict[str, Any],
    style_feature_names: list[str],
    device: torch.device,
) -> TokenBank:
    seed = int(simulation_config.get("random_seed", 42))
    style_seconds = float(simulation_config.get("style_token_seconds", 30.0))
    token_source = str(simulation_config.get("style_token_source", "style_window_head"))
    anchor_step = simulation_config.get("style_token_anchor_step")
    anchor_value = None if anchor_step in (None, "", ...) else int(anchor_step)
    if token_source == "style_window_head":
        anchor_value = None

    style_tokens = build_style_tokens_from_datapack(
        datapack=d_style,
        feature_names=style_feature_names,
        embedder=style_model.embedder,
        seconds=style_seconds,
        scaler=style_scalers.get("style"),
        device=device,
        end_step=anchor_value,
    ).detach()

    token_np = style_tokens.cpu().numpy().astype(np.float32)
    mean_self_len = d_style[:, :, CF.SELF_L].mean(axis=1)

    notruck_cfg = dict(simulation_config.get("notruck_bank", {}))
    truck_cfg = dict(simulation_config.get("truck_bank", {}))
    length_threshold = float(notruck_cfg.get("length_threshold", 7.5))

    notruck_mask = mean_self_len < length_threshold
    truck_mask = ~notruck_mask

    pools_np: dict[str, np.ndarray] = {}

    if bool(notruck_cfg.get("enabled", True)):
        pools_np.update(
            _cluster_to_pools(
                token_np[notruck_mask],
                prefix=str(notruck_cfg.get("prefix", "P")),
                num_clusters=int(notruck_cfg.get("num_clusters", 4)),
                random_seed=seed,
            )
        )

    if bool(truck_cfg.get("enabled", True)):
        pools_np.update(
            _cluster_to_pools(
                token_np[truck_mask],
                prefix=str(truck_cfg.get("prefix", "T")),
                num_clusters=int(truck_cfg.get("num_clusters", 3)),
                random_seed=seed + 17,
            )
        )

    if not pools_np:
        pools_np["P0"] = token_np

    fallback_label = str(notruck_cfg.get("fallback_label", sorted(pools_np.keys())[0]))
    if fallback_label not in pools_np:
        fallback_label = sorted(pools_np.keys())[0]

    pools = {label: torch.tensor(values, dtype=torch.float32) for label, values in pools_np.items()}
    return TokenBank(
        pools=pools,
        fallback_label=fallback_label,
        fallback_to_notruck=bool(truck_cfg.get("fallback_to_notruck", True)),
    )


__all__ = ["TokenBank", "build_style_token_bank"]
