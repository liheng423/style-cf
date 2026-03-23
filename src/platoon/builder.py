from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import os

from src.exps.agent import Agent
from src.exps.configs import data_filter_config, filter_names, style_data_config
from src.exps.models.model_loader import load_state_if_available
from src.exps.models.simulation import SimTensorDict, vstack
from src.exps.models.stylecf import EmbeddingStyleTransformer, StyleTransformer, style_embed_mask, style_embed_update_train_series
from src.exps.test.data_bundle import build_style_datapack, split_eval_windows
from src.exps.test.testing_utils import style_embed_pred_func
from src.exps.utils.scaler_io import load_scaler_payload
from src.exps.utils.utils import load_zen_data
from src.platoon.plat_sim import Env
from src.schema import CFNAMES as CF
from src.stylecf.schema import TensorNames

from .token_bank import TokenBank, build_style_token_bank


def _coerce_style_scalers(payload: Any) -> dict[str, Any]:
    expected = ("enc_x", "dec_x", "style")
    if isinstance(payload, dict):
        missing = [key for key in expected if key not in payload]
        if missing:
            raise ValueError(f"Style scaler payload missing keys: {missing}")
        return {key: payload[key] for key in expected}

    if isinstance(payload, (list, tuple)):
        if len(payload) < len(expected):
            raise ValueError("Style scaler list is shorter than required groups.")
        return {name: payload[idx] for idx, name in enumerate(expected)}

    raise TypeError(f"Unsupported style scaler payload type: {type(payload).__name__}")


@dataclass
class SimulationAssets:
    d_style: Any
    d_test: Any
    style_model: StyleTransformer
    embed_model: EmbeddingStyleTransformer
    style_scalers: dict[str, Any]
    token_bank: TokenBank
    device: torch.device
    simulation_config: dict[str, Any]
    newell_config: dict[str, float]

    @property
    def dt(self) -> float:
        return float(self.simulation_config.get("dt", 0.1))

    @property
    def pred_duration(self) -> float:
        return float(self.simulation_config.get("pred_duration", 4.0))

    @property
    def hist_duration(self) -> float:
        return float(self.simulation_config.get("hist_duration", 6.0))

    @property
    def sim_time(self) -> float:
        return float(self.simulation_config.get("sim_time", 240.0))

    @property
    def use_run_until_exit(self) -> bool:
        return bool(self.simulation_config.get("use_run_until_exit", False))

    @property
    def exit_range(self) -> tuple[float, float]:
        raw = self.simulation_config.get("exit_range", (-1000.0, 1200.0))
        return float(raw[0]), float(raw[1])


class PlatoonSimulationBuilder:
    def __init__(
        self,
        simulation_config: dict[str, Any],
        newell_config: dict[str, Any],
    ) -> None:
        self.simulation_config = simulation_config
        self.newell_config = {k: float(v) for k, v in newell_config.items()}
        self.device = torch.device(str(simulation_config.get("device", "cpu")))
        self.rng = np.random.default_rng(int(simulation_config.get("random_seed", 42)))
        self.assets = self._build_assets()

    def _resolve_data_path(self) -> str:
        path = os.environ.get("ZEN_DATA_PATH") or self.simulation_config.get("datapath")
        if not path:
            raise ValueError("Missing simulation dataset path. Set ZEN_DATA_PATH or simulation_config.datapath")
        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Simulation dataset not found: {path}")
        return path

    def _load_dataset(self):
        raw = load_zen_data(self._resolve_data_path(), rise=True, in_kph=False, kilo_norm=True)
        head = self.simulation_config.get("head")
        if head not in (None, "", ...):
            raw = raw.head(int(head))
        d_full = build_style_datapack(raw, filter_names, data_filter_config)

        style_window = tuple(self.simulation_config.get("style_window", (0, 300)))
        test_window = tuple(self.simulation_config.get("test_window", (300, 900)))
        d_style, d_test = split_eval_windows(d_full, style_window, test_window)
        return d_style, d_test

    def _load_style_model_and_scalers(self):
        style_model = StyleTransformer(style_data_config).to(self.device)

        style_state_path = Path(str(self.simulation_config.get("style_state_path", "")))
        if not style_state_path.exists():
            raise FileNotFoundError(f"Style model state file not found: {style_state_path}")
        style_model = load_state_if_available(style_model, str(style_state_path), self.device, strict=True)
        style_model = style_model.eval()

        scaler_path = Path(str(self.simulation_config.get("style_scaler_path", "")))
        if not scaler_path.exists():
            raise FileNotFoundError(f"Style scaler file not found: {scaler_path}")
        scaler_payload = load_scaler_payload(scaler_path)
        style_scalers = _coerce_style_scalers(scaler_payload)

        embed_model = EmbeddingStyleTransformer(style_model).to(self.device).eval()
        return style_model, embed_model, style_scalers

    def _build_assets(self) -> SimulationAssets:
        d_style, d_test = self._load_dataset()
        style_model, embed_model, style_scalers = self._load_style_model_and_scalers()
        style_feature_names = style_data_config["x_groups"]["style"]["features"]

        token_bank = build_style_token_bank(
            d_style=d_style,
            style_model=style_model,
            style_scalers=style_scalers,
            simulation_config=self.simulation_config,
            style_feature_names=style_feature_names,
            device=self.device,
        )

        return SimulationAssets(
            d_style=d_style,
            d_test=d_test,
            style_model=style_model,
            embed_model=embed_model,
            style_scalers=style_scalers,
            token_bank=token_bank,
            device=self.device,
            simulation_config=self.simulation_config,
            newell_config=self.newell_config,
        )

    def _new_style_agent(self) -> Agent:
        seq_len = int(style_data_config["seq_len"])
        pred_len = int(style_data_config["pred_len"])
        agent = Agent(
            self.assets.embed_model,
            dt=self.assets.dt,
            horizon_len=pred_len,
            historic_step=seq_len,
            scalers=self.assets.style_scalers,
            start_timestep=0,
        )
        agent._update_train_series = style_embed_update_train_series(agent, style_data_config)
        agent._concat = vstack
        return agent

    def _build_template_series(self, style_token: torch.Tensor, veh_len: float) -> SimTensorDict:
        enc_features = list(style_data_config["x_groups"]["enc_x"]["features"])
        dec_features = list(style_data_config["x_groups"]["dec_x"]["features"])

        enc_raw = np.zeros((1, len(enc_features)), dtype=np.float32)
        dec_raw = np.zeros((1, len(dec_features)), dtype=np.float32)

        if CF.SELF_L in enc_features:
            enc_raw[0, enc_features.index(CF.SELF_L)] = float(veh_len)
        if CF.LEAD_L in enc_features:
            enc_raw[0, enc_features.index(CF.LEAD_L)] = float(veh_len)

        enc_scaled = self.assets.style_scalers["enc_x"].transform(enc_raw)
        dec_scaled = self.assets.style_scalers["dec_x"].transform(dec_raw)

        enc_t = torch.tensor(enc_scaled, dtype=torch.float32, device=self.device).rename(TensorNames.T, TensorNames.F)
        dec_t = torch.tensor(dec_scaled, dtype=torch.float32, device=self.device).rename(TensorNames.T, TensorNames.F)
        style_t = style_token.to(self.device).rename(TensorNames.F)

        return SimTensorDict({"enc_x": enc_t, "dec_x": dec_t, "style_embed": style_t}, batch_size=[])

    def build_composition_from_counts(self, counts: dict[str, int]) -> list[str]:
        labels: list[str] = []
        for label, count in counts.items():
            labels.extend([str(label)] * int(count))
        return labels

    def sample_mix_composition(self, distribution: dict[str, float], platoon_len: int) -> list[str]:
        labels = list(distribution.keys())
        probs = np.array([float(distribution[label]) for label in labels], dtype=float)
        if np.any(probs < 0) or not np.any(probs > 0):
            raise ValueError("Invalid style distribution for mix composition.")
        probs = probs / probs.sum()
        sampled = self.rng.choice(labels, size=int(platoon_len), p=probs)
        return sampled.tolist()

    def _build_train_series(self, composition: list[str], veh_lens: list[float]) -> list[SimTensorDict]:
        train_series: list[SimTensorDict] = []
        for idx, style_label in enumerate(composition):
            token = self.assets.token_bank.sample(style_label, self.device, self.rng)
            train_series.append(self._build_template_series(token, veh_lens[idx]))
        return train_series

    def _build_env(self, header_movements: torch.Tensor, train_series: list[SimTensorDict], veh_lens: list[float]) -> Env:
        num_veh = len(train_series)
        agents = [self._new_style_agent() for _ in range(num_veh)]
        pred_func = style_embed_pred_func
        mask_func = style_embed_mask(style_data_config)
        return Env(
            platoon=agents,
            header_movements=header_movements,
            dt=self.assets.dt,
            pred_funcs=[pred_func for _ in range(num_veh)],
            masks=[mask_func for _ in range(num_veh)],
            pred_duration=self.assets.pred_duration,
            hist_duration=self.assets.hist_duration,
            dummy_train_sers=train_series,
            veh_lens=veh_lens,
        )

    def simulate(
        self,
        composition: list[str],
        leader_id: int,
        veh_lens: list[float] | None = None,
        use_run_until_exit: bool | None = None,
    ) -> np.ndarray:
        if veh_lens is None:
            veh_lens = [5.0 for _ in composition]
        if len(veh_lens) != len(composition):
            raise ValueError("veh_lens length must match composition length.")

        leader_id = int(leader_id)
        if leader_id < 0 or leader_id >= int(self.assets.d_test.data.shape[0]):
            raise IndexError(f"leader_id out of range: {leader_id}")

        header = torch.tensor(
            self.assets.d_test[leader_id, :, [CF.SELF_X, CF.SELF_V, CF.SELF_A]],
            dtype=torch.float32,
            device=self.device,
        )
        train_series = self._build_train_series(composition, veh_lens)
        env = self._build_env(header, train_series, veh_lens)

        run_to_exit = self.assets.use_run_until_exit if use_run_until_exit is None else bool(use_run_until_exit)
        if run_to_exit:
            results = env.run_until_exit(self.assets.exit_range, self.assets.newell_config)
        else:
            results = env.start(self.assets.sim_time, self.assets.newell_config)
        return torch.stack(results, dim=0).detach().cpu().numpy()


__all__ = ["SimulationAssets", "PlatoonSimulationBuilder"]
