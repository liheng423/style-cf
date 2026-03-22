from __future__ import annotations

import random
from os import makedirs
from pathlib import Path
from typing import Dict, TYPE_CHECKING

import pandas as pd
import torch
import torch.utils.data
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..schema import CFNAMES as CF

if TYPE_CHECKING:
    from ..exps.agent import Agent
    from ..exps.models.idm import IDM
    from ..exps.utils.datapack import SampleDataPack

try:
    from sko.GA import GA
except ModuleNotFoundError:  # pragma: no cover - exercised when sko is installed
    GA = None  # type: ignore[assignment]


def evaluate_recursive(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    simulator: "Agent",
    config: dict,
):
    """
    Evaluate the whole trajectory by rolling the simulator recursively.
    """

    device = config["device"]
    pred_func = config["pred_func"]
    mask = config["mask"]
    dt = config["resolution"]
    model.eval()
    running_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            y_self = y["self_move"]
            y_leader = y["leader_move"]

            pred_self = simulator.predict(x, y_self, y_leader, pred_func, mask)
            loss = criterion(pred_self, y_self, dt)
            running_loss += loss.item()

    return running_loss / num_batches


def _fitness_function(params, idm_model: "IDM", dataloader: data.DataLoader, config):
    """
    Calculate fitness (loss) for one IDM parameter vector.
    """
    device = config["device"]
    loss_func = config["loss"]
    dt = config["resolution"]
    scaler = config["scaler"]
    start_step = config["start_step"]
    update_func = config["update_func"]
    pred_horizon = config["pred_horizon"]
    historic_step = config["historic_step"]

    from ..exps.agent import Agent

    model = idm_model(params, use_torch=True).to(device)
    simulator = Agent(model, dt, pred_horizon, historic_step, scaler, start_timestep=start_step)
    simulator._update_train_series = update_func(simulator)
    simulator._concat = config["concat"]

    fitness = evaluate_recursive(model, dataloader, loss_func, simulator, config)
    return fitness


def calibrate_idm_genetic(dataloader, idm_model: "IDM", config: dict):
    if GA is None:
        raise ModuleNotFoundError("Package 'sko' is required for IDM genetic calibration.")

    bounds = config.get("bounds", [(15, 30), (0, 3), (0.5, 3), (0.5, 4), (0.5, 4)])
    ga = GA(
        func=lambda params: _fitness_function(params, idm_model, dataloader, config),
        n_dim=5,
        size_pop=int(config.get("size_pop", 10)),
        max_iter=int(config.get("max_iter", 50)),
        prob_mut=float(config.get("prob_mut", 0.2)),
        lb=[bounds[i][0] for i in range(5)],  # type: ignore[index]
        ub=[bounds[i][1] for i in range(5)],  # type: ignore[index]
        precision=float(config.get("precision", 1e-2)),
    )
    ga.to(config["device"])

    best_params, best_loss = ga.run()
    return best_params, best_loss


def calibrate_idm(idm: "IDM", id_datapack: Dict[int, SampleDataPack], config):
    """
    Calibrate IDM per vehicle-ID then save all best parameters as CSV.
    """
    if len(id_datapack) == 0:
        raise ValueError("id_datapack is empty, cannot calibrate IDM.")

    requested_sample_size = int(config.get("sample_size", 1000))
    sample_size = min(requested_sample_size, len(id_datapack))
    rng = random.Random(int(config.get("randomseed", 42)))
    sample_keys = rng.sample(list(id_datapack.keys()), sample_size)

    results = []

    dataset_cls = config.get("dataset_cls")
    if dataset_cls is None:
        from ..exps.datahandle.dataset import IDMDataset as dataset_cls
    dataloader_cls = config.get("dataloader_cls", DataLoader)

    for key in tqdm(sample_keys):
        sample_pack = id_datapack[key]
        idm_dataset = dataset_cls(
            sample_pack[:, :, config["x_groups"]["x"]["features"]],
            sample_pack[:, :, config["y_groups"]["y"]["features"]],
            sample_pack[:, :, config["y_groups"]["y"]["features"]],
            config["downsample"],
        )
        idm_dataloader = dataloader_cls(idm_dataset, 1, collate_fn=lambda x: x[0])

        best_params, best_loss = calibrate_idm_genetic(idm_dataloader, idm, config)
        best_loss_value = float(best_loss[0] if hasattr(best_loss, "__len__") else best_loss)

        results.append(
            {
                "ID": sample_pack[0, 0, CF.SELF_ID],
                "v0": best_params[0],
                "s0": best_params[1],
                "T": best_params[2],
                "a": best_params[3],
                "b": best_params[4],
                "best_loss": best_loss_value,
            }
        )

    df = pd.DataFrame(results)

    save_path = Path(str(config["save_path"]))
    if save_path.suffix.lower() != ".csv":
        save_path = save_path / "idm_calibration.csv"
    makedirs(save_path.parent, exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"Results saved to {save_path}")
    return df


__all__ = [
    "evaluate_recursive",
    "calibrate_idm_genetic",
    "calibrate_idm",
]
