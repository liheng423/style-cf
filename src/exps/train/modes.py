from __future__ import annotations

import os
from typing import Mapping

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...utils.logger import logger
from ..losses import LossWeights, StyleMultiTaskLoss
from ..metrics import compute_embedding_distance_stats, compute_shuffle_gap
from ..models.stylecf import StyleTransformer
from ..utils import utils


def _labels_from_batch(y) -> torch.Tensor | None:
    if hasattr(y, "keys") and "self_id" in y.keys():
        labels = y["self_id"]
        if labels.ndim > 1:
            labels = labels.squeeze(-1)
        return labels.rename(None).long()
    return None


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for p in module.parameters():
        p.requires_grad = enabled


def _mean_stats(stats: Mapping[str, list[float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, values in stats.items():
        out[key] = float(sum(values) / max(len(values), 1))
    return out


def _run_multitask_epoch(
    model: StyleTransformer,
    dataloader: DataLoader,
    criterion: StyleMultiTaskLoss,
    dt: float,
    device: torch.device,
    weights: LossWeights,
    optimizer=None,
    max_norm: float | None = None,
    compute_metrics: bool = False,
    metrics_max_batches: int = 0,
    progress_desc: str | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    stats: dict[str, list[float]] = {
        "total": [],
        "spacing": [],
        "acc": [],
        "contrastive": [],
    }
    metric_stats: dict[str, list[float]] = {
        "shuffle_gap": [],
        "embedding_ratio": [],
        "embedding_intra": [],
        "embedding_inter": [],
    }

    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty.")

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        iterator = tqdm(dataloader, desc=progress_desc, leave=False, disable=False)
        for batch_idx, (x, y) in enumerate(iterator):
            x = x.to(device)
            y = y.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(x)
            losses = criterion.compute(outputs, y, dt=dt, weights=weights)
            total = losses["total"]

            if is_train:
                total.backward()
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()

            for key in ("total", "spacing", "acc", "contrastive"):
                stats[key].append(float(losses[key].detach().item()))

            if (
                compute_metrics
                and (not model.use_dummy_style)
                and batch_idx < metrics_max_batches
                and "style" in x.keys()
            ):
                gap = compute_shuffle_gap(model, x, y, criterion=criterion, dt=dt)
                metric_stats["shuffle_gap"].append(float(gap))

                if isinstance(outputs, tuple):
                    embed = outputs[1].detach()
                else:
                    embed = None
                label_batch = _labels_from_batch(y)
                dist_stats = compute_embedding_distance_stats(embed, label_batch)
                metric_stats["embedding_ratio"].append(float(dist_stats["embedding_ratio"]))
                metric_stats["embedding_intra"].append(float(dist_stats["embedding_intra"]))
                metric_stats["embedding_inter"].append(float(dist_stats["embedding_inter"]))

    mean_metric_stats: dict[str, float] = {}
    for key, values in metric_stats.items():
        clean = [v for v in values if v == v]  # drop NaN
        mean_metric_stats[key] = float(sum(clean) / len(clean)) if clean else float("nan")

    return _mean_stats(stats), mean_metric_stats


def _run_id_supervised_epoch(
    model: StyleTransformer,
    classifier: nn.Module,
    dataloader: DataLoader,
    id_to_class: dict[int, int],
    device: torch.device,
    optimizer=None,
    progress_desc: str | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
        classifier.train()
    else:
        model.eval()
        classifier.eval()

    stats: dict[str, list[float]] = {"id_ce": [], "embedding_ratio": []}
    criterion = nn.CrossEntropyLoss()

    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty.")

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for x, y in tqdm(dataloader, desc=progress_desc, leave=False, disable=False):
            x = x.to(device)
            y = y.to(device)

            labels = _labels_from_batch(y)
            if labels is None:
                raise KeyError("self_id labels are required for stage2_objective='id_supervised'.")

            mapped = torch.full_like(labels, fill_value=-1)
            for raw_id, cls_id in id_to_class.items():
                mapped[labels == int(raw_id)] = int(cls_id)

            valid = mapped >= 0
            if not bool(valid.any()):
                continue

            style_inputs = x["style"].rename(None)
            embeds = model.embedder(style_inputs)
            logits = classifier(embeds)
            loss = criterion(logits[valid], mapped[valid])

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            stats["id_ce"].append(float(loss.detach().item()))
            dist = compute_embedding_distance_stats(embeds.detach(), labels.detach())
            ratio = float(dist["embedding_ratio"])
            if ratio == ratio:
                stats["embedding_ratio"].append(ratio)

    return _mean_stats(stats)


def _build_id_mapping(train_loader: DataLoader) -> dict[int, int]:
    unique_ids: set[int] = set()
    for _, y in train_loader:
        labels = _labels_from_batch(y)
        if labels is None:
            raise KeyError("self_id labels are required for stage2_objective='id_supervised'.")
        unique_ids.update(int(x) for x in labels.cpu().tolist())
    sorted_ids = sorted(unique_ids)
    if not sorted_ids:
        raise ValueError("No SELF_ID labels found in train loader.")
    return {raw_id: idx for idx, raw_id in enumerate(sorted_ids)}


def _run_stage(
    stage_name: str,
    model: StyleTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: StyleMultiTaskLoss,
    train_config: dict,
    device: torch.device,
    epochs: int,
    lr: float,
    weights: LossWeights,
    save_best: bool = False,
) -> None:
    optim_cls = train_config["optim"]
    optimizer = optim_cls(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(lr),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=1e-1, patience=5)

    max_norm = train_config.get("max_norm")
    dt = float(train_config["dt"])
    metrics_batches = int(train_config.get("metrics_eval_batches", 0))

    best_val = float("inf")
    for epoch in range(int(epochs)):
        train_stats, _ = _run_multitask_epoch(
            model,
            train_loader,
            criterion,
            dt=dt,
            device=device,
            weights=weights,
            optimizer=optimizer,
            max_norm=max_norm,
            compute_metrics=False,
            progress_desc=f"[{stage_name}] train {epoch + 1}/{epochs}",
        )
        val_stats, val_metrics = _run_multitask_epoch(
            model,
            val_loader,
            criterion,
            dt=dt,
            device=device,
            weights=weights,
            optimizer=None,
            compute_metrics=metrics_batches > 0,
            metrics_max_batches=metrics_batches,
            progress_desc=f"[{stage_name}] val   {epoch + 1}/{epochs}",
        )
        scheduler.step(val_stats["total"])

        if save_best and val_stats["total"] < best_val:
            best_val = val_stats["total"]
            utils.model_save(model, train_config["best_model_path"])
            logger.info(
                f"[{stage_name}] saved best model at epoch={epoch + 1} "
                f"val_total={val_stats['total']:.6f}"
            )

        logger.info(
            f"[{stage_name}] epoch={epoch + 1}/{epochs} "
            f"train_total={train_stats['total']:.6f} "
            f"train_spacing={train_stats['spacing']:.6f} "
            f"train_acc={train_stats['acc']:.6f} "
            f"train_contrastive={train_stats['contrastive']:.6f} "
            f"val_total={val_stats['total']:.6f} "
            f"val_spacing={val_stats['spacing']:.6f} "
            f"val_acc={val_stats['acc']:.6f} "
            f"val_contrastive={val_stats['contrastive']:.6f} "
            f"val_shuffle_gap={val_metrics['shuffle_gap']:.4f} "
            f"val_embed_ratio={val_metrics['embedding_ratio']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )


def _run_stage2_id_supervised(
    model: StyleTransformer,
    train_loader: DataLoader,
    train_config: dict,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    id_to_class = _build_id_mapping(train_loader)
    classifier = nn.Linear(model.embed_dim, len(id_to_class)).to(device)

    optim_cls = train_config["optim"]
    optimizer = optim_cls(
        list(model.embedder.parameters()) + list(classifier.parameters()),
        lr=float(lr),
    )

    for epoch in range(int(epochs)):
        train_stats = _run_id_supervised_epoch(
            model=model,
            classifier=classifier,
            dataloader=train_loader,
            id_to_class=id_to_class,
            device=device,
            optimizer=optimizer,
            progress_desc=f"[stage2-id] train {epoch + 1}/{epochs}",
        )
        logger.info(
            f"[stage2-id] epoch={epoch + 1}/{epochs} "
            f"train_id_ce={train_stats['id_ce']:.6f} "
            f"train_embed_ratio={train_stats['embedding_ratio']:.4f}"
        )


def run_style_training_mode(
    model: StyleTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: StyleMultiTaskLoss,
    train_config: dict,
    device: torch.device,
) -> StyleTransformer:
    mode = str(train_config.get("training_mode", "single_stage")).lower()
    base_lr = float(train_config.get("lr", 1e-4))

    default_weights = LossWeights(
        spacing=float(train_config.get("w_spacing", criterion.default_weights.spacing)),
        acc=float(train_config.get("w_acc", criterion.default_weights.acc)),
        contrastive=float(train_config.get("w_contrastive", criterion.default_weights.contrastive)),
    )

    if mode == "single_stage":
        model.use_dummy_style = False
        model.dummy_style_mode = "zeros"
        _set_requires_grad(model, True)
        _run_stage(
            stage_name="single",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            train_config=train_config,
            device=device,
            epochs=int(train_config.get("num_epoch", 40)),
            lr=float(train_config.get("stage3_lr", base_lr)),
            weights=default_weights,
            save_best=True,
        )
    elif mode == "three_stage":
        stage1_epochs = int(train_config.get("stage1_epochs", 10))
        stage2_epochs = int(train_config.get("stage2_epochs", 10))
        stage3_epochs = int(train_config.get("stage3_epochs", 20))
        stage2_objective = str(train_config.get("stage2_objective", "contrastive")).lower()

        # Stage 1: no style, stabilize dynamics.
        model.use_dummy_style = True
        model.dummy_style_mode = "zeros"
        _set_requires_grad(model.transfollower, True)
        _set_requires_grad(model.embedder, False)
        _run_stage(
            stage_name="stage1-no-style",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            train_config=train_config,
            device=device,
            epochs=stage1_epochs,
            lr=float(train_config.get("stage1_lr", base_lr)),
            weights=LossWeights(spacing=1.0, acc=default_weights.acc, contrastive=0.0),
            save_best=False,
        )

        # Stage 2: freeze backbone, train embedder.
        model.use_dummy_style = False
        _set_requires_grad(model.transfollower, False)
        _set_requires_grad(model.embedder, True)
        if stage2_objective == "id_supervised":
            _run_stage2_id_supervised(
                model=model,
                train_loader=train_loader,
                train_config=train_config,
                device=device,
                epochs=stage2_epochs,
                lr=float(train_config.get("stage2_lr", base_lr)),
            )
        else:
            _run_stage(
                stage_name="stage2-contrastive",
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                train_config=train_config,
                device=device,
                epochs=stage2_epochs,
                lr=float(train_config.get("stage2_lr", base_lr)),
                weights=LossWeights(spacing=0.0, acc=0.0, contrastive=1.0),
                save_best=False,
            )

        # Stage 3: joint low-lr fine-tune.
        _set_requires_grad(model.transfollower, True)
        _set_requires_grad(model.embedder, True)
        _run_stage(
            stage_name="stage3-joint",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            train_config=train_config,
            device=device,
            epochs=stage3_epochs,
            lr=float(train_config.get("stage3_lr", base_lr * 0.2)),
            weights=default_weights,
            save_best=True,
        )
    else:
        raise ValueError(
            f"Unsupported training_mode={mode}. Expected 'single_stage' or 'three_stage'."
        )

    best_path = str(train_config.get("best_model_path", ""))
    if best_path and os.path.exists(best_path):
        state = torch.load(best_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)

    final_test_stats, final_metrics = _run_multitask_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        dt=float(train_config["dt"]),
        device=device,
        weights=default_weights,
        optimizer=None,
        compute_metrics=bool(int(train_config.get("metrics_eval_batches", 0)) > 0),
        metrics_max_batches=int(train_config.get("metrics_eval_batches", 0)),
        progress_desc="[final-test]",
    )
    logger.info(
        "Final test "
        f"total={final_test_stats['total']:.6f} "
        f"spacing={final_test_stats['spacing']:.6f} "
        f"acc={final_test_stats['acc']:.6f} "
        f"contrastive={final_test_stats['contrastive']:.6f} "
        f"shuffle_gap={final_metrics['shuffle_gap']:.4f} "
        f"embedding_ratio={final_metrics['embedding_ratio']:.4f}"
    )
    return model
