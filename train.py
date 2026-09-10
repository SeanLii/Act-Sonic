#!/usr/bin/env python3
"""Train a classic ACT policy on the G1 Sonic motion-token dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from act.data import (
    NormalizationStats,
    SonicACTDataset,
    dataset_summary,
    read_episode_indices,
    split_episodes,
)
from act.model import ACTPolicy
from act.sonic_contract import SONIC_ACTION_HORIZON, contract_dict
from act.training import compute_loss, seed_everything, seed_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("io_g1_box_cotransport2_sonic_joint"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/act_sonic"))
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=SONIC_ACTION_HORIZON)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=7)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--pad-weight", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-batches", type=int, default=100, help="0 evaluates the complete split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--wandb-project", default="act-sonic-g1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0, help="debug only; 0 means all")
    return parser.parse_args()


def model_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "qpos_dim": 46,
        "action_dim": 78,
        "chunk_size": args.chunk_size,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "nheads": args.nheads,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }


def make_scheduler(optimizer: AdamW, warmup: int, total: int) -> LambdaLR:
    def multiplier(step: int) -> float:
        if step < warmup:
            return max(step, 1) / max(warmup, 1)
        progress = min(1.0, (step - warmup) / max(total - warmup, 1))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, multiplier)


def save_checkpoint(
    path: Path,
    model: ACTPolicy,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    global_step: int,
    best_val: float,
    args: argparse.Namespace,
    stats: NormalizationStats,
    train_episodes: list[int],
    val_episodes: list[int],
) -> None:
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
        "model_config": model_config(args),
        "normalization": stats.as_dict(),
        "sonic_contract": contract_dict(args.chunk_size),
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "args": vars(args),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def validate(
    model: ACTPolicy,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    autocast_context: Any,
) -> dict[str, float]:
    model.eval()
    totals = {name: 0.0 for name in ("loss", "l1", "kl", "pad_bce")}
    count = 0
    for batch_index, batch in enumerate(loader):
        if args.val_batches and batch_index >= args.val_batches:
            break
        qpos = batch["qpos"].to(device, non_blocking=True)
        image = batch["image"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        is_pad = batch["is_pad"].to(device, non_blocking=True)
        with autocast_context():
            outputs = model(qpos, image, actions, is_pad)
            losses = compute_loss(model, outputs, actions, is_pad, args.kl_weight, args.pad_weight)
        for name in totals:
            totals[name] += float(losses[name])
        count += 1
    if count == 0:
        raise RuntimeError("validation loader yielded no batches")
    return {name: value / count for name, value in totals.items()}


def main() -> None:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is visible")
    if args.grad_accum_steps < 1:
        raise ValueError("grad-accum-steps must be >= 1")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    stats = NormalizationStats.from_json(args.dataset_dir / "meta" / "stats.json")
    all_episodes = read_episode_indices(args.dataset_dir)
    train_episodes, val_episodes = split_episodes(all_episodes, args.val_ratio, args.seed)
    train_dataset = SonicACTDataset(
        args.dataset_dir, train_episodes, args.chunk_size, stats, args.image_size, training=True
    )
    val_dataset = SonicACTDataset(
        args.dataset_dir, val_episodes, args.chunk_size, stats, args.image_size, training=False
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, generator=generator, **loader_options
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_options)

    model = ACTPolicy(**model_config(args)).to(device)
    backbone_parameters = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    other_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    optimizer = AdamW(
        [
            {"params": other_parameters, "lr": args.lr},
            {"params": backbone_parameters, "lr": args.backbone_lr},
        ],
        weight_decay=args.weight_decay,
    )
    batches_per_epoch = min(len(train_loader), args.max_train_batches or len(train_loader))
    updates_per_epoch = math.ceil(batches_per_epoch / args.grad_accum_steps)
    scheduler = make_scheduler(optimizer, args.warmup_steps, updates_per_epoch * args.epochs)
    start_epoch, global_step, best_val = 0, 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_val = float(checkpoint["best_val"])

    if args.compile:
        model = torch.compile(model)
    amp_enabled = args.amp != "off" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    autocast_context = (
        (lambda: torch.autocast(device_type=device.type, dtype=amp_dtype))
        if amp_enabled else nullcontext
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and args.amp == "fp16"))

    import wandb

    run_config = vars(args).copy()
    run_config.update(model_config(args))
    run_config.update({
        "train_data": dataset_summary(train_dataset),
        "val_data": dataset_summary(val_dataset),
        "parameters": model.num_parameters if isinstance(model, ACTPolicy) else model._orig_mod.num_parameters,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
    })
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config=run_config,
        dir=str(args.output_dir),
        resume="allow",
    )
    with (args.output_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump({"train": train_episodes, "val": val_episodes}, handle, indent=2)
    print(f"device={device} parameters={run_config['parameters']:,}")
    print(f"train={dataset_summary(train_dataset)} val={dataset_summary(val_dataset)}")
    print(f"wandb={run.url or args.wandb_mode}")

    optimizer.zero_grad(set_to_none=True)
    last_log_time = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        accumulated: dict[str, float] = {name: 0.0 for name in ("loss", "l1", "kl", "pad_bce")}
        interval_batches = 0
        for batch_index, batch in enumerate(progress):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            qpos = batch["qpos"].to(device, non_blocking=True)
            image = batch["image"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            is_pad = batch["is_pad"].to(device, non_blocking=True)
            with autocast_context():
                outputs = model(qpos, image, actions, is_pad)
                losses = compute_loss(model, outputs, actions, is_pad, args.kl_weight, args.pad_weight)
                scaled_loss = losses["loss"] / args.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            is_last = batch_index + 1 == batches_per_epoch
            if (batch_index + 1) % args.grad_accum_steps == 0 or is_last:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            else:
                grad_norm = torch.tensor(float("nan"))
            for name in accumulated:
                accumulated[name] += float(losses[name].detach())
            interval_batches += 1
            progress.set_postfix(l1=f"{float(losses['l1']):.4f}", kl=f"{float(losses['kl']):.4f}")
            if global_step > 0 and global_step % args.log_every == 0 and (
                (batch_index + 1) % args.grad_accum_steps == 0 or is_last
            ):
                elapsed = max(time.monotonic() - last_log_time, 1e-6)
                log = {f"train/{name}": value / interval_batches for name, value in accumulated.items()}
                log.update({
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/backbone_lr": scheduler.get_last_lr()[1],
                    "train/grad_norm": float(grad_norm),
                    "train/samples_per_sec": interval_batches * args.batch_size / elapsed,
                    "epoch": epoch + (batch_index + 1) / batches_per_epoch,
                })
                if device.type == "cuda":
                    log["system/gpu_memory_allocated_gib"] = torch.cuda.memory_allocated() / 2**30
                    log["system/gpu_memory_reserved_gib"] = torch.cuda.memory_reserved() / 2**30
                    log["system/gpu_max_memory_allocated_gib"] = (
                        torch.cuda.max_memory_allocated() / 2**30
                    )
                wandb.log(log, step=global_step)
                accumulated = {name: 0.0 for name in accumulated}
                interval_batches = 0
                last_log_time = time.monotonic()

        val_metrics = validate(model, val_loader, device, args, autocast_context)
        wandb.log({f"val/{name}": value for name, value in val_metrics.items()} | {"epoch": epoch + 1}, step=global_step)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(
                args.output_dir / "best.pt", model, optimizer, scheduler, epoch, global_step,
                best_val, args, stats, train_episodes, val_episodes,
            )
        latest_path = args.output_dir / "latest.pt"
        save_checkpoint(
            latest_path, model, optimizer, scheduler, epoch, global_step, best_val,
            args, stats, train_episodes, val_episodes,
        )
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                args.output_dir / f"epoch_{epoch + 1:04d}.pt", model, optimizer, scheduler,
                epoch, global_step, best_val, args, stats, train_episodes, val_episodes,
            )
        print(f"epoch={epoch + 1} val={val_metrics} best={best_val:.6f}")
    run.finish()


if __name__ == "__main__":
    main()
