# train_flowmatching.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
import torchvision.datasets as datasets

import augmentations as aug
from distributed import init_distributed_mode

import resnet

try:
    import wandb

    _HAS_WANDB = True
except Exception:
    wandb = None
    _HAS_WANDB = False


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with Flow Matching", add_help=False)

    parser.add_argument("--data-dir", type=Path, default="/path/to/imagenet", required=True,
                        help="Path to the image net dataset")
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help="Path to the experiment folder, where all logs/checkpoints will be stored")
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help="Print logs to the stats.txt file every [log-freq-time] seconds")
    parser.add_argument("--arch", type=str, default="resnet50",
                        help="Architecture of the backbone encoder network")
    parser.add_argument("--projector-mlp", default="8192-8192-512",
                        help="Size and number of layers of the MLP projector head")
    parser.add_argument("--time-emb-dim", type=int, default=128,
                        help="Dimension of the sinusoidal time embeddings")
    parser.add_argument("--time-emb-mlp", default="128-512",
                        help="MLP layers for time embedding projection")
    parser.add_argument("--velocity-mlp", default="1536-1024-512",
                        help="MLP layers for velocity predictor (input: 2*embedding_dim + time_emb_dim)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Effective batch size (per worker batch size is [batch-size] / world-size)")
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help="Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256")
    parser.add_argument("--wd", type=float, default=1e-6,
                        help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=10,
                        help="Number of dataloader workers")

    parser.add_argument("--device", default="cuda",
                        help="device to use for training / testing")

    parser.add_argument("--world-size", default=1, type=int,
                        help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist-url", default="env://",
                        help="url used to set up distributed training")

    parser.add_argument("--wandb-project", default=None,
                        help="W&B project name (if provided, wandb will be initialized on rank 0 after 'wandb login')")
    parser.add_argument("--wandb-entity", default=None,
                        help="W&B entity (team/user) if needed")

    parser.add_argument("--ckpt-interval", type=int, default=10,
                        help="Save a full checkpoint every N epochs (master only). 0 = save every epoch")
    
    parser.add_argument("--resolution",type=int, default=224,
                        help="Resolution of the Image")

    return parser


def main(args):
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)

    # Ensure correct GPU mapping
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        gpu = torch.device("cuda", args.local_rank)
    else:
        gpu = torch.device(args.device)

    print(f"[rank {args.rank}] using device {gpu}")

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)
    else:
        stats_file = None

    use_wandb = False
    if args.rank == 0 and args.wandb_project is not None and _HAS_WANDB:
        try:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
            use_wandb = True
            wandb.run.summary["script"] = os.path.basename(__file__)
        except Exception as e:
            print(f"Warning: failed to init wandb: {e}", file=sys.stderr)
            use_wandb = False
    elif args.wandb_project is not None and not _HAS_WANDB and args.rank == 0:
        print("wandb requested but not installed. Install via `pip install wandb` or unset --wandb-project.",
              file=sys.stderr)

    transforms = aug.TrainTransform(args.resolution)

    dataset = datasets.ImageFolder(args.data_dir / "train", transforms)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size

    dataloader_kwargs = dict(
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    if args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2
        dataloader_kwargs["persistent_workers"] = True

    loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    model = FlowMatching(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu], find_unused_parameters=False, gradient_as_bucket_view=True
    )

    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt.get("epoch", 0)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    start_time = last_logging = time.time()
    scaler = torch.amp.GradScaler(device="cuda")
    global_step = start_epoch * len(loader)
    ckpt_interval = max(1, args.ckpt_interval) if args.ckpt_interval != 0 else 1  # default to 1 if 0 given -> but we'll respect 0 meaning every epoch
    save_every_epoch = (args.ckpt_interval == 0)

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()

        epoch_loss_sum = 0.0
        epoch_steps = 0

        for step, ((x, y), _) in enumerate(loader, start=epoch * len(loader)):
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)

            lr = adjust_learning_rate(args, optimizer, loader, step)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            epoch_loss_sum += batch_loss
            epoch_steps += 1
            global_step += 1

            batch_stats = dict(
                epoch=epoch,
                step=step,
                global_step=global_step,
                batch_loss=batch_loss,
                time=int(time.time() - start_time),
                lr=lr,
            )

            if args.rank == 0:
                print(json.dumps(batch_stats))
                if stats_file is not None:
                    print(json.dumps(batch_stats), file=stats_file)

                if use_wandb:
                    try:
                        wandb.log({"batch_loss": batch_loss, "lr": lr, "epoch": epoch, "step": step}, step=global_step)
                    except Exception as e:
                        print(f"Warning: wandb.log failed: {e}", file=sys.stderr)

            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                stats = dict(
                    epoch=epoch,
                    step=step,
                    loss=batch_loss,
                    time=int(current_time - start_time),
                    lr=lr,
                )
                print(json.dumps(stats))
                if stats_file is not None:
                    print(json.dumps(stats), file=stats_file)
                last_logging = current_time

        avg_epoch_loss = epoch_loss_sum / max(1, epoch_steps)

        if args.rank == 0:
            epoch_stats = dict(
                epoch=epoch,
                epoch_loss=avg_epoch_loss,
                steps=epoch_steps,
                time=int(time.time() - start_time),
            )
            print("EPOCH_SUMMARY: " + json.dumps(epoch_stats))
            if stats_file is not None:
                print("EPOCH_SUMMARY: " + json.dumps(epoch_stats), file=stats_file)

            if use_wandb:
                try:
                    wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch}, step=global_step)
                except Exception as e:
                    print(f"Warning: wandb.log failed at epoch end: {e}", file=sys.stderr)

            should_save = save_every_epoch or ((epoch + 1) % ckpt_interval == 0)
            if should_save:
                state = dict(
                    epoch=epoch + 1,
                    model=model.model.module.backbone.state_dict(),
                    optimizer=optimizer.state_dict(),
                )
                ckpt_path = args.exp_dir / "model.pth"
                torch.save(state, ckpt_path)

    if args.rank == 0:
        final_ckpt_path = args.exp_dir / "model_final.pth"
        final_state = dict(
            epoch=args.epochs,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        torch.save(final_state, final_ckpt_path)

        torch.save(model.module.backbone.state_dict(), args.exp_dir / "resnet50.pth")

        if use_wandb:
            try:
                wandb.save(str(final_ckpt_path))
            except Exception:
                pass


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step_adj = step - warmup_steps
        max_steps_adj = max_steps - warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step_adj / max_steps_adj))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb_scale = np.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        embeddings = t * freqs
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def build_mlp(spec, last_layer_bias=False):
    layers = []
    f = list(map(int, spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=last_layer_bias))
    return nn.Sequential(*layers)


class FlowMatching(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )

        self.projection_head = build_mlp(args.projector_mlp)

        self.time_embedding = SinusoidalTimeEmbedding(dim=args.time_emb_dim)
        self.time_projection = build_mlp(args.time_emb_mlp)

        self.velocity_predictor = build_mlp(args.velocity_mlp)

    def forward(self, x, y):
        
        feat_x = self.backbone(x)
        feat_y = self.backbone(y)

        target_x = feat_x.detach()
        target_y = feat_y.detach()

        context_x = self.projection_head(feat_x)
        context_y = self.projection_head(feat_y)

        y_0 = torch.randn_like(target_y)
        t = torch.rand(target_y.shape[0], 1, device=target_y.device)
        y_t = t * target_y + (1 - t) * y_0
        true_velocity_xy = target_y - y_0

        t_emb = self.time_embedding(t)
        t_emb_proj = self.time_projection(t_emb)

        pred_velocity_xy = self.velocity_predictor(torch.cat([y_t, context_x, t_emb_proj], dim=1))
    
        y_0_b = torch.randn_like(target_x)
        t_b = torch.rand(target_x.shape[0], 1, device=target_x.device)
        y_t_b = t_b * target_x + (1 - t_b) * y_0_b
        true_velocity_yx = target_x - y_0_b

        t_emb_b = self.time_embedding(t_b)
        t_emb_proj_b = self.time_projection(t_emb_b)

        pred_velocity_yx = self.velocity_predictor(torch.cat([y_t_b, context_y, t_emb_proj_b], dim=1))

        preds = torch.cat([pred_velocity_xy, pred_velocity_yx], dim=0)
        truths = torch.cat([true_velocity_xy, true_velocity_yx], dim=0)
        loss = F.mse_loss(preds, truths)
        return loss


def exclude_bias_and_norm(p):
    return p.ndim == 1


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(update_norm > 0, (g["eta"] * param_norm / update_norm), one),
                        one,
                    )
                    dp = dp.mul(q)
                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Flow Matching training script", parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
