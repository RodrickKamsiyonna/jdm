# train_flowmatching.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# MODIFICATIONS FOR HUGGING FACE STREAMING (AND DISTRIBUTED FIX):
# 1. Replaced TFRecord loading with HuggingFaceImageNetDataset using `datasets`.
# 2. Implemented a fix for a common distributed training race condition.
#    - Only the main process (rank 0) will now perform the initial dataset setup.
#    - A `torch.distributed.barrier()` is used to make other processes wait,
#      preventing them from failing while the dataset is being initialized.
#
# NOTE: You need to have the datasets library installed: `pip install datasets`
#       and be logged in to Hugging Face: `huggingface-cli login`

from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
import numpy as np
from PIL import Image
import io

import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
from torch.utils.data import IterableDataset

import datasets

import augmentations as aug
from distributed import init_distributed_mode
import resnet

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    wandb = None
    _HAS_WANDB = False

# --- Custom Dataset for Hugging Face Streaming ---

class HuggingFaceImageNetDataset(IterableDataset):
    """
    Custom PyTorch IterableDataset for streaming ImageNet from the Hugging Face Hub.
    Includes a synchronization barrier to prevent race conditions during initialization
    in a distributed training setup.
    """
    def __init__(self, hf_dataset_name, split, transform=None, is_main_process=False):
        """
        Args:
            hf_dataset_name (string): Name of the dataset on Hugging Face Hub.
            split (string): The dataset split to use (e.g., 'train').
            transform (callable, optional): Optional transform to be applied on a sample.
            is_main_process (bool): Flag to indicate if this is the main process (rank 0).
        """
        self.transform = transform
        self.hf_dataset_name = hf_dataset_name
        self.split = split

        # --- DISTRIBUTED TRAINING FIX ---
        # Let only the main process handle the initial dataset setup. This prevents
        # a race condition where multiple processes try to download/prepare metadata
        # simultaneously, which can cause failures on non-main processes.
        if is_main_process:
            print(f"Main process (rank 0) is initializing dataset '{self.hf_dataset_name}'...")
            datasets.load_dataset(
                self.hf_dataset_name,
                split=self.split,
                streaming=True,
                token=True
            )

        # Synchronize all processes. The other processes will wait here until
        # the main process has finished the step above.
        dist.barrier()

        print(f"Rank {dist.get_rank()} is loading dataset stream from cache...")
        # Now all processes can safely load the dataset. It will be loaded from cache
        # for ranks > 0, avoiding the race condition.
        self.dataset = datasets.load_dataset(
            self.hf_dataset_name,
            split=self.split,
            streaming=True,
            token=True,
        )
        print(f"Rank {dist.get_rank()} stream initialized successfully.")
        # --- END OF FIX ---

    def __iter__(self):
        # The 'datasets' library automatically handles sharding for multiple workers.
        for sample in self.dataset:
            # Based on the dataset structure for 'timm/imagenet-1k-wds':
            # image data is in 'jpg', label is in 'cls'.
            image = sample['jpg']  # This is already a PIL Image
            label = sample['cls']
            
            # Ensure it's a PIL Image
            if not isinstance(image, Image.Image):
                print(f"Warning: Rank {dist.get_rank() if dist.is_available() and dist.is_initialized() else 'N/A'} unexpected image type {type(image)}, skipping.", file=sys.stderr)
                continue
    
            if self.transform:
                img1, img2 = self.transform(image)
                yield (img1, img2), label
            else:
                yield image, label
# --- End of Custom Dataset ---

def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with Flow Matching", add_help=False)
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to local dataset (not used for HF streaming)")
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
    parser.add_argument("--num-workers", type=int, default=2,
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
    return parser


def main(args):
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv), file=stats_file)
    else:
        stats_file = None

    use_wandb = False
    if args.rank == 0 and args.wandb_project is not None and _HAS_WANDB:
        try:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
            use_wandb = True
        except Exception as e:
            print(f"Warning: failed to init wandb: {e}", file=sys.stderr)
    elif args.wandb_project is not None and not _HAS_WANDB and args.rank == 0:
        print("wandb requested but not installed. Install via `pip install wandb`.", file=sys.stderr)

    # --- Data Loading Modification for Hugging Face ---
    transforms = aug.TrainTransform()

    dataset = HuggingFaceImageNetDataset(
        hf_dataset_name="timm/imagenet-1k-wds",
        split="train",
        transform=transforms,
        is_main_process=(args.rank == 0) # Pass the flag here
    )
    
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size

    dataloader_kwargs = dict(
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2
        dataloader_kwargs["persistent_workers"] = True

    loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    imagenet_train_size = 1281167
    steps_per_epoch = math.ceil(imagenet_train_size / args.batch_size)
    if args.rank == 0:
        print(f"Using an estimated {steps_per_epoch} steps per epoch.")
    
    # --- End of Data Loading Modification ---

    model = FlowMatching(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu], find_unused_parameters=False, gradient_as_bucket_view=True
    )

    optimizer = LARS(
        model.parameters(), lr=0, weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("Resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt.get("epoch", 0)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    global_step = start_epoch * steps_per_epoch
    save_every_epoch = (args.ckpt_interval == 0)
    ckpt_interval = args.ckpt_interval if not save_every_epoch else 1

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for step, ((x, y), _) in enumerate(loader):
            if step >= steps_per_epoch:
                break
            effective_step = global_step + step
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)
            lr = adjust_learning_rate(args, optimizer, steps_per_epoch, effective_step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                stats = dict(epoch=epoch, step=step, loss=loss.item(), time=int(current_time - start_time), lr=lr)
                print(json.dumps(stats), file=stats_file if stats_file else sys.stdout)
                if use_wandb:
                    try:
                        wandb.log({"batch_loss": stats["loss"], "lr": lr}, step=effective_step)
                    except Exception as e:
                        print(f"Warning: wandb.log failed: {e}", file=sys.stderr)
                last_logging = current_time
        global_step += steps_per_epoch
        if args.rank == 0:
            should_save = save_every_epoch or ((epoch + 1) % ckpt_interval == 0)
            if should_save:
                state = dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict())
                torch.save(state, args.exp_dir / "model.pth")
    if args.rank == 0:
        torch.save(model.module.backbone.state_dict(), args.exp_dir / "resnet50.pth")

def adjust_learning_rate(args, optimizer, steps_per_epoch, step):
    max_steps = args.epochs * steps_per_epoch
    warmup_steps = 10 * steps_per_epoch
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
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        embeddings = t.squeeze(-1)[:, None] * freqs[None, :]
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
        self.backbone, self.embedding = resnet.__dict__[args.arch](zero_init_residual=True)
        projector_spec = f"{self.embedding}-{args.projector_mlp}"
        self.projection_head = build_mlp(projector_spec)
        time_proj_spec = f"{args.time_emb_dim}-{args.time_emb_mlp}"
        self.time_embedding = SinusoidalTimeEmbedding(dim=args.time_emb_dim)
        self.time_projection = build_mlp(time_proj_spec)
        time_emb_output_dim = int(args.time_emb_mlp.split('-')[-1])
        projector_output_dim = int(args.projector_mlp.split('-')[-1])
        velocity_input_dim = self.embedding + projector_output_dim + time_emb_output_dim
        velocity_spec = f"{velocity_input_dim}-{args.velocity_mlp}"
        self.velocity_predictor = build_mlp(velocity_spec, last_layer_bias=True)
    def forward(self, x, y):
        feat_x, feat_y = self.backbone(x), self.backbone(y)
        target_x, target_y = feat_x.detach(), feat_y.detach()
        context_x, context_y = self.projection_head(feat_x), self.projection_head(feat_y)
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
        return F.mse_loss(preds, truths)

def exclude_bias_and_norm(p):
    return p.ndim == 1

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, eta=eta,
                        weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad
                if dp is None: continue
                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])
                if g["lars_adaptation_filter"] is None or not g["lars_adaptation_filter"](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0, (g["eta"] * param_norm / update_norm), one),
                                    one)
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
