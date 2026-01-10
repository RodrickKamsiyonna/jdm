# train_flowmatching_xla.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Optimized for TPU v4 (8 cores) training with torch_xla
#
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
import torchvision.datasets as datasets

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

import augmentations as aug
import resnet

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    wandb = None
    _HAS_WANDB = False


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with Flow Matching on TPU", add_help=False)

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
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of dataloader workers per TPU core")
    parser.add_argument("--wandb-project", default=None,
                        help="W&B project name (if provided, wandb will be initialized on rank 0 after 'wandb login')")
    parser.add_argument("--wandb-entity", default=None,
                        help="W&B entity (team/user) if needed")
    parser.add_argument("--ckpt-interval", type=int, default=10,
                        help="Save a full checkpoint every N epochs (master only). 0 = save every epoch")
    parser.add_argument("--resolution", type=int, default=224,
                        help="Resolution of the Image")
    parser.add_argument("--local-crops-number", type=int, default=8,
                        help="Number of local crops")

    return parser


def _mp_fn(index, args):
    """Main training function for each TPU core"""
    torch.manual_seed(42 + index)
    np.random.seed(42 + index)
    
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    
    is_master = rank == 0
    
    if is_master:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)
    else:
        stats_file = None

    use_wandb = False
    if is_master and args.wandb_project is not None and _HAS_WANDB:
        try:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
            use_wandb = True
            wandb.run.summary["script"] = os.path.basename(__file__)
        except Exception as e:
            print(f"Warning: failed to init wandb: {e}", file=sys.stderr)
            use_wandb = False
    elif args.wandb_project is not None and not _HAS_WANDB and is_master:
        print("wandb requested but not installed. Install via `pip install wandb` or unset --wandb-project.",
              file=sys.stderr)

    # Setup data augmentation
    transforms = aug.TrainTransform(args.resolution, local_crops_number=args.local_crops_number)
    
    dataset = datasets.ImageFolder(args.data_dir / "train", transforms)
    
    # XLA distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    assert args.batch_size % world_size == 0
    per_device_batch_size = args.batch_size // world_size
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=False,  # TPU doesn't need pin_memory
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # Wrap with ParallelLoader for XLA
    loader = pl.MpDeviceLoader(loader, device)
    
    # Initialize model
    model = FlowMatching(args).to(device)
    
    # AdamW optimizer with weight decay
    base_lr = args.base_lr * args.batch_size / 256
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=args.wd,
        betas=(0.9, 0.999)
    )
    
    # Load checkpoint if exists
    if (args.exp_dir / "model.pth").is_file():
        if is_master:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt.get("epoch", 0)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0
    
    start_time = last_logging = time.time()
    global_step = start_epoch * len(loader)
    ckpt_interval = max(1, args.ckpt_interval) if args.ckpt_interval != 0 else 1
    save_every_epoch = (args.ckpt_interval == 0)
    
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        epoch_loss_sum = 0.0
        epoch_steps = 0
        
        for step, (crops_list, _) in enumerate(loader, start=epoch * len(loader)):
            # crops_list is a tuple of lists from multi-crop augmentation
            # Each element is a list of 10 crops: [global_1, global_2, local_1, ..., local_8]
            
            # Separate global and local crops
            # crops_list shape: (batch_size, 10, C, H, W) where 10 = 2 global + 8 local
            batch_size = len(crops_list)
            
            # Stack all crops from the batch
            all_global_1 = torch.stack([crops[0] for crops in crops_list]).to(device)
            all_global_2 = torch.stack([crops[1] for crops in crops_list]).to(device)
            all_locals = [torch.stack([crops[i] for crops in crops_list]).to(device) 
                         for i in range(2, 2 + args.local_crops_number)]
            
            # Adjust learning rate with warmup and cosine annealing
            lr = adjust_learning_rate(args, optimizer, loader, step)
            
            optimizer.zero_grad()
            
            # Forward pass
            loss = model(all_global_1, all_global_2, all_locals)
            
            # Backward pass
            loss.backward()
            xm.optimizer_step(optimizer)  # XLA optimizer step
            
            # Get loss value for logging
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
            
            if is_master:
                print(json.dumps(batch_stats))
                if stats_file is not None:
                    print(json.dumps(batch_stats), file=stats_file)
                
                if use_wandb:
                    try:
                        wandb.log({"batch_loss": batch_loss, "lr": lr, "epoch": epoch, "step": step}, step=global_step)
                    except Exception as e:
                        print(f"Warning: wandb.log failed: {e}", file=sys.stderr)
            
            current_time = time.time()
            if is_master and current_time - last_logging > args.log_freq_time:
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
        
        # Synchronize across TPU cores
        xm.rendezvous('epoch_end')
        
        avg_epoch_loss = epoch_loss_sum / max(1, epoch_steps)
        
        if is_master:
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
                # Save on CPU to avoid XLA device issues
                state = {k: v.cpu() for k, v in model.backbone.state_dict().items()}
                ckpt_path = args.exp_dir / "model.pth"
                xm.save(state, ckpt_path)
    
    # Final checkpoint save
    if is_master:
        final_ckpt_path = args.exp_dir / "model_final.pth"
        final_state = dict(
            epoch=args.epochs,
            model={k: v.cpu() for k, v in model.state_dict().items()},
            optimizer={k: v.cpu() if isinstance(v, torch.Tensor) else v 
                      for k, v in optimizer.state_dict().items()},
        )
        xm.save(final_state, final_ckpt_path)
        
        backbone_state = {k: v.cpu() for k, v in model.backbone.state_dict().items()}
        xm.save(backbone_state, args.exp_dir / "resnet50.pth")
        
        if use_wandb:
            try:
                wandb.save(str(final_ckpt_path))
            except Exception:
                pass


def adjust_learning_rate(args, optimizer, loader, step):
    """Cosine annealing with warmup"""
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
    """Build MLP from string specification"""
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

    def forward(self, global_1, global_2, local_crops):
        """
        Args:
            global_1: First global crop (batch_size, 3, 224, 224)
            global_2: Second global crop (batch_size, 3, 224, 224)
            local_crops: List of 8 local crops, each (batch_size, 3, 96, 96)
        """
        # Extract features from global crops (targets)
        feat_global_1 = self.backbone(global_1)
        feat_global_2 = self.backbone(global_2)
        
        target_global_1 = feat_global_1.detach()
        target_global_2 = feat_global_2.detach()
        
        # Extract features from all local crops (contexts)
        feat_locals = [self.backbone(local_crop) for local_crop in local_crops]
        context_locals = [self.projection_head(feat) for feat in feat_locals]
        
        # Flow matching: locals predict globals
        all_pred_velocities = []
        all_true_velocities = []
        
        # Each local crop predicts both global crops
        for context_local in context_locals:
            # Predict global_1 from local
            y_0_1 = torch.randn_like(target_global_1)
            t_1 = torch.rand(target_global_1.shape[0], 1, device=target_global_1.device)
            y_t_1 = t_1 * target_global_1 + (1 - t_1) * y_0_1
            true_velocity_1 = target_global_1 - y_0_1
            
            t_emb_1 = self.time_embedding(t_1)
            t_emb_proj_1 = self.time_projection(t_emb_1)
            
            pred_velocity_1 = self.velocity_predictor(
                torch.cat([y_t_1, context_local, t_emb_proj_1], dim=1)
            )
            
            # Predict global_2 from local
            y_0_2 = torch.randn_like(target_global_2)
            t_2 = torch.rand(target_global_2.shape[0], 1, device=target_global_2.device)
            y_t_2 = t_2 * target_global_2 + (1 - t_2) * y_0_2
            true_velocity_2 = target_global_2 - y_0_2
            
            t_emb_2 = self.time_embedding(t_2)
            t_emb_proj_2 = self.time_projection(t_emb_2)
            
            pred_velocity_2 = self.velocity_predictor(
                torch.cat([y_t_2, context_local, t_emb_proj_2], dim=1)
            )
            
            all_pred_velocities.extend([pred_velocity_1, pred_velocity_2])
            all_true_velocities.extend([true_velocity_1, true_velocity_2])
        
        # Concatenate all predictions and targets
        preds = torch.cat(all_pred_velocities, dim=0)
        truths = torch.cat(all_true_velocities, dim=0)
        
        # MSE loss
        loss = F.mse_loss(preds, truths)
        return loss


def main(args):
    """Entry point for XLA multiprocessing"""
    xmp.spawn(_mp_fn, args=(args,), nprocs=None)  # nprocs=None uses all TPU cores


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Flow Matching training script for TPU", parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
