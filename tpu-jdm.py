# train_flowmatching_tpu.py
# Adapted for TPU training with TFRecords

import os
import sys
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
import torch_xla.debug.metrics as met

# Assuming you have a way to parse TFRecords into PyTorch tensors
# This example uses a placeholder. You'll need to implement `TFRecordImageFolderDataset`.
# Consider using libraries like `tfrecord` or `tensorflow` directly.
# from tfrecord_dataset import TFRecordImageFolderDataset # Placeholder import
import tfrecord_dataset # Placeholder import for your TFRecord dataset implementation

import augmentations as aug
import resnet as resnet # Use the TPU-adapted ResNet

# --- LARS Optimizer (unchanged) ---
def exclude_bias_and_norm(p):
    return p.ndim == 1

class LARS(torch.optim.Optimizer):
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

# --- Model Components (unchanged, assuming they are TPU compatible) ---
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1) # Use math.log
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        embeddings = t * freqs
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def build_mlp(spec, last_layer_bias=False):
    layers = []
    f = list(map(int, spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        # BatchNorm1d might need adjustment for TPU, but often works
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

# --- Learning Rate Scheduler (unchanged) ---
def adjust_learning_rate(args, optimizer, loader_len, step):
    max_steps = args.epochs * loader_len
    warmup_steps = 10 * loader_len
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

# --- Main Training Function (adapted for TPU) ---
def _mp_fn(index, args):
    """Multiprocess function for XLA."""
    torch.manual_seed(42) # Set seed for reproducibility

    # Initialize XLA distributed environment
    device = xm.xla_device()
    xm.master_print(f"Process {index} using device: {device}")

    # Setup experiment directory (only on master)
    if xm.is_master_ordinal():
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        xm.master_print(" ".join(sys.argv))
        xm.master_print(" ".join(sys.argv), file=stats_file)
    else:
        stats_file = None

    # Transforms (assuming they are TPU compatible)
    transforms = aug.TrainTransform()

    # --- Data Loading (adapted for TFRecords) ---
    # You need to implement `tfrecord_dataset.TFRecordImageFolderDataset`
    # This is a placeholder structure. You'll need to write the actual dataset class.
    try:
        dataset_train = tfrecord_dataset.TFRecordImageFolderDataset(
            data_dir=args.data_dir / "train",
            transform=transforms,
            # Add other necessary arguments for your TFRecord parser
        )
        # Create a DistributedSampler-like behavior using sharding in tf.data if needed,
        # or rely on XLA's parallel loader's sharding.
        # PyTorch DataLoader with TFRecord dataset
        loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size // xm.xrt_world_size(), # Per core batch size
            num_workers=args.num_workers,
            pin_memory=False, # Not needed for TPU
            shuffle=False, # Shuffling handled by tf.data or XLA loader if needed
            drop_last=True # Often recommended for distributed training
        )
        xm.master_print(f"Created DataLoader with {len(loader)} steps per epoch")
    except Exception as e:
        xm.master_print(f"Error creating DataLoader: {e}")
        raise e

    # --- Model, Optimizer, DDP (adapted for TPU) ---
    model = FlowMatching(args).to(device)
    # SyncBatchNorm might behave differently on TPU. Test carefully.
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # Might not be needed or behave differently

    optimizer = LARS(
        model.parameters(),
        lr=0, # Initial LR will be set by adjust_learning_rate
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    # Resume from checkpoint logic (adapted for TPU)
    start_epoch = 0
    if (args.exp_dir / "model.pth").is_file():
        xm.master_print("Resuming from checkpoint...")
        # Load checkpoint on CPU first, then move to device
        ckpt = torch.load(args.exp_dir / "model.pth", map_location='cpu')
        start_epoch = ckpt.get("epoch", 0)
        # Use strict=False if you have issues with keys (e.g., from SyncBatchNorm)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        xm.master_print(f"Resumed from epoch {start_epoch}")

    # Wrap model for XLA DDP
    model = torch_xla.distributed.parallel_loader.MpModelWrapper(model)

    # --- Training Loop ---
    start_time = last_logging = time.time()
    global_step = start_epoch * len(loader)

    # Use ParallelLoader for efficient data loading on TPU
    loader = pl.ParallelLoader(loader, [device])

    for epoch in range(start_epoch, args.epochs):
        model.train()
        para_loader = loader.per_device_loader(device)
        xm.master_print(f"Starting epoch {epoch}")

        epoch_loss_sum = 0.0
        epoch_steps = 0

        for step, ((x, y), _) in enumerate(para_loader): # Iterate over parallel loader
            x = x.to(device)
            y = y.to(device)

            lr = adjust_learning_rate(args, optimizer, len(loader), global_step)

            optimizer.zero_grad()
            # AMP might not be directly applicable or needed on TPU as XLA handles it.
            # If needed, investigate torch_xla.amp or XLA-specific mixed precision settings.
            # with torch.amp.autocast(device_type="xla"): # Might not be supported or necessary
            loss = model(x, y) # Call the wrapped model

            loss.backward()
            # Use XLA's optimizer step instead of scaler.step
            xm.optimizer_step(optimizer) # This handles the XLA device transfer and step

            # Reduce loss across replicas for logging (optional but good practice)
            reduced_loss = xm.mesh_reduce('loss_reduce', loss, lambda x: sum(x) / len(x))

            batch_loss = reduced_loss.item()
            epoch_loss_sum += batch_loss
            epoch_steps += 1
            global_step += 1

            # Logging (only on master)
            if xm.is_master_ordinal():
                batch_stats = dict(
                    epoch=epoch,
                    step=step,
                    global_step=global_step,
                    batch_loss=batch_loss, # Use reduced loss for logging
                    time=int(time.time() - start_time),
                    lr=lr,
                )
                xm.master_print(json.dumps(batch_stats))
                if stats_file is not None:
                    xm.master_print(json.dumps(batch_stats), file=stats_file)

                current_time = time.time()
                if current_time - last_logging > args.log_freq_time:
                    xm.master_print(json.dumps(batch_stats)) # Log periodically
                    if stats_file is not None:
                        xm.master_print(json.dumps(batch_stats), file=stats_file)
                    last_logging = current_time

        # Epoch end logic (only on master)
        if xm.is_master_ordinal():
            avg_epoch_loss = epoch_loss_sum / max(1, epoch_steps)
            epoch_stats = dict(
                epoch=epoch,
                epoch_loss=avg_epoch_loss,
                steps=epoch_steps,
                time=int(time.time() - start_time),
            )
            xm.master_print("EPOCH_SUMMARY: " + json.dumps(epoch_stats))
            if stats_file is not None:
                xm.master_print("EPOCH_SUMMARY: " + json.dumps(epoch_stats), file=stats_file)

            # Save checkpoint (only on master, every N epochs or at the end)
            should_save = ((epoch + 1) % max(1, args.ckpt_interval) == 0) or (epoch == args.epochs - 1)
            if should_save:
                xm.master_print(f"Saving checkpoint at epoch {epoch + 1}")
                # Get the underlying model state dict
                model_state = model._model.state_dict() # Access the actual model inside MpModelWrapper
                state = dict(
                    epoch=epoch + 1,
                    model=model_state,
                    optimizer=optimizer.state_dict(),
                )
                ckpt_path = args.exp_dir / "model.pth"
                # Save from master process only
                xm.save(state, ckpt_path, master_only=True)
                xm.master_print(f"Checkpoint saved to {ckpt_path}")

    # Final save (only on master)
    if xm.is_master_ordinal():
        xm.master_print("Saving final checkpoint")
        final_model_state = model._model.state_dict()
        final_state = dict(
            epoch=args.epochs,
            model=final_model_state,
            optimizer=optimizer.state_dict(),
        )
        final_ckpt_path = args.exp_dir / "model_final.pth"
        xm.save(final_state, final_ckpt_path, master_only=True)
        xm.master_print(f"Final checkpoint saved to {final_ckpt_path}")

        # Save backbone weights
        backbone_path = args.exp_dir / f"{args.arch}.pth"
        xm.save(model._model.backbone.state_dict(), backbone_path, master_only=True)
        xm.master_print(f"Backbone weights saved to {backbone_path}")

    # Print XLA metrics (optional, useful for debugging)
    # xm.master_print(met.metrics_report())


def main():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with Flow Matching on TPU", add_help=False)

    # Arguments (mostly unchanged)
    parser.add_argument("--data-dir", type=Path, default="/path/to/imagenet", required=True,
                        help="Path to the dataset folder containing /train")
    parser.add_argument("--exp-dir", type=Path, default="./exp_tpu",
                        help="Path to the experiment folder")
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help="Print logs every [log-freq-time] seconds")
    parser.add_argument("--arch", type=str, default="resnet50",
                        help="Architecture")
    parser.add_argument("--projector-mlp", default="8192-8192-512",
                        help="Projector MLP")
    parser.add_argument("--time-emb-dim", type=int, default=128,
                        help="Time embedding dim")
    parser.add_argument("--time-emb-mlp", default="128-512",
                        help="Time embedding MLP")
    parser.add_argument("--velocity-mlp", default="1536-1024-512", # Check input dim calculation
                        help="Velocity predictor MLP")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2048, # Effective batch size
                        help="Effective batch size (total across all cores)")
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help="Base learning rate, scaled by batch size / 256")
    parser.add_argument("--wd", type=float, default=1e-6,
                        help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, # Reduce for TPU if needed
                        help="Number of dataloader workers")

    # TPU specific arguments (can be passed via xmp.spawn or environment)
    # These might be handled by xmp.spawn automatically
    # parser.add_argument("--num_cores", type=int, default=8) # Usually set by TPU runtime

    args = parser.parse_args()

    # Start distributed training using xmp.spawn
    xmp.spawn(_mp_fn, args=(args,), nprocs=8, start_method='fork') # Adjust nprocs if needed


if __name__ == "__main__":
    main()
