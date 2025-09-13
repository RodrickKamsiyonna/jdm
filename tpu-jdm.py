#!/usr/bin/env python3
# tpu-jdm-fixed.py
import os
import sys
import time
import math
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader

# your local imports - keep them as in your original repo
import datasets
import augmentations as aug
import resnet

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    wandb = None
    _HAS_WANDB = False

# ------------------------------
# Utility: robust distributed init
# ------------------------------
def init_distributed_mode_from_env(args):
    # read env vars as early as possible
    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # IMPORTANT: set CUDA device before calling init_process_group when using NCCL
    torch.cuda.set_device(args.local_rank)
    # init process group
    dist.init_process_group(backend="nccl", init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    return args

# ------------------------------
# Hugging Face streaming iterable dataset (keeps previous pattern)
# ------------------------------
class HuggingFaceImageNetDataset(IterableDataset):
    def __init__(self, hf_dataset_name, split, transform=None, dataset_size=None):
        self.transform = transform
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self.dataset_size = dataset_size if dataset_size is not None else 1281167

        # get distributed info if available
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1

        # Only rank 0 prints or triggers potential downloads
        if self.rank == 0:
            print(f"[Rank 0] Initializing Hugging Face dataset: {self.hf_dataset_name} ({self.split})", flush=True)

        # load streaming dataset (non-blocking) — token=True may require HF login
        # Wrap in try/except to expose network/auth issues
        try:
            self.dataset = datasets.load_dataset(
                self.hf_dataset_name,
                split=self.split,
                streaming=True,
                token=True
            )
        except Exception as e:
            print(f"[Rank {self.rank}] dataset load failed: {e}", file=sys.stderr, flush=True)
            raise

        # Barrier to sync processes *after* dataset object is created and after device is set
        if dist.is_available() and dist.is_initialized():
            # plain dist.barrier() works because we set device before init_process_group
            dist.barrier()

    def __iter__(self):
        for sample in self.dataset:
            # streaming dataset from timm/imagenet-1k-wds usually has 'jpg' bytes and 'cls' label
            img = sample.get("jpg", None)
            label = sample.get("cls", None)

            # If the dataset returns raw bytes, convert to PIL as your transform expects
            try:
                if isinstance(img, (bytes, bytearray)):
                    from PIL import Image
                    img = Image.open(io.BytesIO(img)).convert("RGB")
            except Exception:
                # If transform expects PIL, skip or log
                pass

            if self.transform:
                # transform should return a tensor
                x1, x2 = self.transform(img)
                # yield a pair (x1, x2) and the label
                yield (x1, x2), label
            else:
                yield img, label

    def get_dataset_size(self):
        return self.dataset_size

# ------------------------------
# Model, time emb, mlp builders (kept from your original)
# ------------------------------
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

# ------------------------------
# LR schedule helper (kept)
# ------------------------------
def adjust_learning_rate(args, optimizer, steps_per_epoch, global_step):
    max_steps = args.epochs * steps_per_epoch
    warmup_steps = 10 * steps_per_epoch
    base_lr = args.base_lr * args.batch_size / 256
    step = global_step
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

# ------------------------------
# Main
# ------------------------------
def get_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with Flow Matching", add_help=False)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--exp-dir", type=Path, default="./exp")
    parser.add_argument("--log-freq-time", type=int, default=60)
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--projector-mlp", default="8192-8192-512")
    parser.add_argument("--time-emb-dim", type=int, default=128)
    parser.add_argument("--time-emb-mlp", default="128-512")
    parser.add_argument("--velocity-mlp", default="1536-1024-512")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--base-lr", type=float, default=0.2)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--ckpt-interval", type=int, default=10)
    return parser

def main(args):
    # IMPORTANT: read env & set device before initializing process group
    args = init_distributed_mode_from_env(args)

    # local device for tensors
    local_rank = args.local_rank
    device = torch.device(f"cuda:{local_rank}")

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv), file=stats_file)
    else:
        stats_file = None

    # wandb only on rank 0
    use_wandb = False
    if args.rank == 0 and args.wandb_project is not None and _HAS_WANDB:
        try:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
            use_wandb = True
        except Exception as e:
            print(f"Warning: failed to init wandb: {e}", file=sys.stderr)

    # transforms (adapt to your aug implementation)
    transforms = aug.TrainTransform()

    dataset = HuggingFaceImageNetDataset(
        hf_dataset_name="timm/imagenet-1k-wds",
        split="train",
        transform=transforms,
        dataset_size=1281167
    )

    # batch size per device
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

    loader = DataLoader(dataset, **dataloader_kwargs)

    imagenet_train_size = dataset.get_dataset_size()
    steps_per_epoch = math.ceil(imagenet_train_size / args.batch_size)
    if args.rank == 0:
        print(f"Using an estimated {steps_per_epoch} steps per epoch.", flush=True)

    # model -> device and wrap DDP
    model = FlowMatching(args).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=False, gradient_as_bucket_view=True
    )

    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    start_epoch = 0
    scaler = torch.amp.GradScaler(device="cuda")
    start_time = last_logging = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for step, ((x, y), _) in enumerate(loader):
            global_step = epoch * steps_per_epoch + step

            # move to correct device
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            lr = adjust_learning_rate(args, optimizer, steps_per_epoch, global_step)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            epoch_loss_sum += batch_loss
            epoch_steps += 1

            if args.rank == 0:
                batch_stats = dict(epoch=epoch, step=step, global_step=global_step, batch_loss=batch_loss, time=int(time.time() - start_time), lr=lr)
                print(json.dumps(batch_stats), flush=True)
                if stats_file is not None:
                    print(json.dumps(batch_stats), file=stats_file, flush=True)
                if use_wandb:
                    try:
                        wandb.log({"batch_loss": batch_loss, "lr": lr, "epoch": epoch, "step": step}, step=global_step)
                    except Exception as e:
                        print(f"Warning: wandb.log failed: {e}", file=sys.stderr, flush=True)

            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                stats = dict(epoch=epoch, step=step, loss=batch_loss, time=int(current_time - start_time), lr=lr)
                print(json.dumps(stats), flush=True)
                if stats_file is not None:
                    print(json.dumps(stats), file=stats_file, flush=True)
                last_logging = current_time

        avg_epoch_loss = epoch_loss_sum / max(1, epoch_steps)
        if args.rank == 0:
            epoch_stats = dict(epoch=epoch, epoch_loss=avg_epoch_loss, steps=epoch_steps, time=int(time.time() - start_time))
            print("EPOCH_SUMMARY: " + json.dumps(epoch_stats), flush=True)
            if stats_file is not None:
                print("EPOCH_SUMMARY: " + json.dumps(epoch_stats), file=stats_file, flush=True)

            if use_wandb:
                try:
                    wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch}, step=global_step)
                except Exception as e:
                    print(f"Warning: wandb.log failed at epoch end: {e}", file=sys.stderr, flush=True)

            should_save = ((epoch + 1) % args.ckpt_interval == 0) or (args.ckpt_interval == 0)
            if should_save:
                state = dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict())
                ckpt_path = args.exp_dir / "model.pth"
                torch.save(state, ckpt_path)

    if args.rank == 0:
        final_ckpt_path = args.exp_dir / "model_final.pth"
        final_state = dict(epoch=args.epochs, model=model.state_dict(), optimizer=optimizer.state_dict())
        torch.save(final_state, final_ckpt_path)
        torch.save(model.module.backbone.state_dict(), args.exp_dir / "resnet50.pth")
        if use_wandb:
            try:
                wandb.save(str(final_ckpt_path))
            except Exception:
                pass

if __name__ == "__main__":
    parser = get_arguments()
    args = parser.parse_args()
    main(args)
