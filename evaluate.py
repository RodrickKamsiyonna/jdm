# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib

from torch import nn, optim
from torchvision import datasets, transforms
import torch
import numpy as np
from torch.utils.data import TensorDataset


import resnet


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb_scale = np.log(10000) / (half_dim - 1)
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
    def __init__(self, args, arch='resnet50', projector_mlp= '2048-4096-2048', time_emb_dim= '1024', time_emb_mlp='1024-2048-1024' , velocity_mlp = '5120-10240-2048'):
        super().__init__()
        self.backbone, self.embedding = resnet.__dict__[arch](
            zero_init_residual=True
        )
        self.projection_head = build_mlp(projector_mlp)
        self.time_embedding = SinusoidalTimeEmbedding(dim=time_emb_dim)
        self.time_projection = build_mlp(time_emb_mlp)
        self.velocity_predictor = build_mlp(velocity_mlp)

    def forward(self, x, y):
        # This forward pass is for training, we only need the components for sampling
        raise NotImplementedError("This model is intended for sampling in this script.")

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )
    
    parser.add_argument("--evaluation-mode", type=str, default="linear_probe", 
                        choices=['linear_probe', 'single_shot_flow'],
                        help="Choose the evaluation method.")
    parser.add_argument("--flow-model-path", type=Path, 
                        help="Path to the pretrained flow matching model (required for single_shot_flow mode)")
    parser.add_argument("--num-flow-samples", type=int, default=50,
                        help="Number of synthetic samples to generate per class for the linear head training.")
    parser.add_argument("--sampling-steps", type=int, default=100,
                        help="Number of integration steps for flow matching sampling.")

    # Data
    parser.add_argument("--data-dir", type=Path, help="path to dataset")
    parser.add_argument(
        "--train-percent",
        default=100,
        type=int,
        choices=(100, 10, 1),
        help="size of traing set in percent (only for 'linear_probe' mode)",
    )

    # Checkpoint
    parser.add_argument("--pretrained", type=Path, required=True, help="path to pretrained backbone model")
    parser.add_argument(
        "--exp-dir",
        default="./checkpoint/lincls/",
        type=Path,
        metavar="DIR",
        help="path to checkpoint directory",
    )
    parser.add_argument(
        "--print-freq", default=100, type=int, metavar="N", help="print frequency"
    )

    # Model
    parser.add_argument("--arch", type=str, default="resnet50")

    # Optim
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--lr-backbone",
        default=0.0,
        type=float,
        metavar="LR",
        help="backbone base learning rate",
    )
    parser.add_argument(
        "--lr-head",
        default=0.3,
        type=float,
        metavar="LR",
        help="classifier base learning rate",
    )
    parser.add_argument(
        "--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument(
        "--weights",
        default="freeze",
        type=str,
        choices=("finetune", "freeze"),
        help="finetune or freeze resnet weights",
    )

    # Running
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )

    return parser


def extract_features_and_stats(loader, backbone, device):
    backbone.eval()
    features_list = []
    targets_list = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            features = backbone(images)
            features_list.append(features.cpu())
            targets_list.append(targets)
    all_features = torch.cat(features_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)

    mean = all_features.mean(dim=0, keepdim=True)
    std = all_features.std(dim=0, keepdim=True) + 1e-6  # Avoid division by zero

    return all_features, all_targets, mean, std


def standardize_features(features, mean, std):
    return (features - mean) / std


def main():
    parser = get_arguments()
    args = parser.parse_args()
    
    if args.evaluation_mode == 'single_shot_flow' and not args.flow_model_path:
        raise ValueError("--flow-model-path is required for 'single_shot_flow' evaluation mode.")

    if args.train_percent in {1, 10}:
        args.train_files = urllib.request.urlopen(
            f"https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{args.train_percent}percent.txt"
        ).readlines()
    args.ngpus_per_node = torch.cuda.device_count()
    if "SLURM_JOB_ID" in os.environ:
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
    # single-node distributed training
    args.rank = 0
    args.dist_url = f"tcp://localhost:{random.randrange(49152, 65535)}"
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    backbone, embedding_dim = resnet.__dict__[args.arch](zero_init_residual=True)
    state_dict = torch.load(args.pretrained, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    if "backbone" in list(state_dict.keys())[0]:
        state_dict = {
            key.replace("module.backbone.", ""): value
            for (key, value) in state_dict.items()
        }
    backbone.load_state_dict(state_dict, strict=True) # Use strict=True for backbone

    head = nn.Linear(embedding_dim, 1000)
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()
    model = nn.Sequential(backbone, head)
    model.cuda(gpu)

    if args.weights == "freeze":
        backbone.requires_grad_(False)
        head.requires_grad_(True)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    criterion = nn.CrossEntropyLoss().cuda(gpu)

    param_groups = [dict(params=head.parameters(), lr=args.lr_head)]
    if args.weights == "finetune":
        param_groups.append(dict(params=backbone.parameters(), lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # automatically resume from checkpoint if it exists
    if (args.exp_dir / "checkpoint.pth").is_file():
        ckpt = torch.load(args.exp_dir / "checkpoint.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        best_acc = ckpt["best_acc"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        start_epoch = 0
        best_acc = argparse.Namespace(top1=0, top5=0)

    traindir = args.data_dir / "train"
    valdir = args.data_dir / "val"
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    
    kwargs = dict(
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

    # --- MODIFIED: Logic to switch between evaluation modes ---
    if args.evaluation_mode == 'linear_probe':
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        if args.train_percent in {1, 10}:
            train_dataset.samples = []
            for fname in args.train_files:
                fname = fname.decode().strip()
                cls = fname.split("_")[0]
                train_dataset.samples.append(
                    (traindir / cls / fname, train_dataset.class_to_idx[cls])
                )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=train_sampler, **kwargs
        )
    
    elif args.evaluation_mode == 'single_shot_flow':
        if args.rank == 0:
            print("--- Running Single-Shot Flow Evaluation ---")
            
            # 1. Load the Flow Matching model
            flow_model = FlowMatching(args, arch=args.arch)
            flow_model_ckpt = torch.load(args.flow_model_path, map_location='cpu')['model']
            flow_model.load_state_dict(flow_model_ckpt)
            flow_model.cuda(gpu)
            flow_model.eval()
            print(f"Flow Matching model loaded from {args.flow_model_path}")

            # 2. Get one image per class from the validation set
            single_shot_data = {}
            for img, label in val_dataset:
                if label not in single_shot_data:
                    single_shot_data[label] = img.unsqueeze(0).cuda(gpu)
                if len(single_shot_data) == 1000:
                    break
            
            print(f"Collected {len(single_shot_data)} single-shot images for context.")

            # 3. Generate synthetic training data
            generated_features = []
            generated_labels = []
            
            backbone.eval()
            with torch.no_grad():
                for label, img_tensor in single_shot_data.items():
                    print(f"Generating {args.num_flow_samples} samples for class {label}...")
                    
                    # Get context vector from the single image
                    context_feature = backbone(img_tensor)
                    projected_context = flow_model.projection_head(context_feature)
                    
                    # Generate samples
                    synthetic_samples = sample_with_flow_matching(
                        flow_model,
                        projected_context,
                        num_samples=args.num_flow_samples,
                        num_steps=args.sampling_steps,
                        device=gpu
                    )
                    
                    # Add original context vector and synthetic samples to our new training set
                    generated_features.append(context_feature.cpu())
                    generated_features.append(synthetic_samples.cpu())
                    
                    labels_tensor = torch.full((args.num_flow_samples + 1,), label, dtype=torch.long)
                    generated_labels.append(labels_tensor)

            all_features = torch.cat(generated_features, dim=0)
            all_labels = torch.cat(generated_labels, dim=0)
            
            print(f"Generated a synthetic training set with {all_features.shape[0]} samples.")

            # 4. Create a new DataLoader for the synthetic features
            synthetic_dataset = TensorDataset(all_features, all_labels)
            # We train on rank 0 and broadcast the trained head, so no sampler needed here.
            # Batch size can be larger as we are not loading images
            train_loader = torch.utils.data.DataLoader(
                synthetic_dataset, 
                batch_size=min(1024, all_features.shape[0]), 
                shuffle=True
            )
        
        # Synchronize all processes to wait for rank 0 to finish generation
        torch.distributed.barrier()

    # Precompute mean and std for standardization (only for linear_probe mode)
    mean = None
    std = None
    if args.evaluation_mode == 'linear_probe' and args.weights == "freeze":
        if args.rank == 0:
            print("Extracting features and computing mean/std...")
            features, targets, mean, std = extract_features_and_stats(train_loader, backbone, gpu)
            mean = mean.cuda(gpu)
            std = std.cuda(gpu)
            # Replace train_loader with standardized features
            standardized_features = standardize_features(features, mean, std)
            standardized_dataset = TensorDataset(standardized_features, targets)
            train_loader = torch.utils.data.DataLoader(
                standardized_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True
            )
        torch.distributed.barrier()
        if args.rank != 0:
            mean = torch.zeros(1, embedding_dim).cuda(gpu)
            std = torch.ones(1, embedding_dim).cuda(gpu)
        torch.distributed.broadcast(mean, 0)
        torch.distributed.broadcast(std, 0)

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        
        # --- MODIFIED: Handle training logic based on mode ---
        if args.evaluation_mode == 'linear_probe':
            if args.weights == "finetune":
                model.train()
            elif args.weights == "freeze":
                model.eval() # Backbone is frozen, only head is trained
            
            for step, (features_or_images, target) in enumerate(
                train_loader, start=epoch * len(train_loader)
            ):
                target = target.cuda(gpu, non_blocking=True)
                if args.weights == "freeze":
                    features = features_or_images.cuda(gpu, non_blocking=True)
                    output = head(features)
                else:
                    images = features_or_images.cuda(gpu, non_blocking=True)
                    output = model(images)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % args.print_freq == 0:
                    torch.distributed.reduce(loss.div_(args.world_size), 0)
                    if args.rank == 0:
                        pg = optimizer.param_groups
                        lr_head = pg[0]["lr"]
                        lr_backbone = pg[1]["lr"] if len(pg) == 2 else 0
                        stats = dict(
                            epoch=epoch, step=step, lr_backbone=lr_backbone,
                            lr_head=lr_head, loss=loss.item(),
                            time=int(time.time() - start_time),
                        )
                        print(json.dumps(stats))
                        print(json.dumps(stats), file=stats_file)
        
        elif args.evaluation_mode == 'single_shot_flow':
             # New training loop for training the head on generated features
            if args.rank == 0:
                head.train() # Only train the head
                for step, (features, target) in enumerate(train_loader):
                    features = features.cuda(gpu, non_blocking=True)
                    target = target.cuda(gpu, non_blocking=True)
                    
                    output = head(features) # Pass features directly to the head
                    loss = criterion(output, target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if step % args.print_freq == 0:
                        stats = dict(
                            epoch=epoch, step=step, lr_head=optimizer.param_groups[0]["lr"],
                            loss=loss.item(), time=int(time.time() - start_time),
                        )
                        print(json.dumps(stats))
                        print(json.dumps(stats), file=stats_file)
            
            # Broadcast the trained head from rank 0 to all other processes
            torch.distributed.barrier()
            for param in head.parameters():
                torch.distributed.broadcast(param.data, 0)
        
        # evaluate
        model.eval()
        if args.rank == 0:
            top1 = AverageMeter("Acc@1")
            top5 = AverageMeter("Acc@5")
            with torch.no_grad():
                for images, target in val_loader:
                    output = model(images.cuda(gpu, non_blocking=True))
                    acc1, acc5 = accuracy(
                        output, target.cuda(gpu, non_blocking=True), topk=(1, 5)
                    )
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
            best_acc.top1 = max(best_acc.top1, top1.avg)
            best_acc.top5 = max(best_acc.top5, top5.avg)
            stats = dict(
                epoch=epoch, acc1=top1.avg, acc5=top5.avg,
                best_acc1=best_acc.top1, best_acc5=best_acc.top5,
            )
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_file)

        scheduler.step()
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                best_acc=best_acc,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
            )
            torch.save(state, args.exp_dir / "checkpoint.pth")


# --- NEW: Sampling function ---
@torch.no_grad()
def sample_with_flow_matching(flow_model, context_vector, num_samples, num_steps, device):
    """
    Generates samples using the flow matching model via Euler integration.
    """
    # Repeat context for batch generation
    context = context_vector.repeat(num_samples, 1)
    
    # Start with random noise from a standard normal distribution
    y_t = torch.randn(num_samples, flow_model.embedding, device=device)
    
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((num_samples, 1), i * dt, device=device)
        
        # Predict velocity
        t_emb = flow_model.time_embedding(t)
        t_emb_proj = flow_model.time_projection(t_emb)
        
        velocity = flow_model.velocity_predictor(torch.cat([y_t, context, t_emb_proj], dim=1))
        
        # Update sample using Euler step
        y_t = y_t + velocity * dt
        
    return y_t
# --- END NEW ---

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()

def handle_sigterm(signum, frame):
    pass

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == "__main__":
    main()
