# evaluate.py
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


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )

    # Data
    parser.add_argument("--data-dir", type=Path, help="path to dataset")
    parser.add_argument(
        "--train-percent",
        default=100,
        type=int,
        choices=(100, 10, 1),
        help="size of training set in percent",
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
        default=4,
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
            # store on CPU to avoid holding GPU memory
            features_list.append(features.cpu())
            targets_list.append(targets)
    all_features = torch.cat(features_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)

    mean = all_features.mean(dim=0, keepdim=True)
    std = all_features.std(dim=0, keepdim=True) + 1e-6  # Avoid division by zero

    return all_features, all_targets, mean, std


def standardize_features(features, mean, std):
    # features, mean, std should be on same device before calling this
    return (features - mean) / std


def main():
    parser = get_arguments()
    args = parser.parse_args()

    # optional remote subset listing
    if args.train_percent in {1, 10}:
        try:
            args.train_files = urllib.request.urlopen(
                f"https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{args.train_percent}percent.txt"
            ).readlines()
        except Exception:
            print("Warning: failed to download train_percent file. Expecting local dataset layout.")
            args.train_files = None

    # decide distributed mode
    # If torchrun/env vars are present -> use them
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.rank = int(os.environ.get("RANK", 0))
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.dist_url = "env://"
        args.use_torchrun = True
    else:
        # fallback: detect available GPUs and spawn if >1
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node
        args.rank = 0
        args.local_rank = 0
        args.dist_url = f"tcp://localhost:{random.randrange(49152, 65535)}"
        args.use_torchrun = False

    # SLURM handlers (unchanged)
    if "SLURM_JOB_ID" in os.environ:
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)

    # Launch worker(s)
    if args.use_torchrun:
        # torchrun already spawned processes, just call main_worker with LOCAL_RANK
        main_worker(args.local_rank, args)
    else:
        # if multiple GPUs, spawn; otherwise call directly (single GPU)
        if args.world_size > 1:
            torch.multiprocessing.spawn(main_worker, (args,), args.world_size)
        else:
            main_worker(0, args)


def main_worker(gpu, args):
    # gpu is the local GPU index for this process
    args.gpu = gpu
    # Initialize distributed if world_size > 1
    distributed = args.world_size > 1

    if distributed:
        # Use env:// when launched with torchrun, otherwise the tcp:// we set earlier.
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=(args.rank + gpu) if not args.use_torchrun else int(os.environ.get("RANK", 0)),
        )

    # only rank 0 handles logs/stat file creation
    rank = int(os.environ.get("RANK", args.rank + (gpu if not args.use_torchrun else 0)))
    args.rank = rank

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)
    else:
        stats_file = None

    # set device
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Build backbone
    backbone, embedding_dim = resnet.__dict__[args.arch](zero_init_residual=True)

    # Load pretrained state dict robustly
    state_dict = torch.load(args.pretrained, map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    # if this is a wrapped checkpoint, try to unwrap keys
    try:
        first_key = list(state_dict.keys())[0]
        if "backbone" in first_key or first_key.startswith("module.backbone"):
            state_dict = {
                key.replace("module.backbone.", "").replace("backbone.", ""): value
                for (key, value) in state_dict.items()
            }
    except Exception:
        pass

    # load with strict=False to avoid failure on small key mismatches
    backbone.load_state_dict(state_dict, strict=False)

    # head and model
    head = nn.Linear(embedding_dim, 1000)
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()
    model = nn.Sequential(backbone, head)
    model.to(device)

    if args.weights == "freeze":
        backbone.requires_grad_(False)
        head.requires_grad_(True)

    if distributed:
        # wrap with DDP
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu] if torch.cuda.is_available() else None)

    criterion = nn.CrossEntropyLoss().to(device)

    # optimizer
    param_groups = [dict(params=head.parameters(), lr=args.lr_head)]
    if args.weights == "finetune":
        param_groups.append(dict(params=backbone.parameters(), lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # resume if checkpoint exists (non-strict restore, safest to load optimizer if present)
    start_epoch = 0
    if (args.exp_dir / "checkpoint.pth").is_file():
        ckpt = torch.load(args.exp_dir / "checkpoint.pth", map_location="cpu")
        start_epoch = ckpt.get("epoch", 0)
        best_acc = ckpt.get("best_acc", argparse.Namespace(top1=0, top5=0))
        try:
            model.load_state_dict(ckpt["model"])
        except Exception:
            print("Warning: failed to load full model state; continuing with loaded backbone + fresh head.")
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass
    else:
        best_acc = argparse.Namespace(top1=0, top5=0)

    # prepare datasets and loaders
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

    # batch size per process; avoid zero
    per_process_batch = max(1, args.batch_size // max(1, args.world_size))
    kwargs = dict(
        batch_size=per_process_batch,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

    # Training dataset
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
    if args.train_percent in {1, 10} and getattr(args, "train_files", None):
        # rebuild samples list from provided file names
        train_dataset.samples = []
        for fname in args.train_files:
            fname = fname.decode().strip()
            cls = fname.split("_")[0]
            train_dataset.samples.append(
                (str(traindir / cls / fname), train_dataset.class_to_idx[cls])
            )

    # distributed sampler only in distributed mode
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size if not distributed else per_process_batch,
        num_workers=args.workers, pin_memory=True, shuffle=(train_sampler is None)
    )

    # Precompute mean and std for standardization (only when freezing the backbone)
    mean = None
    std = None
    if args.weights == "freeze":
        if args.rank == 0:
            print("Extracting features and computing mean/std...")
            # Use backbone in eval and extract features (on device)
            features, targets, mean_cpu, std_cpu = extract_features_and_stats(train_loader, backbone, device)
            # mean/std are on CPU; we will broadcast GPU copies if distributed
            mean = mean_cpu
            std = std_cpu
            # standardize features on CPU
            standardized_features = standardize_features(features, mean, std)
            standardized_dataset = TensorDataset(standardized_features, targets)
            # replace train_loader with standardized CPU features loader
            train_loader = torch.utils.data.DataLoader(
                standardized_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True
            )

        # synchronize mean/std across processes (if distributed)
        if distributed:
            # rank 0 needs to send mean/std to others. Move to GPU for broadcast.
            if args.rank == 0:
                mean_gpu = mean.to(device)
                std_gpu = std.to(device)
            else:
                mean_gpu = torch.zeros(1, embedding_dim, device=device)
                std_gpu = torch.ones(1, embedding_dim, device=device)
            torch.distributed.barrier()
            torch.distributed.broadcast(mean_gpu, src=0)
            torch.distributed.broadcast(std_gpu, src=0)
            # ensure all ranks have CPU copies (we'll keep features on CPU)
            mean = mean_gpu.cpu()
            std = std_gpu.cpu()
        else:
            # single-process: mean/std already present on CPU
            pass

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):

        if args.weights == "finetune":
            model.train()
        elif args.weights == "freeze":
            model.eval()  # Backbone frozen; head will be used on features or backbone outputs

        # if distributed and using a DistributedSampler, set epoch for shuffling
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for step, (features_or_images, target) in enumerate(
            train_loader, start=epoch * max(1, len(train_loader))
        ):
            target = target.to(device, non_blocking=True)
            if args.weights == "freeze":
                # features are either tensors (CPU) or already on device; ensure on device
                if features_or_images.device.type == "cpu":
                    features = features_or_images.to(device, non_blocking=True)
                else:
                    features = features_or_images.to(device, non_blocking=True)
                output = head(features)
            else:
                images = features_or_images.to(device, non_blocking=True)
                output = model(images)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.print_freq == 0:
                # reduce loss across processes only when distributed
                if distributed:
                    loss_reduced = loss.detach()
                    torch.distributed.reduce(loss_reduced, dst=0)
                    if args.rank == 0:
                        reduced_loss = (loss_reduced / args.world_size).item()
                    else:
                        reduced_loss = None
                else:
                    reduced_loss = loss.item()

                if args.rank == 0:
                    pg = optimizer.param_groups
                    lr_head = pg[0]["lr"]
                    lr_backbone = pg[1]["lr"] if len(pg) == 2 else 0
                    stats = dict(
                        epoch=epoch, step=step, lr_backbone=lr_backbone,
                        lr_head=lr_head, loss=reduced_loss,
                        time=int(time.time() - start_time),
                    )
                    print(json.dumps(stats))
                    if stats_file:
                        print(json.dumps(stats), file=stats_file)

        # evaluate
        if args.rank == 0:
            model.eval()
            top1 = AverageMeter("Acc@1")
            top5 = AverageMeter("Acc@5")
            with torch.no_grad():
                for images, target in val_loader:
                    images = images.to(device, non_blocking=True)
                    output = model(images)
                    acc1, acc5 = accuracy(
                        output, target.to(device, non_blocking=True), topk=(1, 5)
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
            if stats_file:
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
