# evaluate_xla.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Linear evaluation on TPU with torch_xla

from pathlib import Path
import argparse
import json
import sys
import time

from torch import nn, optim
from torchvision import datasets, transforms
import torch

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

import resnet


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet with TPU"
    )

    # Data
    parser.add_argument("--data-dir", type=Path, required=True, help="path to dataset")
    parser.add_argument(
        "--train-percent",
        default=100,
        type=int,
        choices=(100, 10, 1),
        help="size of training set in percent",
    )

    # Checkpoint
    parser.add_argument("--pretrained", type=Path, required=True, help="path to pretrained model")
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
    parser.add_argument(
        "--resolution",
        default=224,
        type=int,
        help="Resolution of the image",
    )
    parser.add_argument(
        "--classes",
        default=1000,
        type=int,
        help="Number of classes",
    )
    
    return parser


def _mp_fn(index, args):
    """Main evaluation function for each TPU core"""
    torch.manual_seed(42)
    
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
    
    # Load pretrained backbone
    backbone, embedding = resnet.__dict__[args.arch](zero_init_residual=True)
    state_dict = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    
    if "model" in state_dict:
        state_dict = state_dict["model"]
        state_dict = {
            key.replace("module.backbone.", "").replace("module.", ""): value
            for (key, value) in state_dict.items()
        }
    
    backbone.load_state_dict(state_dict, strict=False)
    
    # Create linear head
    head = nn.Linear(embedding, args.classes)
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()
    
    model = nn.Sequential(backbone, head)
    model = model.to(device)
    
    # Freeze or finetune backbone
    if args.weights == "freeze":
        backbone.requires_grad_(False)
        head.requires_grad_(True)
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Optimizer
    param_groups = [dict(params=head.parameters(), lr=args.lr_head)]
    if args.weights == "finetune":
        param_groups.append(dict(params=backbone.parameters(), lr=args.lr_backbone))
    
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # Load checkpoint if exists
    if (args.exp_dir / "checkpoint.pth").is_file():
        ckpt = torch.load(args.exp_dir / "checkpoint.pth", map_location="cpu", weights_only=False)
        start_epoch = ckpt["epoch"]
        best_acc = ckpt["best_acc"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        start_epoch = 0
        best_acc = {"top1": 0, "top5": 0}
    
    # Data loading
    traindir = args.data_dir / "train"
    valdir = args.data_dir / "val"
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(args.resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    
    # Handle subset training (1% or 10%)
    if args.train_percent in {1, 10}:
        import urllib.request
        train_files = urllib.request.urlopen(
            f"https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{args.train_percent}percent.txt"
        ).readlines()
        train_dataset.samples = []
        for fname in train_files:
            fname = fname.decode().strip()
            cls = fname.split("_")[0]
            train_dataset.samples.append(
                (traindir / cls / fname, train_dataset.class_to_idx[cls])
            )
    
    # Distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42
    )
    
    assert args.batch_size % world_size == 0
    per_device_batch_size = args.batch_size // world_size
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=per_device_batch_size,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=False
    )
    
    # Wrap with XLA parallel loader
    train_loader = pl.MpDeviceLoader(train_loader, device)
    val_loader = pl.MpDeviceLoader(val_loader, device)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        if args.weights == "finetune":
            model.train()
        elif args.weights == "freeze":
            model.eval()
        
        train_sampler.set_epoch(epoch)
        
        for step, (images, target) in enumerate(train_loader, start=epoch * len(train_loader)):
            images = images.to(device)
            target = target.to(device)
            
            output = model(images)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            xm.optimizer_step(optimizer)
            
            if step % args.print_freq == 0:
                loss_val = loss.item()
                if is_master:
                    pg = optimizer.param_groups
                    lr_head = pg[0]["lr"]
                    lr_backbone = pg[1]["lr"] if len(pg) == 2 else 0
                    stats = dict(
                        epoch=epoch,
                        step=step,
                        lr_backbone=lr_backbone,
                        lr_head=lr_head,
                        loss=loss_val,
                        time=int(time.time() - start_time),
                    )
                    print(json.dumps(stats))
                    if stats_file is not None:
                        print(json.dumps(stats), file=stats_file)
        
        # Evaluate
        model.eval()
        top1 = AverageMeter("Acc@1")
        top5 = AverageMeter("Acc@5")
        
        with torch.no_grad():
            for images, target in val_loader:
                images = images.to(device)
                target = target.to(device)
                
                output = model(images)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
        
        # Aggregate metrics across all TPU cores
        top1_avg = xm.mesh_reduce('top1_avg', top1.avg, lambda x: sum(x) / len(x))
        top5_avg = xm.mesh_reduce('top5_avg', top5.avg, lambda x: sum(x) / len(x))
        
        if is_master:
            best_acc["top1"] = max(best_acc["top1"], top1_avg)
            best_acc["top5"] = max(best_acc["top5"], top5_avg)
            
            stats = dict(
                epoch=epoch,
                acc1=top1_avg,
                acc5=top5_avg,
                best_acc1=best_acc["top1"],
                best_acc5=best_acc["top5"],
            )
            print(json.dumps(stats))
            if stats_file is not None:
                print(json.dumps(stats), file=stats_file)
        
        scheduler.step()
        
        # Save checkpoint
        if is_master:
            state = dict(
                epoch=epoch + 1,
                best_acc=best_acc,
                model={k: v.cpu() for k, v in model.state_dict().items()},
                optimizer={k: v.cpu() if isinstance(v, torch.Tensor) else v 
                          for k, v in optimizer.state_dict().items()},
                scheduler=scheduler.state_dict(),
            )
            xm.save(state, args.exp_dir / "checkpoint.pth")
        
        xm.rendezvous('epoch_end')


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


def main(args):
    """Entry point for XLA multiprocessing"""
    xmp.spawn(_mp_fn, args=(args,), nprocs=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Linear evaluation on TPU", parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
