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

from torch import nn
from torchvision import transforms
import torch
from tqdm import tqdm

# Added for Hugging Face streaming
from datasets import load_dataset

import resnet


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Extract features from a pretrained model on ImageNet using streaming data"
    )

    # Checkpoint and Model
    parser.add_argument("--pretrained", type=Path, required=True, help="path to pretrained model")
    parser.add_argument("--arch", type=str, default="resnet50", help="model architecture")

    # MODIFIED: Added a directory to save the feature representations
    parser.add_argument(
        "--save-dir",
        default="./representations/",
        type=Path,
        metavar="DIR",
        help="path to directory for saving features and labels",
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
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--print-freq", default=100, type=int, metavar="N", help="print frequency"
    )

    return parser


def main():
    parser = get_arguments()
    args = parser.parse_args()
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
        # Create the directory to save features
        args.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Features will be saved to: {args.save_dir}")
        print(" ".join(sys.argv))

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    # --- MODEL LOADING ---
    # We only need the backbone for feature extraction
    backbone, embedding = resnet.__dict__[args.arch](zero_init_residual=True)
    state_dict = torch.load(args.pretrained, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
        state_dict = {
            key.replace("module.backbone.", ""): value
            for (key, value) in state_dict.items()
        }
    backbone.load_state_dict(state_dict, strict=False)
    backbone.cuda(gpu)
    
    # Set the model to evaluation mode and wrap with DDP
    backbone.eval()
    model = torch.nn.parallel.DistributedDataParallel(backbone, device_ids=[gpu])


    # --- DATA LOADING ---
    # Use the same transform for both train and validation for consistent feature extraction
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    
def apply_transforms(batch):
    """Applies transforms to a batch of examples."""
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    batch["image"] = [transform(img.convert("RGB")) for img in batch["jpg"]]
    batch["label"] = batch["cls"]
    return batch


    if args.rank == 0:
        print("Streaming ImageNet data from 'timm/imagenet-1k-wds' on Hugging Face Hub...")

    # Load and shard the training dataset
    train_dataset = (
        load_dataset("timm/imagenet-1k-wds", split="train", streaming=True)
        .shard(num_shards=args.world_size, index=args.rank)
        .map(apply_transforms, remove_columns=["jpg", "cls"])
    )

    # Load and shard the validation dataset
    val_dataset = (
        load_dataset("timm/imagenet-1k-wds", split="validation", streaming=True)
        .shard(num_shards=args.world_size, index=args.rank)
        .map(apply_transforms, remove_columns=["jpg", "cls"])
    )
    
    kwargs = dict(
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

    # --- FEATURE EXTRACTION ---
    print(f"Rank {args.rank}: Starting feature extraction for the training set...")
    train_features, train_labels = extract_features(model, train_loader, gpu, args)
    
    # Save features and labels for the current rank's shard
    torch.save(train_features, args.save_dir / f"train_features_rank_{args.rank}.pt")
    torch.save(train_labels, args.save_dir / f"train_labels_rank_{args.rank}.pt")
    print(f"Rank {args.rank}: Saved training features and labels.")

    # Synchronize all processes before starting validation set extraction
    torch.distributed.barrier()

    print(f"Rank {args.rank}: Starting feature extraction for the validation set...")
    val_features, val_labels = extract_features(model, val_loader, gpu, args)
    
    # Save features and labels for the current rank's shard
    torch.save(val_features, args.save_dir / f"val_features_rank_{args.rank}.pt")
    torch.save(val_labels, args.save_dir / f"val_labels_rank_{args.rank}.pt")
    print(f"Rank {args.rank}: Saved validation features and labels.")

    if args.rank == 0:
        print("\n--- EXTRACTION COMPLETE ---")
        print(f"All feature and label files have been saved in: '{args.save_dir}'")
        print("Each file is named with its rank (e.g., train_features_rank_0.pt).")
        print("You will need to load and concatenate these files to get the full dataset.")


def extract_features(model, loader, gpu, args):
    """
    Iterates over a dataloader, extracts features using the model, and returns
    the features and labels as concatenated tensors.
    """
    features_list = []
    labels_list = []
    
    # Use tqdm for a progress bar if it's rank 0
    iterable_loader = tqdm(loader, desc="Extracting", disable=(args.rank != 0))

    with torch.no_grad():
        for batch in iterable_loader:
            images = batch["image"].cuda(gpu, non_blocking=True)
            labels = batch["label"]
            
            # Get features from the model
            features = model(images)

            # Append features and labels to lists (move to CPU to save GPU memory)
            features_list.append(features.detach().cpu())
            labels_list.append(labels.cpu())

    # Concatenate all features and labels from the lists
    all_features = torch.cat(features_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    
    return all_features, all_labels


# --- Helper functions (unchanged from original) ---

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()

def handle_sigterm(signum, frame):
    pass

if __name__ == "__main__":
    main()
