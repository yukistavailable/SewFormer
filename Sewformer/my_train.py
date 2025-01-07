import argparse
import os

# My modules
import sys
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(""))))
pkg_path = "{}/SewFactory/packages".format(root_path)
sys.path.insert(0, pkg_path)
print(pkg_path)


import customconfig

import models
from data import MyGarmentDetrDataset
from warm_cosine_scheduler import GradualWarmupScheduler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/my_train.yaml")
    parser.add_argument("-d", "--dir_paths", type=str, nargs="*")
    parser.add_argument("-u", "--used_ratio", type=float, default=1.0)
    args = parser.parse_args()

    np.set_printoptions(precision=4, suppress=True)
    with open("configs/my_train.yaml", "r") as f:
        config = yaml.safe_load(f)

    system_info = customconfig.Properties("./system.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_datasets = []
    val_datasets = []
    train_loaders = []
    val_loaders = []
    # Dataset Class
    for dir_path in args.dir_paths:
        all_garment_dir_paths = [
            os.path.join(dir_path, x) for x in os.listdir(dir_path)
        ]
        # sort all_garment_dir_paths
        all_garment_dir_paths = sorted(all_garment_dir_paths)
        train_garment_dir_paths = all_garment_dir_paths[
            : int(0.8 * len(all_garment_dir_paths) * args.used_ratio)
        ]
        val_garment_dir_paths = all_garment_dir_paths[
            int(0.8 * len(all_garment_dir_paths) * args.used_ratio) : int(len(all_garment_dir_paths) * args.used_ratio)
        ]
        train_dataset = MyGarmentDetrDataset(train_garment_dir_paths, config=config)
        val_dataset = MyGarmentDetrDataset(val_garment_dir_paths, config=config)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["trainer"]["batch_size"],
            shuffle=True,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["trainer"]["batch_size"],
            shuffle=False,
            pin_memory=True,
        )

        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    model, criterion = models.build_model(config)
    if torch.cuda.is_available():
        model.to(device)
        criterion.to(device)
    if config["NN"]["step-trained"] is not None and os.path.exists(
        config["NN"]["step-trained"]
    ):
        model.load_state_dict(
            torch.load(config["NN"]["step-trained"], map_location="cuda")[
                "model_state_dict"
            ]
        )
        print(
            "Train::Info::Load Pre-step-trained model: {}".format(
                config["NN"]["step-trained"]
            )
        )
    # model = nn.DataParallel(model, device_ids=[0])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Train::Info::Number of params: {n_parameters}")

    # Train
    ## Set Optimizer
    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": float(config["trainer"]["lr_backbone"]),
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts,
        lr=float(config["trainer"]["lr"] / 8),
        weight_decay=float(config["trainer"]["weight_decay"]),
    )

    ## Set Scheduler
    steps_per_epoch = 1000  # len(train_dataset) // config["trainer"]["batch_size"]
    if (
        "lr_scheduling" in config["trainer"]
        and config["trainer"]["lr_scheduling"] == "OneCycleLR"
    ):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["trainer"]["lr"],
            epochs=config["trainer"]["epochs"],
            steps_per_epoch=steps_per_epoch,
            cycle_momentum=False,  # to work with Adam
        )
    elif (
        "lr_scheduling" in config["trainer"]
        and config["trainer"]["lr_scheduling"] == "warm_cosine"
    ):
        consine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["trainer"]["epochs"] * steps_per_epoch,
            eta_min=0,
            last_epoch=-1,
        )
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=8,
            total_epoch=5 * steps_per_epoch,
            after_scheduler=consine_scheduler,
        )

    else:
        scheduler = None
        print("TrainerDetr::Warning::no learning scheduling set")

    def preprocess_batch(batch):
        images, gt = batch["image"], batch["ground_truth"]
        images = images.to(device)
        gt_stitches = gt["masked_stitches"].to(device)
        gt_edge_mask = gt["stitch_edge_mask"].to(device)
        reindex_stitches = gt["reindex_stitches"].to(device)
        if len(gt_stitches.shape) == 3:
            # (B, 1, N) -> (B, N)
            gt_stitches = gt_stitches.squeeze(1)
        if len(gt_edge_mask.shape) == 3:
            # (B, 1, N) -> (B, N)
            gt_edge_mask = gt_edge_mask.squeeze(1)
        if len(reindex_stitches.shape) == 4:
            # (B, 1, N, N) -> (B, N, N)
            reindex_stitches = reindex_stitches.squeeze(1)

        gt["masked_stitches"] = gt_stitches
        gt["stitch_edge_mask"] = gt_edge_mask
        gt["reindex_stitches"] = reindex_stitches

        flag = False
        label_indices = gt["label_indices"]
        for i in range(len(label_indices)):
            if not torch.all(label_indices[i] == label_indices[0]):
                print("Train::Warning::label_indices are not the same")
                flag = True
        if not flag:
            gt["label_indices"] = label_indices[0]

        return images, gt

    # --- Training ---
    best_val_loss: Optional[int] = None
    for epoch in tqdm(range(config["trainer"]["epochs"])):
        model.train()
        criterion.train()

        train_loader_iters = [iter(loader) for loader in train_loaders]
        iter_idx = 0
        stitch_acc = 0
        stitch_loss = 0
        total_loss = 0
        while True:
            train_loader_iter = train_loader_iters[iter_idx % len(train_loader_iters)]
            iter_idx += 1
            try:
                batch = next(train_loader_iter)
            except StopIteration:
                break
        
            images, gt = preprocess_batch(batch)

            outputs = model(
                images,
                gt_stitches=gt["masked_stitches"],
                gt_edge_mask=gt["stitch_edge_mask"],
                return_stitches=config["trainer"]["return_stitches"],
            )
            loss, loss_dict = criterion(outputs, gt, epoch=epoch)
            stitch_acc += loss_dict["stitch_acc"]
            stitch_loss += loss_dict["stitch_ce_loss"]
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"Train::Info::Epoch: {epoch}, loss: {total_loss / iter_idx}")
        print(f"Train::Info::Epoch: {epoch}, stitch_acc: {stitch_acc / iter_idx}")
        print(f"Train::Info::Epoch: {epoch}, stitch_loss: {stitch_loss / iter_idx}")


        model.eval()
        criterion.eval()
        with torch.no_grad():
            valid_losses, valid_loss_dict = [], {}
            for j, val_loader in enumerate(val_loaders):
                for i, batch in enumerate(val_loader):
                    images, gt = preprocess_batch(batch)
                    outputs = model(
                        images,
                        gt_stitches=gt["masked_stitches"],
                        gt_edge_mask=gt["stitch_edge_mask"],
                        return_stitches=config["trainer"]["return_stitches"],
                    )
                    if 0 <= i < 5:
                        # save image
                        tmp_outputs = {}
                        for key in outputs:
                            if isinstance(outputs[key], torch.Tensor):
                                tmp_outputs[key] = outputs[key].cpu()
                        MyGarmentDetrDataset.save_prediction_single(
                            tmp_outputs,
                            f"logs/{epoch}_{j}_{i}.svg",
                            f"logs/{epoch}_{j}_{i}.png",
                            f"logs/{epoch}_{j}_{i}_spec.json",
                            return_stitches=True,
                            config=config,
                        )
                    loss
                    loss, loss_dict = criterion(outputs, gt, epoch=epoch)
                    valid_losses.append(loss.item())
                    if len(valid_loss_dict) == 0:
                        valid_loss_dict = {"valid_" + key: [] for key in loss_dict}
                    for key, val in loss_dict.items():
                        if val is not None:
                            valid_loss_dict["valid_" + key].append(val.cpu())

            valid_losses = np.mean(valid_losses)
            for key in valid_loss_dict:
                valid_loss_dict[key] = np.mean(valid_loss_dict[key])

            if best_val_loss is None or valid_losses < best_val_loss:
                best_val_loss = valid_losses
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    "checkpoints/best_model.pth",
                )
                print(
                    f"Train::Info::Save model at epoch: {epoch}, loss: {best_val_loss}"
                )
