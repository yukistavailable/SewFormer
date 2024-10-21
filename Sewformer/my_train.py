import argparse
import os

# My modules
import sys
from pprint import pprint

import numpy as np
import torch
import yaml

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pkg_path = "{}/SewFactory/packages".format(root_path)
sys.path.insert(0, pkg_path)
print(pkg_path)


import customconfig

import models
from data import GarmentDetrDataset
from experiment import ExperimentWrappper
from metrics.eval_detr_metrics import eval_detr_metrics
from trainer import TrainerDetr


def get_values_from_args():
    """command line arguments to control the run for running wandb Sweeps!"""
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        help="YAML configuration file",
        type=str,
        default="configs/my_train.yaml",
    )
    parser.add_argument("--test-only", "-t", action="store_true", default=False)
    parser.add_argument("--local_rank", default=0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    return config, args


if __name__ == "__main__":
    from pprint import pprint

    np.set_printoptions(precision=4, suppress=True)
    config, args = get_values_from_args()
    system_info = customconfig.Properties("./system.json")

    experiment = ExperimentWrappper(
        config,  # set run id in cofig to resume unfinished run!
        system_info["wandb_username"],
        no_sync=False,
    )

    # Dataset Class
    # data_class = getattr(data, config["dataset"]["class"])
    dataset = GarmentDetrDataset(
        system_info["datasets_path"],
        system_info["sim_root"],
        config["dataset"],
        gt_caching=True,
        feature_caching=False,
    )

    trainer = TrainerDetr(
        config["trainer"],
        experiment,
        dataset,
        config["data_split"],
        with_norm=True,
        with_visualization=config["trainer"]["with_visualization"],
    )  # only turn on visuals on custom garment data
    trainer.init_randomizer()

    # --- Model ---
    model, criterion = models.build_model(config)
    model_without_ddp = model

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    # Wrap model

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

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Train::Info::Number of params: {n_parameters}")

    if not args.test_only:
        trainer.fit(model, model_without_ddp, criterion, rank, config)
    else:
        config["loss"]["lepoch"] = -1
        if config["NN"]["pre-trained"] is None or not os.path.exists(
            config["NN"]["pre-trained"]
        ):
            print("Train::Error:Pre-trained model should be set for test only mode")
            raise ValueError("Pre-trained model should be set for test")

    # --- Final evaluation ----
    model.load_state_dict(experiment.get_best_model()["model_state_dict"])
    datawrapper = trainer.datawraper
    final_metrics = eval_detr_metrics(model, criterion, datawrapper, None, "validation")
    experiment.add_statistic("valid_on_best", final_metrics, log="Validation metrics")
    pprint(final_metrics)
    final_metrics = eval_detr_metrics(model, criterion, datawrapper, None, "test")
    experiment.add_statistic("test_on_best", final_metrics, log="Test metrics")
    pprint(final_metrics)
