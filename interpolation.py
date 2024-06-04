import os
import sys

import PIL
import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from PIL import Image

current_path = os.getcwd()
base_path = os.path.dirname(current_path)

sys.path.append(base_path)

from experiment import ExperimentWrappper
from models.backbone import build_backbone
from models.garment_detr_2d import GarmentDETRv6
from models.transformer import build_transformer


def is_img_file(fn):
    img_endfix = ["png", "PNG", "jpg", "jpeg", "jpg", "JPEG", "JPG"]
    return fn.split(".")[-1] in img_endfix


def load_source_appearance(img_fn):
    ref_img = PIL.Image.open(img_fn).convert("RGB")
    h, w = ref_img.size
    min_size, max_size = min(h, w), max(h, w)
    pad_ref_img = T.Pad(
        padding=(int((max_size - h) / 2), int((max_size - w) / 2)), fill=255
    )(ref_img)
    img_tensor = T.Compose([T.Resize((384, 384)), T.ToTensor()])(pad_ref_img)
    return img_tensor.unsqueeze(0), img_fn


img_transform = T.Compose([T.Resize((384, 384)), T.ToTensor()])

config_file_path = "../configs/test.yaml"
test_type = "deepfashion"
save_root = "../outputs/deepfashion/"

with open(config_file_path, "r") as f:
    config = yaml.safe_load(f)

config["experiment"][
    "local_dir"
] = "../garment_outputs/Detr2d-V6-final-dif-ce-focal-schd-agp"
config["dataset"][
    "panel_classification"
] = "../assets/data_configs/panel_classes_condenced.json"
config["dataset"]["filter_by_params"] = "../assets/data_configs/param_filter.json"
config["NN"][
    "pre-trained"
] = "../assets/ckpts/Detr2d-V6-final-dif-ce-focal-schd-agp_checkpoint_37.pth"

shape_experiment = ExperimentWrappper(config, wandb_username="")
shape_dataset, shape_datawrapper = shape_experiment.load_detr_dataset(
    [],  # assuming dataset root structure
    {
        "feature_caching": False,
        "gt_caching": False,
    },  # NOTE: one can change some data configuration for evaluation purposes here!
    unseen=True,
    batch_size=1,
)

shape_model, criterion, device = shape_experiment.load_detr_model(
    shape_dataset.config, others=False
)

num_classes = config["dataset"]["max_pattern_len"]  # 23
backbone = backbone = build_backbone(config)
panel_transformer = build_transformer(config)
model = GarmentDETRv6(
    backbone, panel_transformer, num_classes, 14, 22, edge_kwargs=config["NN"]
)

model = nn.DataParallel(model, device_ids=["cuda:0"])
model.load_state_dict(torch.load(config["NN"]["pre-trained"])["model_state_dict"])

input_image_path_1 = "../assets/data/deepfashion/WOMEN-Pants-id_00001944-01_1_front.jpg"
image_1 = Image.open(input_image_path_1)

input_image_path_2 = "../assets/data/deepfashion/WOMEN-Pants-id_00007458-02_1_front.jpg"
image_2 = Image.open(input_image_path_2)

img_tensor_1 = img_transform(image_1).unsqueeze(0)
img_tensor_2 = img_transform(image_2).unsqueeze(0)

output_1 = model(img_tensor_1, output_panel_memory=True)
print(output_1["panel_memory"].shape)

output_2 = model(img_tensor_2, output_panel_memory=True)
print(output_2["panel_memory"].shape)

panel_memory_1 = output_1["panel_memory"]
bs, c, h, w = panel_memory_1.shape
panel_memory_1 = panel_memory_1.view(bs, c, -1).permute(2, 0, 1)

panel_memory_2 = output_2["panel_memory"]
bs, c, h, w = panel_memory_2.shape
panel_memory_2 = panel_memory_2.view(bs, c, -1).permute(2, 0, 1)

panel_memory_interpolation = (panel_memory_1 + panel_memory_2) / 2

output = model(img_tensor_1, panel_memory=panel_memory_interpolation)

interpolation_num = 5
for i in range(interpolation_num + 1):
    panel_memory_interpolation = panel_memory_1 * ((5 - i) / 5) + panel_memory_2 * (
        i / 5
    )
    output = model(img_tensor_1, panel_memory=panel_memory_interpolation)
    dataname = f"tmp{i}"
    save_to = "../outputs/interpolation/"
    _, _, prediction_img = shape_dataset.save_prediction_single(
        output, dataname, save_to, return_stitches=False
    )
