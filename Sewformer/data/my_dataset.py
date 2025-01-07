import os

# Do avoid a need for changing Evironmental Variables outside of this script
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
root_path = os.path.dirname(os.path.dirname(os.path.abspath(parentdir)))
pkg_path = "{}/SewFactory/packages".format(root_path)
print(pkg_path)
sys.path.insert(0, pkg_path)

from data.panel_classes import PanelClasses
from data.pattern_converter import InvalidPatternDefError, NNSewingPattern
from data.transforms import SampleToTensor, my_tv_make_img_transforms
import data.transforms as transforms


class MyGarmentDetrDataset(Dataset):
    def __init__(
        self,
        dir_paths: List[str],
        max_panel_len: int = 14,
        max_pattern_len: int = 23,
        max_num_stitches: int = 28,
        max_stitch_edges: int = 56,
        config=None,
    ):
        # It is assumed that the root_dir is like this:
        # |-- root_dir
        #     |-- dress_0XAVEH5G53_0_241020-10-07-05
        #        |-- dress_0XAVEH5G53_0_render_front.png
        #        |-- dress_0XAVEH5G53_0_specification.json
        #     |-- dress_0XAVEH5G53_1_241020-15-50-35
        #        |-- dress_0XAVEH5G53_1_render_front.png
        #        |-- dress_0XAVEH5G53_1_specification.json
        # ...

        self.panel_classifier = PanelClasses(config["dataset"]["panel_classification"])
        self.idx_to_img_path = {}
        self.idx_to_spec_json_path = {}
        self.idx_to_img = {}
        self.idx_to_gt = {}
        self.max_panel_len = max_panel_len
        self.max_pattern_len = max_pattern_len
        print("Update max_pattern_len", len(self.panel_classifier))
        self.max_pattern_len = len(self.panel_classifier)
        self.max_num_stitches = max_num_stitches
        self.max_stitch_edges = max_stitch_edges
        self.img_transform = my_tv_make_img_transforms()
        self.transforms = [SampleToTensor()]

        stats = config["dataset"]['standardize']
        self.transforms.append(transforms.GTtandartization(stats['gt_shift'], stats['gt_scale']))

        count = 0
        for dir_path in dir_paths:
            if os.path.isdir(dir_path):
                file_paths = [
                    os.path.join(dir_path, file_name)
                    for file_name in os.listdir(dir_path)
                ]
                flag = False
                for file_path in file_paths:
                    if file_path.endswith("front.png"):
                        self.idx_to_img_path[count] = file_path
                        flag = True
                        break
                if not flag:
                    continue
                flag = False
                for file_path in file_paths:
                    if file_path.endswith("specification.json"):
                        self.idx_to_spec_json_path[count] = file_path
                        flag = True
                        break
                if not flag:
                    continue
                count += 1

    @staticmethod
    def free_edges_mask(pattern, stitches, num_stitches):
        """
        Construct the mask to identify edges that are not connected to any other
        """
        mask = np.ones((pattern.shape[0], pattern.shape[1]), dtype=bool)
        max_edge = pattern.shape[1]

        for side in stitches[:, :num_stitches]:  # ignore the padded part
            for edge_id in side:
                mask[edge_id // max_edge][edge_id % max_edge] = False

        return mask

    @staticmethod
    def match_edges(free_edge_mask, stitches=None, max_num_stitch_edges=56):
        stitch_edges = np.ones((1, max_num_stitch_edges)) * (-1)
        valid_edges = (~free_edge_mask.reshape(-1)).nonzero()
        stitch_edge_mask = np.zeros((1, max_num_stitch_edges))
        if stitches is not None:
            stitches = np.transpose(stitches)
            reindex_stitches = np.zeros((1, max_num_stitch_edges, max_num_stitch_edges))
        else:
            reindex_stitches = None

        batch_edges = valid_edges[0]
        num_edges = batch_edges.shape[0]
        stitch_edges[:, :num_edges] = batch_edges
        stitch_edge_mask[:, :num_edges] = 1
        if stitches is not None:
            for stitch in stitches:
                side_i, side_j = stitch
                if side_i != -1 and side_j != -1:
                    reindex_i, reindex_j = (
                        np.where(stitch_edges[0] == side_i)[0],
                        np.where(stitch_edges[0] == side_j)[0],
                    )
                    reindex_stitches[0, reindex_i, reindex_j] = 1
                    reindex_stitches[0, reindex_j, reindex_i] = 1

        return stitch_edges * stitch_edge_mask, stitch_edge_mask, reindex_stitches

    @staticmethod
    def split_pos_neg_pairs(stitches, num_max_edges=3000):
        stitch_ind = np.triu_indices_from(stitches[0], 1)
        pos_ind = [
            [stitch_ind[0][i], stitch_ind[1][i]]
            for i in range(stitch_ind[0].shape[0])
            if stitches[0, stitch_ind[0][i], stitch_ind[1][i]]
        ]
        neg_ind = [
            [stitch_ind[0][i], stitch_ind[1][i]]
            for i in range(stitch_ind[0].shape[0])
            if not stitches[0, stitch_ind[0][i], stitch_ind[1][i]]
        ]

        assert len(neg_ind) >= num_max_edges
        neg_ind = neg_ind[:num_max_edges]
        pos_inds = np.expand_dims(np.array(pos_ind), axis=1)
        neg_inds = np.repeat(
            np.expand_dims(np.array(neg_ind), axis=0), repeats=pos_inds.shape[0], axis=0
        )
        indices = np.concatenate((pos_inds, neg_inds), axis=1)
        return indices

    def _empty_panels_mask(self, num_edges):
        """Empty panels as boolean mask"""

        mask = np.zeros(len(num_edges), dtype=bool)
        mask[num_edges == 0] = True

        return mask

    def _read_pattern(
        self,
        spec_file_path,
        pad_panels_to_len=None,
        pad_panel_num=None,
        pad_stitches_num=None,
        with_placement=False,
        with_stitches=False,
        with_stitch_tags=False,
    ):
        """Read given pattern in tensor representation from file"""

        # spec_list: {'front': 'dress_sleeveless_02OIQPCOWU_specification.json', 'back': 'dress_sleeveless_02OIQPCOWU_specification.json'}
        if "dress_sleeveless" in spec_file_path:
            template_name = "one_panel_dress_sleeveless"
        elif "dress" in spec_file_path:
            template_name = "one_panel_dress"
        elif "jumpsuit_sleeveless" in spec_file_path:
            template_name = "jumpsuit_sleeveless"
        elif "jumpsuit" in spec_file_path:
            template_name = "jumpsuit"
        else:
            raise ValueError("Unknown template name", spec_file_path)

        pattern = NNSewingPattern(
            spec_file_path,
            panel_classifier=self.panel_classifier,
            template_name=template_name,
        )
        patterns = [pattern]

        pat_tensor = NNSewingPattern.multi_pattern_as_tensors(
            patterns,
            pad_panels_to_len,
            pad_panels_num=pad_panel_num,
            pad_stitches_num=pad_stitches_num,
            with_placement=with_placement,
            with_stitches=with_stitches,
            with_stitch_tags=with_stitch_tags,
        )
        return pat_tensor

    def _load_spec_dict(self, spec_file_path):
        return {}

    def _get_sample_info(self, spec_file_path: str):
        """
        Get features and Ground truth prediction for requested data example
        """

        # try:
        #     image = Image.open(datapoint_name).convert("RGB")
        # except Exception as e:
        #     print(e)
        # image = Image.open(self._swap_name(datapoint_name)).convert("RGB")
        # if self.feature_caching:
        #     self.feature_cached[datapoint_name] = image

        # GT -- pattern
        ground_truth = self._get_pattern_ground_truth(spec_file_path)
        # if self.gt_caching:
        #     self.gt_cached[gt_folder] = ground_truth
        return ground_truth

    def _get_pattern_ground_truth(self, spec_file_path):
        """Get the pattern representation with 3D placement"""
        patterns = self._read_pattern(
            spec_file_path,
            pad_panels_to_len=self.max_panel_len,
            pad_panel_num=self.max_pattern_len,
            pad_stitches_num=self.max_num_stitches,
            with_placement=True,
            with_stitches=True,
            with_stitch_tags=True,
        )
        (
            pattern,
            num_edges,
            num_panels,
            rots,
            tranls,
            stitches,
            num_stitches,
            stitch_adj,
            stitch_tags,
            aug_outlines,
        ) = patterns
        free_edges_mask = self.free_edges_mask(pattern, stitches, num_stitches)
        empty_panels_mask = self._empty_panels_mask(num_edges)  # useful for evaluation

        ground_truth = {
            "outlines": pattern,
            "num_edges": num_edges,
            "rotations": rots,
            "translations": tranls,
            "num_panels": num_panels,
            "empty_panels_mask": empty_panels_mask,
            "num_stitches": num_stitches,
            "stitches": stitches,
            "stitch_adj": stitch_adj,
            "free_edges_mask": free_edges_mask,
            "stitch_tags": stitch_tags,
        }

        if aug_outlines[0] is not None:
            ground_truth.update({"aug_outlines": aug_outlines})

        # stitches
        if ground_truth["stitch_adj"] is not None:
            masked_stitches, stitch_edge_mask, reindex_stitches = self.match_edges(
                ground_truth["free_edges_mask"],
                stitches=ground_truth["stitches"],
                max_num_stitch_edges=self.max_stitch_edges,
            )
            label_indices = self.split_pos_neg_pairs(
                reindex_stitches, num_max_edges=1000
            )

            ground_truth.update(
                {
                    "masked_stitches": masked_stitches,
                    "stitch_edge_mask": stitch_edge_mask,
                    "reindex_stitches": reindex_stitches,
                    "label_indices": label_indices,
                }
            )

        return ground_truth

    @staticmethod
    def prediction_to_stitches(
        free_mask_logits, similarity_matrix, return_stitches=False
    ):
        free_mask = (torch.sigmoid(free_mask_logits.squeeze(-1)) > 0.5).flatten()
        if not return_stitches:
            simi_matrix = similarity_matrix + similarity_matrix.transpose(0, 1)
            simi_matrix = torch.masked_fill(
                simi_matrix, (~free_mask).unsqueeze(0), -float("inf")
            )
            simi_matrix = torch.masked_fill(simi_matrix, (~free_mask).unsqueeze(-1), 0)
            num_stitches = free_mask.nonzero().shape[0] // 2
        else:
            simi_matrix = similarity_matrix
            num_stitches = simi_matrix.shape[0] // 2
        simi_matrix = torch.triu(simi_matrix, diagonal=1)
        stitches = []

        for i in range(num_stitches):
            index = (simi_matrix == torch.max(simi_matrix)).nonzero()
            stitches.append((index[0, 0].cpu().item(), index[0, 1].cpu().item()))
            simi_matrix[index[0, 0], :] = -float("inf")
            simi_matrix[index[0, 1], :] = -float("inf")
            simi_matrix[:, index[0, 0]] = -float("inf")
            simi_matrix[:, index[0, 1]] = -float("inf")

        if len(stitches) == 0:
            stitches = None
        else:
            stitches = np.array(stitches)
            if stitches.shape[0] != 2:
                stitches = np.transpose(stitches, (1, 0))
        return stitches

    @staticmethod
    def _pred_to_pattern(prediction, return_stitches=False, config=None):
        """Convert given predicted value to pattern object"""

        panel_classifier = PanelClasses(config["dataset"]["panel_classification"])
        gt_shifts = config["dataset"]['standardize']['gt_shift']
        gt_scales = config["dataset"]['standardize']['gt_scale']

        for key in gt_shifts:
            if key == 'stitch_tags':  
                # ignore stitch tags update if explicit tags were not used
                continue
            
            pred_numpy = prediction[key].detach().cpu().numpy()
            if key == 'outlines' and len(pred_numpy.shape) == 2: 
                pred_numpy = pred_numpy.reshape(config["dataset"]["max_pattern_len"], config["dataset"]["max_panel_len"], 4)


            # Currently, we do not do any standardization
            prediction[key] = pred_numpy * gt_scales[key] + gt_shifts[key]

        # recover stitches
        if "stitches" in prediction:  # if somehow prediction already has an answer
            stitches = prediction["stitches"]
        elif "stitch_tags" in prediction:  # stitch tags to stitch list
            pass
        elif "edge_cls" in prediction and "edge_similarity" in prediction:
            stitches = MyGarmentDetrDataset.prediction_to_stitches(
                prediction["edge_cls"],
                prediction["edge_similarity"],
                return_stitches=return_stitches,
            )
        else:
            stitches = None

        # Construct the pattern from the data
        pattern = NNSewingPattern(view_ids=False, panel_classifier=panel_classifier)

        try:
            pattern.pattern_from_tensors(
                prediction["outlines"],
                panel_rotations=prediction["rotations"],
                panel_translations=prediction["translations"],
                stitches=stitches,
                padded=True,
            )
        except (RuntimeError, InvalidPatternDefError):
            pass

        return pattern

    @staticmethod
    def save_prediction_single(
        prediction,
        svg_file_path,
        png_file_path,
        spec_file_path,
        return_stitches=False,
        config=None,
    ):
        for key in prediction.keys():
            prediction[key] = prediction[key][0]

        pattern = MyGarmentDetrDataset._pred_to_pattern(
            prediction, return_stitches=return_stitches, config=config
        )
        # try:
        if True:
            pattern.my_serialize(
                svg_file=svg_file_path, png_file=png_file_path, spec_file=spec_file_path
            )
        # except (RuntimeError, InvalidPatternDefError, TypeError):
        #     print("GarmentDetrDataset::Error:: serializing skipped")



    def __getitem__(self, idx):
        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()
        if idx in self.idx_to_img:
            img = self.idx_to_img[idx]
        else:
            img = Image.open(self.idx_to_img_path[idx]).convert("RGB")
            img = self.img_transform(img)
            self.idx_to_img[idx] = img

        if idx in self.idx_to_gt:
            ground_truth = self.idx_to_gt[idx]
        else:
            ground_truth = self._get_sample_info(self.idx_to_spec_json_path[idx])
            self.idx_to_gt[idx] = ground_truth
        sample = {"image": img, "ground_truth": ground_truth}
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __len__(self):
        return len(self.idx_to_img_path)
