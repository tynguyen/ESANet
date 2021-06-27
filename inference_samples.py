# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
from glob import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src.prepare_data import prepare_data

HOME = os.environ["HOME"]


def _load_img(fp):
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


if __name__ == "__main__":
    # arguments
    parser = ArgumentParserRGBDSegmentation(
        description="Efficient RGBD Indoor Sematic Segmentation (Inference)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=HOME + "/bags/june_25th/kinect_rgbd_person_act",
        help="Root dir to the data where color images are given by {data_root}/color and depth images are given by {data_root}/depth",
    )
    parser.set_common_args()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="trained_models/nyuv2/r34_NBt1D_scenenet.pth",
        # default="trained_models/sunrgbd/r34_NBt1D.pth",
        help="Path to the checkpoint of the trained model.",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1.0,
        help="Additional depth scaling factor to apply.",
    )
    args = parser.parse_args()

    # dataset
    # TODO: set dataset to be sunrgbd
    # args.dataset = "sunrgbd"
    args.pretrained_on_imagenet = False  # we are loading other weights anyway
    dataset, preprocessor = prepare_data(args, with_input_orig=True)
    n_classes = dataset.n_classes_without_void

    # model and checkpoint loading
    model, device = build_model(args, n_classes=n_classes)
    checkpoint = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print("Loaded checkpoint from {}".format(args.ckpt_path))

    model.eval()
    model.to(device)

    # get samples
    rgb_filepaths = sorted(glob(os.path.join(args.data_root, "color/*.jpg")))
    depth_filepaths = sorted(glob(os.path.join(args.data_root, "depth/*.png")))
    assert args.modality == "rgbd", "Only RGBD inference supported so far"
    assert len(rgb_filepaths) == len(depth_filepaths)
    filepaths = zip(rgb_filepaths, depth_filepaths)

    # inference
    for fp_rgb, fp_depth in filepaths:
        # load sample
        img_rgb = _load_img(fp_rgb)
        img_depth = _load_img(fp_depth).astype("float32") * args.depth_scale
        h, w, _ = img_rgb.shape

        # preprocess sample
        sample = preprocessor({"image": img_rgb, "depth": img_depth})

        # add batch axis and copy to device
        image = sample["image"][None].to(device)
        depth = sample["depth"][None].to(device)

        # apply network
        pred = model(image, depth)
        pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
        pred = torch.argmax(pred, dim=1)
        pred = pred.cpu().numpy().squeeze().astype(np.uint8)

        # show result
        pred_colored = dataset.color_label(pred, with_void=False)
        fig, axs = plt.subplots(1, 3, figsize=(16, 3))
        [ax.set_axis_off() for ax in axs.ravel()]
        axs[0].imshow(img_rgb)
        axs[1].imshow(img_depth, cmap="gray")
        axs[2].imshow(pred_colored)

        plt.suptitle(
            f"Image: ({os.path.basename(fp_rgb)}, "
            f"{os.path.basename(fp_depth)}), Model: {args.ckpt_path}"
        )
        # plt.savefig('./result.jpg', dpi=150)
        plt.show()
