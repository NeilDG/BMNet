import glob
import random
from pathlib import Path
import kornia
import numpy as np
import torch
import torchvision.utils
from torch.utils import data
import cv2
import torchvision.transforms as transforms
import os

from adapter import shadow_datasets


def assemble_img_list(img_dir, opts):
    img_list = glob.glob(img_dir)

    if (opts["img_to_load"] > 0):
        img_list = img_list[0: opts["img_to_load"]]

    for i in range(0, len(img_list)):
        img_list[i] = img_list[i].replace("\\", "/")

    return img_list

def load_shadow_train_dataset(ws_path, ns_path, ws_istd_path, ns_istd_path, load_size, opts):
    initial_ws_list = assemble_img_list(ws_path, opts)
    initial_ns_list = assemble_img_list(ns_path, opts)

    temp_list = list(zip(initial_ws_list, initial_ns_list))
    random.shuffle(temp_list)
    initial_ws_list, initial_ns_list = zip(*temp_list)

    initial_istd_ws_list = assemble_img_list(ws_istd_path, opts)
    initial_istd_ns_list = assemble_img_list(ns_istd_path, opts)

    temp_list = list(zip(initial_istd_ws_list, initial_istd_ns_list))
    random.shuffle(temp_list)
    initial_istd_ws_list, initial_istd_ns_list = zip(*temp_list)

    ws_list = []
    ns_list = []

    dataset_repeats = 1
    for i in range(0, dataset_repeats): #TEMP: formerly 0-1
        ws_list += initial_ws_list
        ns_list += initial_ns_list

    print("Length of images: %d %d"  % (len(ws_list), len(ns_list)))
    img_length = len(ws_list)

    data_loader = torch.utils.data.DataLoader(
        shadow_datasets.ShadowTrainDataset(img_length, ws_list, ns_list, 1),
        batch_size=load_size,
        num_workers=int(opts["num_workers"]),
        shuffle=False,
        pin_memory=True,
        pin_memory_device=opts["cuda_device"]
    )

    return data_loader

def load_istd_dataset(ws_istd_path, ns_istd_path, mask_istd_path, load_size, opts):

    initial_istd_ws_list = assemble_img_list(ws_istd_path, opts)
    initial_istd_ns_list = assemble_img_list(ns_istd_path, opts)
    initial_istd_mask_list = assemble_img_list(mask_istd_path, opts)

    temp_list = list(zip(initial_istd_ws_list, initial_istd_ns_list, initial_istd_mask_list))
    random.shuffle(temp_list)

    print("Length of images: %d %d %d" % (len(initial_istd_ws_list), len(initial_istd_ns_list), len(initial_istd_mask_list)))
    img_length = len(initial_istd_ws_list)

    data_loader = torch.utils.data.DataLoader(
        shadow_datasets.ShadowISTDDataset(img_length, initial_istd_ws_list, initial_istd_ns_list, initial_istd_mask_list, 1),
        batch_size=load_size,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        pin_memory_device=opts["cuda_device"]
    )

    return data_loader

def load_srd_dataset(ws_istd_path, ns_istd_path, mask_istd_path, load_size, opts):

    initial_istd_ws_list = assemble_img_list(ws_istd_path, opts)
    initial_istd_ns_list = assemble_img_list(ns_istd_path, opts)
    initial_istd_mask_list = assemble_img_list(mask_istd_path, opts)

    temp_list = list(zip(initial_istd_ws_list, initial_istd_ns_list, initial_istd_mask_list))
    random.shuffle(temp_list)

    print("Length of images: %d %d %d" % (len(initial_istd_ws_list), len(initial_istd_ns_list), len(initial_istd_mask_list)))
    img_length = len(initial_istd_ws_list)

    data_loader = torch.utils.data.DataLoader(
        shadow_datasets.ShadowSRDDataset(img_length, initial_istd_ws_list, initial_istd_ns_list, initial_istd_mask_list, 1),
        batch_size=load_size,
        num_workers=int(opts["num_workers"]),
        shuffle=False,
        pin_memory=True,
        pin_memory_device=opts["cuda_device"]
    )

    return data_loader

def load_places_dataset(path, load_size, opts):
    ws_list = assemble_img_list(path, opts)
    if(opts["img_to_load"] > 0):
        ws_list = ws_list[0:opts["img_to_load"]]

    ws_list = ["X:/Places Dataset/Places365_test_00001292.jpg",
               "X:/Places Dataset/Places365_test_00003500.jpg",
               "X:/Places Dataset/Places365_test_00004141.jpg",
               "X:/Places Dataset/Places365_test_00085268.jpg",
               "X:/Places Dataset/Places365_test_00225331.jpg",
               "X:/Places Dataset/Places365_test_00230542.jpg",
               "X:/Places Dataset/Places365_test_00232509.jpg",
               "X:/Places Dataset/Places365_test_00232710.jpg",
               "X:/Places Dataset/Places365_test_00232749.jpg",
               "X:/Places Dataset/Places365_test_00237288.jpg",
               "X:/Places Dataset/Places365_test_00237182.jpg",
               "X:/Places Dataset/Places365_test_00294021.jpg",
               "X:/Places Dataset/Places365_test_00085268.jpg"]
    # ws_list = ws_list[100000:328497]

    print("Length of images: %d" % (len(ws_list)))
    img_length = len(ws_list)

    data_loader = torch.utils.data.DataLoader(
        shadow_datasets.PlacesDataset(img_length, ws_list),
        batch_size=load_size,
        num_workers=int(opts["num_workers"]),
        shuffle=False,
        pin_memory=True,
        pin_memory_device=opts["cuda_device"]
    )

    return data_loader


