import argparse

import kornia.color
import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.utils

import MainNet.options.options as option
from MainNet.utils import util
from MainNet.models import create_model
from adapter import dataset_loader

def main():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./MainNet/options/train/train_Enhance.yml',
                        help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)

    opts = {}
    opts["img_to_load"] = -1
    opts["num_workers"] = 12
    opts["cuda_device"] = "cuda:0"

    model = create_model(opt)
    model_parameters = filter(lambda p: p.requires_grad, model.netG.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("-----------------------")
    print("BMNet total parameters: ", params)
    print("-----------------------")

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # ws_istd = "X:/ISTD_Dataset/test/test_A/*.png"
    # ns_istd = "X:/ISTD_Dataset/test/test_C/*.png"
    # mask_istd = "X:/ISTD_Dataset/test/test_B/*.png"
    # test_loader = dataset_loader.load_istd_dataset(ws_istd, ns_istd, mask_istd, 128, opts)
    # save_dir = "./reports/ISTD/"
    #
    # for i, (file_name, rgb_ws, rgb_ns, shadow_mask) in enumerate(test_loader, 0):
    #     rgb_ws = rgb_ws.to(device)
    #     rgb_ns = rgb_ns.to(device)
    #     shadow_mask = shadow_mask.to(device)
    #
    #     train_data = {}
    #     train_data["LQ"] = rgb_ws
    #     train_data["GT"] = rgb_ns
    #     train_data["MASK"] = shadow_mask
    #
    #     model.feed_data(train_data)
    #     model.test()
    #
    #     istd_results = model.get_results()
    #
    #     for j in range(0, np.size(file_name)):
    #         impath = save_dir + file_name[j] + ".png"
    #         torchvision.utils.save_image(istd_results[j], impath, normalize=True)
    #         print("Saving " +impath)
    #
    # ws_istd = "X:/SRD_Test/srd/shadow/*.jpg"
    # ns_istd = "X:/SRD_Test/srd/shadow_free/*.jpg"
    # mask_istd = "X:/SRD_Test/srd/mask/*.jpg"
    # test_loader = dataset_loader.load_srd_dataset(ws_istd, ns_istd, mask_istd, 128, opts)
    # save_dir = "./reports/SRD/"
    #
    # for i, (file_name, rgb_ws, rgb_ns, shadow_mask) in enumerate(test_loader, 0):
    #     rgb_ws = rgb_ws.to(device)
    #     rgb_ns = rgb_ns.to(device)
    #     shadow_mask = shadow_mask.to(device)
    #
    #     train_data = {}
    #     train_data["LQ"] = rgb_ws
    #     train_data["GT"] = rgb_ns
    #     train_data["MASK"] = shadow_mask
    #
    #     model.feed_data(train_data)
    #     model.test()
    #
    #     srd_results = model.get_results()
    #     resize_op = torchvision.transforms.Resize((160, 210), torchvision.transforms.InterpolationMode.BICUBIC)
    #     srd_results = resize_op(srd_results)
    #
    #     for j in range(0, np.size(file_name)):
    #         impath = save_dir + file_name[j] + ".png"
    #         torchvision.utils.save_image(srd_results[j], impath, normalize=True)
    #         print("Saving " + impath)

    opts["img_to_load"] = -1
    # ws_places = "X:/Places Dataset/*.jpg"
    ws_places = "X:/Places Dataset/Places365_test_00230542.jpg"
    test_loader = dataset_loader.load_places_dataset(ws_places, 64, opts)
    save_dir = "./reports/Places/"

    for i, (file_name, rgb_ws) in enumerate(test_loader, 0):
        rgb_ws = rgb_ws.to(device)

        train_data = {}
        train_data["LQ"] = rgb_ws
        train_data["GT"] = rgb_ws
        sm_matte = kornia.color.rgb_to_grayscale(rgb_ws)
        sm_matte = torch.isfinite(sm_matte) & (sm_matte > 0.0)
        train_data["MASK"] = sm_matte

        model.feed_data(train_data)
        model.test()

        places_results = model.get_results()
        # resize_op = torchvision.transforms.Resize((160, 210), torchvision.transforms.InterpolationMode.BICUBIC)
        # srd_results = resize_op(srd_results)

        for j in range(0, np.size(file_name)):
            impath = save_dir + file_name[j] + ".jpg"
            torchvision.utils.save_image(places_results[j], impath, normalize=True)
            print("Saving " + impath)



if __name__ == "__main__":
    main()