import os
import math
import argparse
import random
import logging
import sys

import numpy as np

sys.path.append('/home/jieh/Projects/Shadow/ColorTrans')
import torch
import torch.multiprocessing as mp
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from adapter import dataset_loader

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='/ColorTrans/options/train/train_Enhance.yml',
                        help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings

    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')


    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    ### data loader
    rgb_dir_ws = "X:/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.*"
    rgb_dir_ns = "X:/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.*"
    rgb_dir_ws = rgb_dir_ws.format(dataset_version="v69_places")
    rgb_dir_ns = rgb_dir_ns.format(dataset_version="v69_places")

    ws_istd = "X:/ISTD_Dataset/test/test_A/*.png"
    ns_istd = "X:/ISTD_Dataset/test/test_C/*.png"
    mask_istd = "X:/ISTD_Dataset/test/test_B/*.png"

    opts = {}
    opts["img_to_load"] = -1
    opts["num_workers"] = 12
    opts["cuda_device"] = "cuda:0"
    train_loader = dataset_loader.load_shadow_train_dataset(rgb_dir_ws, rgb_dir_ns, ws_istd, ns_istd, 96, opts=opts)

    #### create model
    total_epochs = 10
    model = create_model(opt)
    model_parameters = filter(lambda p: p.requires_grad, model.netG.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("-----------------------")
    print("ColorTrans total parameters: ", params)
    print("-----------------------")

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0


    best_psnr_avg = 0
    best_step_avg = 0
    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))

    for epoch in range(start_epoch, total_epochs + 2):
        total_psnr = 0
        total_psnr_rev = 0
        total_loss = 0
        print_iter = 0

        for i, (_, rgb_ws, rgb_ns, shadow_map, shadow_matte) in enumerate(train_loader, 0):
            current_step += 1
            rgb_ws = rgb_ws.to(device)
            rgb_ns = rgb_ns.to(device)
            shadow_map = shadow_map.to(device)
            shadow_matte = shadow_matte.to(device)

            train_data = {}
            train_data["LQ"] = rgb_ws
            train_data["GT"] = rgb_ns
            train_data["MASK"] = shadow_matte

            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            model.feed_data(train_data)
            ############# compute importance
            # if not opt['train']['ewc']:

            model.optimize_parameters(current_step)

            ### log
            if current_step % opt['logger']['print_freq'] == 0:
                print_iter += 1  ############################################################## new
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '

                total_loss += logs['l_total']
                total_psnr += logs['psnr']
                total_psnr_rev += logs['psnr_rev']
                mean_total = total_loss / print_iter
                mean_psnr = total_psnr / print_iter
                mean_psnr_rev = total_psnr_rev / print_iter
                message += '{:s}: {:.4e} '.format('mean_total_loss', mean_total)
                message += '{:s}: {:} '.format('mesn_psnr', mean_psnr)
                message += '{:s}: {:} '.format('mesn_psnr_rev', mean_psnr_rev)

                if rank <= 0:
                    logger.info(message)


        #### save models and training states
        if epoch % opt['logger']['save_checkpoint_epoch'] == 0:
            if rank <= 0:
                logger.info('Saving models and training states.')
                model.save(epoch)
                model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
        # tb_logger.close()

if __name__ == '__main__':
    main()