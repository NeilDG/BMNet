import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from torch.nn.parallel import DataParallel, DistributedDataParallel
import MainNet.models.networks as networks
import MainNet.models.lr_scheduler as lr_scheduler

from ColorTrans.models import vgg_loss_model
from .base_model import BaseModel
from MainNet.models.loss import CharbonnierLoss,histcal
from MainNet.models.loss_new import SSIMLoss,VGGLoss,GradientLoss,LabLoss
import torch.nn.functional as F
import random
from MainNet.metrics.calculate_PSNR_SSIM import psnr_np
logger = logging.getLogger('base')


class SIEN_Model(BaseModel):
    def __init__(self, opt):
        super(SIEN_Model, self).__init__(opt)

        self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        # print network
        self.print_network()
        self.load()

####################################################################
        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
                self.mse = nn.MSELoss().to(self.device)
                self.cri_grad = GradientLoss().to(self.device)
                self.cri_vgg = vgg_loss_model.VGGPerceptualLoss().to(self.device)
                self.cri_lab = LabLoss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
                # self.cri_vgg = VGGLoss(id=4).to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))


            self.l_pix_w = train_opt['pixel_weight']
            self.l_ssim_w = train_opt['ssim_weight']
            self.l_vgg_w = train_opt['vgg_weight']

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['fix_some_part']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        LQ_IMG = data['LQ']
        GT_IMG = data['GT']
        MASK_IMG = data['MASK']
        # LQright_IMG = data['LQright']
        self.var_L = LQ_IMG.to(self.device)
        self.mask = MASK_IMG
        # self.varright_L = LQright_IMG.to(self.device)
        if need_GT:
            self.real_H = GT_IMG.to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0


    def optimize_parameters(self, step):
        if self.opt['train']['fix_some_part'] and step < self.opt['train']['fix_some_part']:
            self.set_params_lr_zero()

        self.netG.zero_grad() ################################################# new add
        self.optimizer_G.zero_grad()

        # LR_right = self.varright_L
        out, maskcolor = self.netG(self.var_L, self.mask, self.real_H.detach(),rev = False)
        if self.opt['train']['dual']:
            out_rev,_ = self.netG(self.var_L, self.mask, self.real_H.detach(),rev = True)

        gt = self.real_H

        # histgt = histcal(self.real_H.detach())
        # histout = histcal(out)

        l_total = self.cri_pix(out, gt)+0.4*self.cri_lab(out, gt.detach())
        if self.opt['train']['color']:
            l_total += 0.1*self.cri_pix(maskcolor, self.real_H/(torch.mean(self.real_H,1).unsqueeze(1)+1e-8))
        if self.opt['train']['dual']:
            l_total += 0.1*self.cri_pix(out_rev, self.var_L.detach())
                  # + 0.1*self.cri_pix(maskcolor, self.real_H/(torch.mean(self.real_H,1).unsqueeze(1)+1e-8))
        # l_total+=0.1*self.cri_vgg(self.var_L.detach(), out, gt)
        # l_total+=2e2*self.cri_pix(histout, histgt)
        # l_total+=0.1*self.cri_ssim(out,gt)


        l_total.backward()
        self.optimizer_G.step()
        self.fake_H = out
        psnr = psnr_np(self.fake_H.detach(), self.real_H.detach())
        if self.opt['train']['dual']:
            psnr_rev = psnr_np(out_rev.detach(), self.var_L.detach())
        else:
            psnr_rev = psnr
        # set log
        self.log_dict['psnr'] = psnr.item()
        self.log_dict['psnr_rev'] = psnr_rev.item() ### psnr_rev.item()
        self.log_dict['l_total'] = l_total.item()


    def test(self):
        self.netG.eval()
        with torch.no_grad():
            out, maskcolor = self.netG(self.var_L, self.mask, self.real_H.detach(),rev = False)
            self.fake_H = out
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def get_results(self):
        # return self.fake_H.detach().float().cpu()
        return self.fake_H.detach()

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        # load_path_G = "./MainNet/output/experiments/STEN/models/10_G.pth" #override
        print("Load path of model: ", load_path_G)
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])


    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def save_best(self,name):
        self.save_network(self.netG, 'best'+name, 0)
