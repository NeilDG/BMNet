# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:02:01 2020

@author: delgallegon
"""
from matplotlib.lines import Line2D

import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import visdom


class VisdomReporter:
    _sharedInstance = None

    @staticmethod
    def initialize():
        VisdomReporter._sharedInstance = VisdomReporter()

    @staticmethod
    def getInstance():
        return VisdomReporter._sharedInstance

    def __init__(self):
        self.vis= visdom.Visdom()
        
        self.image_windows = {}
        self.loss_windows = {}
        self.text_windows = {}
    
    def plot_image(self, img_tensor, caption, normalize = True):
        img_group = vutils.make_grid(img_tensor[:16], nrow = 8, padding=2, normalize=normalize).cpu()
        if hash(caption) not in self.image_windows:
            self.image_windows[hash(caption)] = self.vis.images(img_group, opts = dict(caption = caption))
        else:
            self.vis.images(img_group, win = self.image_windows[hash(caption)], opts = dict(caption = caption))

    def plot_text(self, text):
        if(hash(text) not in self.text_windows):
            self.text_windows[hash(text)] = self.vis.text(text, opts = dict(caption = text))
        else:
            self.vis.text(text, win = self.text_windows[hash(text)], opts = dict(caption = text))

    def plot_train_test_loss(self, loss_key, iteration, losses_dict, caption_dict, label):
        colors = ['r', 'g', 'black', 'darkorange', 'olive', 'palevioletred', 'rosybrown', 'cyan', 'slategray', 'darkmagenta', 'linen', 'chocolate']

        x = [i for i in range(iteration, iteration + len(losses_dict["TRAIN_LOSS_KEY"]))]
        loss_keys = list(losses_dict.keys())
        caption_keys = list(caption_dict.keys())

        plt.plot(x, losses_dict[loss_keys[0]], colors[0], label=str(caption_dict[caption_keys[0]]))
        plt.plot(x, losses_dict[loss_keys[1]], colors[1], label=str(caption_dict[caption_keys[1]]))
        plt.legend(loc='lower right')

        if loss_key not in self.loss_windows:
            self.loss_windows[loss_key] = self.vis.matplot(plt, opts = dict(caption = "Losses" + " " + str(label)))
        else:
            self.vis.matplot(plt, win = self.loss_windows[loss_key], opts = dict(caption = "Losses" + " " + str(label)))