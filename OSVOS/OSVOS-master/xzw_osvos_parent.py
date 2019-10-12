# -*- coding: utf-8 -*-
"""
OSVOS residual unet
"""
import cv2
import os
import sys
import time
import datetime
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
sys.path.append(os.path.abspath('./'))
import torch.nn as nn
import torch
from torch.utils import data
from torchvision.transforms import Compose, ColorJitter
import OSVOS.libs.transforms as tf
from OSVOS.libs.utils import evaluate_iou, class_balanced_cross_entropy_loss, \
    adjust_learning_rate, load_checkpoint, save_checkpoint_lite, load_checkpoint_lite
from OSVOS.libs.visualize import Dashboard
from torch.autograd import Variable
from OSVOS.libs.models import ResUNet2, ResUNet1
import ipdb
from OSVOS.MuticlassDiceLoss import MulticlassDiceLoss, DiceLoss, DiceCoeff, dice_coeff

import warnings
import math
from operator import mul
from functools import reduce

import torch
import matplotlib.pyplot as plt

class FocalLoss(nn.Module):
    def __init__(self, class_num=5, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)
        if P.dim() > 2:
            P = P.view(P.size(0), P.size(1), -1)
            P = P.permute(0, 2, 1).contiguous()
            P = P.view(-1, P.size(-1))
        # print(P.shape)
        #print(targets.shape)
        ids = targets.view(-1, 1)
        class_mask = inputs.data.new(ids.size(0), C).fill_(0)
        class_mask = Variable(class_mask)

        #print(ids.shape)
        class_mask = class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)
        if class_mask.device != P.device:
            class_mask = class_mask.to(P.device)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
class DAVIS2017(Dataset):

    def __init__(self, split, db_root_dir=None, transforms=None):
        self.split = split
        self.db_root_dir = db_root_dir
        self.transforms = None
        self.all_label = True
        self.channel3 = True
        self.interval = 0

        images_list = []
        labels_list = []

        if self.split == 'train':
            fname = 'train'
        elif self.split == 'val':
            fname = 'val'
        else:
            raise Exception('Only support train and val!')

        frame_image = 'images_file/'
        frame_label = 'labels_file/'

        images = os.listdir(os.path.join(db_root_dir, frame_image, fname))
        images.sort()
        if '.DS_Store' in images:
            images.remove('.DS_Store')

        images_path = list(map(lambda x: os.path.join(db_root_dir, frame_image, fname, x), images))
        if os.path.join(db_root_dir, frame_image, fname, '.DS_Store') in images_path:
            images_path.remove(os.path.join(db_root_dir, frame_image, fname, '.DS_Store'))
        images_list.extend(images_path)

        labels = os.listdir(os.path.join(db_root_dir, frame_label, fname))
        if '.DS_Store' in labels:
            labels.remove('.DS_Store')

        labels.sort()

        labels_path = list(map(lambda x: os.path.join(db_root_dir, frame_label, fname, x), labels))

        labels_list.extend(labels_path)

        print(len(labels_list), len(images_list))
        assert (len(labels_list) == len(images_list))

        self.images_list = images_list
        self.labels_list = labels_list

        print('Done initializing ' + fname + ' Dataset')

    def __getitem__(self, index):

        if self.channel3:
            image = cv2.imread(os.path.join(self.images_list[index]), 0)

            if index < self.interval + 1:
                image_fore = image
                image_after = cv2.imread(os.path.join(self.images_list[index + self.interval]), 0)
            elif index > len(self.images_list) - self.interval - 1:
                image_fore = cv2.imread(os.path.join(self.images_list[index - self.interval]), 0)
                image_after = image
            else:
                image_fore = cv2.imread(os.path.join(self.images_list[index - self.interval]), 0)
                image_after = cv2.imread(os.path.join(self.images_list[index + self.interval]), 0)

            if image_fore.shape == image.shape and image.shape == image_after.shape:
                image = np.array([image_fore, image, image_after])  # 3, X, Y
                image = np.transpose(image, axes=[1, 2, 0])  # X, Y, 3
            else:
                image = cv2.imread(os.path.join(self.images_list[index]))  # X, Y, 3

        else:
            image = cv2.imread(os.path.join(self.images_list[index]))  # X, Y, 3


        image = cv2.resize(image, (400, 400), cv2.INTER_CUBIC)

        label = cv2.imread(os.path.join(self.labels_list[index]), 0)
        label = cv2.resize(label, (400, 400), cv2.INTER_NEAREST)

        if self.all_label == False:
            label[label == 2] = 0
            label[label == 3] = 0
            label[label == 4] = 2

        image = np.transpose(image, axes=[2, 0, 1])

        sample = {'image': image, 'label': label, 'raw_image': image}

        return sample

    def __len__(self):
        return len(self.images_list)


class DavisParent:

    def __init__(self, model_name):
        # parameter
        self.paras = edict()

        self.paras.epoch = 0

        # optim parameter
        self.paras.lr = 1e-4
        self.paras.momentum = 0.99
        self.paras.weight_decay = 1e-4

        self.paras.num_epochs = 120

        # data
        self.paras.train_batch_size = 2
        self.paras.test_batch_size = 2
        self.train_parent_loader = None
        self.val_parent_loader = None

        # loss
        self.paras.best_val_dice = 0
        self.model_name = model_name
        # Todo mofify while using node>36
        self.db_root_dir = '/root/root/Experiment/'
        self.basepath = '/root/root/Experiment/OSVOS/segmentation/'

        self.num_workers = 16
        self.device = torch.device("cuda")
        self.resume = False
        self.checkpoint_dir = None
        self.model = None

        self.opt = None
        self.criterion = FocalLoss()
        self.adjust_lr = adjust_learning_rate

        self.all_label = True
        self.class_balance = 'Batch_size'
        self.loss_exclude = False
        self.cross_entropy = False
        self.loss_epoch = True

        self.deeplab = False

    def init_dataloader(self):

        self.data_loader = {}

        train_transform = Compose([tf.RandomHorizontalFlip(),
                                   tf.ScaleNRotate(rots=(-15, 15), scales=(.9, 1.1)),
                                   # tf.Gamma(gamma = 0.8),
                                   ColorJitter(brightness=0.8, contrast=0.8, saturation=0.4, hue=0.1),
                                   tf.ToTensor(),
                                   tf.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])


        val_transform = Compose([tf.ToTensor(),
                                 tf.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])

        train_dataset = DAVIS2017(split='train', db_root_dir=self.db_root_dir,
                                  transforms=train_transform)

        self.train_parent_loader = data.DataLoader(train_dataset, batch_size=self.paras.train_batch_size,
                                                   shuffle=True, num_workers=self.num_workers)

        val_dataset = DAVIS2017(split='val', db_root_dir=self.db_root_dir,
                                transforms=val_transform)

        self.val_parent_loader = data.DataLoader(val_dataset, batch_size=self.paras.test_batch_size,
                                                 shuffle=True, num_workers=self.num_workers)

    def init_net(self):
        # self.checkpoint_dir = os.path.join(self.basepath, 'checkpoints', self.model_name,
        #                                    datetime.datetime.now().strftime("%Y-%m-%d-%X"))
        self.checkpoint_dir = os.path.join(self.basepath, 'checkpoints', self.model_name)

        if self.resume and os.path.exists(self.checkpoint_dir):
            print('Resume from {}'.format(self.checkpoint_dir))

            state_dict, optim_dict = load_checkpoint_lite(self.checkpoint_dir)

            self.model = ResUNet2()
            model_state = self.model.state_dict()

            model_pretrained = {}
            for k, v in state_dict.items():

                # ipdb.set_trace()
                k = k.replace('module.', '')
                if k in model_state.keys():
                    print(k)
                    model_pretrained[k] = v
            model_state.update(model_pretrained)

            self.model.load_state_dict(state_dict=model_state)
            # self.model.load_state_dict(state_dict)

        else:
            self.checkpoint_dir = os.path.join(self.basepath, 'checkpoints', self.model_name)
            # datetime.datetime.now().strftime("%Y-%m-%d-%X"))

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            # if self.deeplab:
            #     self.model = DeepLabv3_plus().to(self.device)
            # else:
            self.model = ResUNet2().to(self.device)

        self.model = torch.nn.DataParallel(self.model)
        return self

    def class_balance_weight(self, num_ex, num_mito, num_mem, num_nu, num_gr):
        rate = (num_ex + num_mem + num_nu) / (num_mito + num_gr)
        # print(rate)
        weights = [10000/num_ex, 10000/num_mito, 10000/num_mem, 10000/num_nu, 10000/num_gr ]
        # print(weights)
        return weights

    def train_parent_epoch(self, epoch):

        loss_epoch = []
        iou_epoch = []
        acc_epoch = []
        dice_coeff_epoch = []
        dice_coeff_all_epoch = []

        num_iter = len(self.train_parent_loader)

        #     if batch_idx >2:
        #         break
        #     print(batch_idx)
        #     start = time.time()
        #     img, labels, raw_img = sample['image'], sample['label'], sample['raw_image']
        #     img, labels = img.float().to(model.device), labels.float().to(model.device)
        #     outputs = model.model(img)
        # a=[]
        # for i in range(0,4791):
        #     if len(np.unique(model.train_parent_loader.dataset[i]['label']))!=5:
        #         print(i,"  ",np.unique(model.train_parent_loader.dataset[i]['label']))



        for batch_idx, sample in enumerate(self.train_parent_loader):
            start = time.time()
            img, labels, raw_img = sample['image'], sample['label'], sample['raw_image']
            img, labels = img.float().to(self.device), labels.float().to(self.device)
            # ipdb.set_trace()

            # ipdb.set_trace()
            outputs = self.model(img)
            loss_batch = 0
            num_middle = len(outputs)
            loss_middle = [0] * num_middle

            for middle_idx in range(num_middle):

                output = outputs[middle_idx]

                if self.class_balance == 'Batch_size':

                    Voxnum = float(self.paras.train_batch_size * labels.shape[1] * labels.shape[2])
                    if self.all_label:
                        # ipdb.set_trace()
                        # weights = [Voxnum/float(torch.sum(labels == 0)), Voxnum/float(torch.sum(labels == 1)), Voxnum/float(torch.sum(labels == 2))
                        #        ,Voxnum/float(torch.sum(labels == 3)), Voxnum/float(torch.sum(labels == 4))]
                        # print(weights)
                        weights = self.class_balance_weight(float(torch.sum(labels == 0)),
                                                            float(torch.sum(labels == 1)),
                                                            float(torch.sum(labels == 2)),
                                                            float(torch.sum(labels == 3)),
                                                            float(torch.sum(labels == 4)))
                        # if middle_idx == 1:
                        #     print(weights)
                        # weights = [5, 100,20,30,100]
                    else:
                        weights = [Voxnum / float(torch.sum(labels == 0)), Voxnum / float(torch.sum(labels == 1)),
                                   Voxnum / float(torch.sum(labels == 2))]

                    weights_norm = torch.FloatTensor(weights) / torch.FloatTensor(weights).sum()

                    class_weights = torch.FloatTensor(weights_norm).to(self.device)
                    if self.loss_exclude:
                        if self.all_label:
                            n_class = 5
                        else:
                            n_class = 3
                            tmp = labels.clone()
                            tmp[tmp > 0] = 1
                            tmp = tmp[:, np.newaxis, :, :]
                            tmp = tmp.repeat(1, n_class, 1, 1)

                            loss_middle[middle_idx] = F.nll_loss(F.log_softmax(output) * tmp, labels.long(),
                                                                 class_weights)
                    elif self.cross_entropy:
                        loss_middle[middle_idx] = F.cross_entropy(output, labels.long(), class_weights)


                    else:
                        # ipdb.set_trace()
                        loss_middle[middle_idx] = self.criterion(F.log_softmax(output), labels.long())

                elif self.class_balance == 'Image_size':

                    Voxnum = float(labels.shape[1] * labels.shape[2])
                    for img_idx in range(labels.shape[0]):
                        loss_per_image = 0
                        weights = [Voxnum / float(torch.sum(labels == 0)), Voxnum / float(torch.sum(labels == 1)),
                                   Voxnum / float(torch.sum(labels == 2))
                            , Voxnum / float(torch.sum(labels == 3)), Voxnum / float(torch.sum(labels == 4))]
                        weights_norm = torch.FloatTensor(weights) / torch.FloatTensor(weights).sum()

                        class_weights = torch.FloatTensor(weights_norm).to(self.device)
                        # ipdb.set_trace()
                        loss_per_image += F.nll_loss(F.log_softmax(output[img_idx][np.newaxis, :, :, :]),
                                                     labels[img_idx][np.newaxis, :, :].long(), class_weights)
                        # ipdb.set_trace()
                        # loss_per_image += F.nll_loss(F.log_softmax(output[img_idx]),labels[img_idx].long(), class_weights)

                    loss_middle[middle_idx] = loss_per_image

            if self.loss_epoch:
                loss_batch += (1 - epoch / self.paras.num_epochs) * sum(loss_middle[:-1]) / (num_middle - 1) + \
                              loss_middle[-1]
            else:
                loss_batch += loss_middle[-1]

            output = outputs[-1]

            bin_mask = torch.argmax(torch.softmax(output, 1), dim=1)

            iou_all, iou_mean, acc, dice_coeff = evaluate_iou(bin_mask, labels, self.all_label)

            te = time.time() - start

            loss_epoch.append(loss_batch.data.cpu().numpy())
            iou_epoch.append(iou_mean)
            acc_epoch.append(acc)
            dice_coeff_epoch.append(dice_coeff.mean())
            # All dice
            dice_coeff_all_epoch.append(dice_coeff)

            self.opt.zero_grad()
            loss_batch.backward()
            self.opt.step()
            if batch_idx%4==0:
                print(
                    "{0}: [{1}][{2}/{3}] Time {batch_time:.3f} Loss {loss:.4f} IOU {iou:.4f} Acc {acc:.4f} dice {dice_coeff:.4f}".format(
                        'Training', epoch, batch_idx, num_iter, batch_time=te, loss=loss_batch.item(), iou=iou_mean,
                        acc=acc, dice_coeff=dice_coeff.mean()))
                if self.all_label:
                    print("IOU: {:.4f} {:.4f} {:.4f} {:.4f}".format(iou_all[0], iou_all[1], iou_all[2], iou_all[3]))
                    print("DICE: {:.4f} {:.4f} {:.4f} {:.4f}".format(dice_coeff[0], dice_coeff[1], dice_coeff[2],
                                                                     dice_coeff[3]))

                else:
                    print("IOU: {:.4f} {:.4f}".format(iou_all[0], iou_all[1]))
                    print("DICE: {:.4f} {:.4f}".format(dice_coeff[0], dice_coeff[1]))
                print(weights)

        loss_epoch = np.array(loss_epoch)
        iou_epoch = np.array(iou_epoch)
        acc_epoch = np.array(acc_epoch)
        dice_coeff_epoch = np.array(dice_coeff_epoch)
        dice_coeff_all_epoch = np.array(dice_coeff_all_epoch)

        loss = loss_epoch.mean()
        iou = iou_epoch.mean()
        acc = acc_epoch.mean()
        dice_coeff = dice_coeff_epoch.mean()
        dice_coeff_each = dice_coeff_all_epoch.mean(axis=0)

        return loss, iou, acc, dice_coeff, dice_coeff_each

    def val_parent_epoch(self, epoch):

        loss_epoch = []
        iou_epoch = []
        acc_epoch = []
        dice_coeff_epoch = []
        dice_coeff_all_epoch = []

        num_iter = len(self.val_parent_loader)
        val_ind = 1
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.val_parent_loader):



                start = time.time()
                img, labels, raw_img = sample['image'], sample['label'], sample['raw_image']
                img, labels = img.float().to(self.device), labels.float().to(self.device)
                loss_batch = 0
                outputs = self.model(img)

                num_middle = len(outputs)
                loss_middle = [0] * num_middle
                for middle_idx in range(num_middle):
                    output = outputs[middle_idx]

                    # exist_label = np.unique(labels[img_idx])

                    if self.class_balance == 'Batch_size':

                        Voxnum = float(self.paras.train_batch_size * labels.shape[1] * labels.shape[2])
                        if self.all_label:
                            weights = self.class_balance_weight(float(torch.sum(labels == 0)),
                                                                float(torch.sum(labels == 1)),
                                                                float(torch.sum(labels == 2)),
                                                                float(torch.sum(labels == 3)),
                                                                float(torch.sum(labels == 4)))

                        else:
                            weights = [Voxnum / float(torch.sum(labels == 0)), Voxnum / float(torch.sum(labels == 1)),
                                       Voxnum / float(torch.sum(labels == 2))]

                        weights_norm = torch.FloatTensor(weights) / torch.FloatTensor(weights).sum()

                        class_weights = torch.FloatTensor(weights_norm).to(self.device)
                        if self.loss_exclude:
                            if self.all_label:
                                n_class = 5
                            else:
                                n_class = 3
                                tmp = labels.clone()
                                tmp[tmp > 0] = 1
                                tmp = tmp[:, np.newaxis, :, :]
                                tmp = tmp.repeat(1, n_class, 1, 1)

                                loss_middle[middle_idx] = F.nll_loss(F.log_softmax(output) * tmp, labels.long(),
                                                                     class_weights)
                        elif self.cross_entropy:
                            loss_middle[middle_idx] = F.cross_entropy(output, labels.long(), class_weights)


                        else:
                            # ipdb.set_trace()
                            loss_middle[middle_idx] = self.criterion(F.log_softmax(output), labels.long())
                        # loss_middle[middle_idx] = class_balanced_cross_entropy_loss(output=output, label=labels)
                        # loss_middle[middle_idx] = F.nll_loss(F.log_softmax(output), labels.long())



                    elif self.class_balance == 'Image_size':

                        Voxnum = float(labels.shape[1] * labels.shape[2])

                        for img_idx in range(labels.shape[0]):
                            loss_per_image = 0

                            weights = [Voxnum / float(torch.sum(labels == 0)), Voxnum / float(torch.sum(labels == 1)),
                                       Voxnum / float(torch.sum(labels == 2))

                                , Voxnum / float(torch.sum(labels == 3)), Voxnum / float(torch.sum(labels == 4))]

                            weights_norm = torch.FloatTensor(weights) / torch.FloatTensor(weights).sum()

                            class_weights = torch.FloatTensor(weights_norm).to(self.device)

                            loss_per_image += F.nll_loss(F.log_softmax(output[img_idx][np.newaxis, :, :, :]),

                                                         labels[img_idx][np.newaxis, :, :].long(), class_weights)

                        loss_middle[middle_idx] = loss_per_image

                # loss_batch += (1 - epoch / self.paras.num_epochs) * sum(loss_middle[:-1]) / (num_middle - 1) + \
                #               loss_middle[-1]
                if self.loss_epoch:
                    loss_batch += (1 - epoch / self.paras.num_epochs) * sum(loss_middle[:-1]) / (num_middle - 1) + \
                                  loss_middle[-1]
                else:
                    loss_batch += loss_middle[-1]

                te = time.time() - start
                output = outputs[-1]
                bin_mask = torch.argmax(torch.softmax(output, 1), dim=1)

                iou_all, iou_mean, acc, dice_coeff = evaluate_iou(bin_mask, labels, self.all_label)

                loss_epoch.append(loss_batch.data.cpu().numpy())
                iou_epoch.append(iou_mean)
                acc_epoch.append(acc)
                dice_coeff_epoch.append(dice_coeff.mean())
                # All dice
                dice_coeff_all_epoch.append(dice_coeff)
                if batch_idx % 4 == 0:
                    print(
                        "{0}: [{1}][{2}/{3}] Time {batch_time:.3f} Loss {loss:.4f} IOU {iou:.4f} Acc {acc:.4f} dice {dice_coeff:.4f}".format(
                            'Validation', epoch, batch_idx, num_iter, batch_time=te, loss=loss_batch.item(), iou=iou_mean,
                            acc=acc, dice_coeff=dice_coeff.mean()))
                    if self.all_label:
                        print("IOU: {:.4f} {:.4f} {:.4f} {:.4f}".format(iou_all[0], iou_all[1], iou_all[2], iou_all[3]))
                        print("DICE: {:.4f} {:.4f} {:.4f} {:.4f}".format(dice_coeff[0], dice_coeff[1], dice_coeff[2],
                                                                         dice_coeff[3]))

                    else:
                        print("IOU: {:.4f} {:.4f}".format(iou_all[0], iou_all[1]))
                        print("DICE: {:.4f} {:.4f}".format(dice_coeff[0], dice_coeff[1]))

        loss_epoch = np.array(loss_epoch)
        iou_epoch = np.array(iou_epoch)
        acc_epoch = np.array(acc_epoch)
        dice_coeff_epoch = np.array(dice_coeff_epoch)
        dice_coeff_all_epoch = np.array(dice_coeff_all_epoch)

        loss = loss_epoch.mean()
        iou = iou_epoch.mean()
        acc = acc_epoch.mean()
        dice_coeff = dice_coeff_epoch.mean()
        dice_coeff_each = dice_coeff_all_epoch.mean(axis=0)  # Four means

        return loss, iou, acc, dice_coeff, dice_coeff_each

    def main(self):
        print(self.paras)
        self.init_dataloader()

        torch.cuda.synchronize()
        loss_train_epochs = []
        loss_val_epochs = []

        iou_train_epochs = []
        iou_val_epochs = []

        acc_train_epochs = []
        acc_val_epochs = []

        dice_coeff_train_epochs = []
        dice_coeff_val_epochs = []

        dice_coeff_each_train_epochs = []
        dice_coeff_each_val_epochs = []

        # ipdb.set_trace()
        if self.deeplab:
            self.opt = torch.optim.Adam([{'params': self.model.module.parameters(), 'lr': self.paras.lr * 0.01}],
                                        lr=self.paras.lr,
                                        weight_decay=self.paras.weight_decay)


        else:
            self.opt = torch.optim.Adam([{'params': self.model.module.encoder.parameters(), 'lr': self.paras.lr * 0.01},
                                         {'params': self.model.module.decoder.parameters(), 'lr': self.paras.lr}],
                                        lr=self.paras.lr,
                                        weight_decay=self.paras.weight_decay)

        for epoch in range(self.paras.epoch, self.paras.num_epochs):

            start = time.time()
            # ipdb.set_trace()

            loss_train_epoch, iou_train_epoch, acc_train_epoch, dice_coeff_train_epoch, dice_coeff_each_train_epoch \
                = self.train_parent_epoch(epoch=epoch)

            loss_val_epoch, iou_val_epoch, acc_val_epoch, dice_coeff_val_epoch, dice_coeff_each_val_epoch \
                = self.val_parent_epoch(epoch=epoch)

            self.adjust_lr(optimizer=self.opt, init_lr=self.paras.lr, epoch=epoch, step=30)

            te = time.time() - start
            loss_train_epochs.append(loss_train_epoch)
            loss_val_epochs.append(loss_val_epoch)

            iou_train_epochs.append(iou_train_epoch)
            iou_val_epochs.append(iou_val_epoch)

            acc_train_epochs.append(acc_train_epoch)
            acc_val_epochs.append(acc_val_epoch)

            dice_coeff_train_epochs.append(dice_coeff_train_epoch)
            dice_coeff_val_epochs.append(dice_coeff_val_epoch)

            dice_coeff_each_train_epochs.append(dice_coeff_each_train_epoch)
            dice_coeff_each_val_epochs.append(dice_coeff_each_val_epoch)

            print(
                'epoch:%d, epoch_time:%.4f, loss_train:%.4f, loss_val:%.4f, iou_train:%.4f, iou_val:%.4f, acc_train:%.4f, '
                'acc_val:%.4f, dice_train:%.4f, dice_val:%.4f' % (
                    epoch, te, loss_train_epoch, loss_val_epoch, iou_train_epoch, iou_val_epoch, acc_train_epoch,
                    acc_val_epoch,
                    dice_coeff_train_epoch, dice_coeff_val_epoch))

            self.paras.epoch = epoch + 1

            # save model

            if dice_coeff_val_epoch > self.paras.best_val_dice:
                print("Saving checkpoints.")
                self.paras.best_val_dice = dice_coeff_val_epoch

                save_checkpoint_lite(checkpoint_dir=self.checkpoint_dir,
                                     model=self.model,
                                     optim=self.opt,
                                     paras=self.paras)
#
# for i in range(0,4191):
#     image = cv2.imread(labels_list[i],0)
#     if len(np.unique(image))<2:
#         print(np.unique(image))
#         print(labels_list[i])




if __name__ == '__main__':
    model = DavisParent(model_name='0725_resnet101_lr-4_bs12_resnet101_adam_decay30_ck')
    model.init_net()
    model.main()
