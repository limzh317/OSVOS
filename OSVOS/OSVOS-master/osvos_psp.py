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

# PyTorch includes
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
from OSVOS.pspnet import PSPNet

import torch
import matplotlib.pyplot as plt


class DAVIS2017(Dataset):

    def __init__(self, split, db_root_dir=None, transforms=None):
        self.split = split
        self.db_root_dir = db_root_dir
        self.transforms = None
        self.all_label = True
        self.crop = False
        self.channel3 = False
        self.denoise = False
        self.interval = 0
        self.cla = False
        self.filter = False


        images_list = []
        labels_list = []

        if self.split == 'train':
            fname = 'train'
        elif self.split == 'val':
            fname = 'val'
        else:
            raise Exception('Only support train and val!')
        if self.crop:
            frame_image = 'crop_images/'
            frame_label = 'crop_labels/'
        elif self.denoise:
            frame_image = '../../share_data/denoise_images/'
            frame_label = 'labels/'
        elif self.filter:
            frame_image = 'images_filter/'
            frame_label = 'labels_filter/'
        else:
            frame_image = 'images/'
            frame_label = 'labels/'


        images = os.listdir(os.path.join(db_root_dir, frame_image, fname))
        images.sort()
        #print(images)

        images_path = list(map(lambda x: os.path.join(db_root_dir, frame_image, fname, x), images))
        images_list.extend(images_path)

        labels = os.listdir(os.path.join(db_root_dir, frame_label, fname))
        labels.sort()
        #print(labels)

        labels_path = list(map(lambda x: os.path.join(db_root_dir, frame_label, fname, x), labels))
        labels_list.extend(labels_path)

        #ipdb.set_trace()
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
            image = cv2.imread(os.path.join(self.images_list[index])) # X, Y, 3

        if self.cla:
            clahe = cv2.createCLAHE(clipLimit=self.cla, tileGridSize=(8, 8))
            #ipdb.set_trace()
            image[:, :, 0] = image[:, :, 0]
            image[:, :, 1] = clahe.apply(image[:, :, 1].astype(np.uint8))
            image[:, :, 2] = clahe.apply(image[:, :, 2].astype(np.uint8))






        image = cv2.resize(image, (400, 400), cv2.INTER_CUBIC)

        label = cv2.imread(os.path.join(self.labels_list[index]), 0)
        label = cv2.resize(label, (400, 400), cv2.INTER_NEAREST)


        if self.all_label == False:

            label[label == 2] = 0
            label[label == 3] = 0
            label[label == 4] = 2


        image = np.transpose(image, axes=[2, 0, 1])

        sample = {'image': image, 'label': label, 'raw_image': image}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.images_list)




class DavisParent:

    def __init__(self, model_name):
        # parameter
        self.paras = edict()

        self.paras.epoch = 0

        # optim parameter
        self.paras.lr = 1e-3
        self.paras.momentum = 0.99
        self.paras.weight_decay = 1e-4

        self.paras.num_epochs = 120

        # data
        self.paras.train_batch_size = 8
        self.paras.test_batch_size = 8
        self.train_parent_loader = None
        self.val_parent_loader = None

        # loss
        self.paras.best_val_dice = 0
        self.model_name = model_name
        #Todo mofify while using node>36
        self.db_root_dir = 'OSVOS/data/'
        self.basepath = '/p300/segmentation/'

        self.num_workers = 16
        self.device = torch.device("cuda")
        self.resume = False
        self.checkpoint_dir = None
        self.model = None

        self.opt = None
        self.criterion = class_balanced_cross_entropy_loss

        # visualization
        self.use_visdom = True
        self.vis = Dashboard(env=self.model_name, server='http://10.10.10.100', port=31370)
        self.adjust_lr = adjust_learning_rate

        self.all_label = True
        self.class_balance = 'Batch_size'
        self.loss_exclude = False
        self.cross_entropy = False
        self.loss_epoch = True

    def init_dataloader(self):

        self.data_loader = {}

        train_transform = Compose([tf.RandomHorizontalFlip(),
                                   tf.RandomVerticalFlip(),
                                   #tf.ScaleNRotate(rots=(-15, 15), scales=(.9, 1.1)),
                                   #tf.Gamma(gamma = 1.5),
                                   ColorJitter(brightness=0.8, contrast=0.8, saturation=0.4, hue=0.1),
                                   tf.ToTensor(),
                                   tf.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])

        # train_transform = Compose([tf.RandomHorizontalFlip(),
        #                            tf.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
        #                            tf.ToTensor(),
        #                            tf.Normalize(mean=[0.485, 0.456, 0.406],
        #                                         std=[0.229, 0.224, 0.225])])

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

            self.model = PSPNet().to(self.device)
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
                                               #datetime.datetime.now().strftime("%Y-%m-%d-%X"))

            if not os.path.exists(self.checkpoint_dir):

                os.makedirs(self.checkpoint_dir)

            self.model = PSPNet().to(self.device)


        self.model = torch.nn.DataParallel(self.model)
        return self

    def vis_plot(self, vis_mask, vis_gt, vis_image, epoch, val_ind):
        if np.mod(epoch, 10) == 0:
            plt.axis('off')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)

            plt.subplot(1, 3, 1)
            plt.title('Prediction')
            plt.imshow(vis_mask, cmap="gray")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title('GT')
            plt.imshow(vis_gt, cmap="gray")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title('Raw image')
            plt.imshow(vis_image[0], cmap="gray")
            plt.axis('off')

            saveRoot = 'data/Output/val/' + self.model_name + '/epoch_' + str(epoch) + '/'
            if not os.path.exists(saveRoot):
                os.makedirs(saveRoot)
            plt.savefig(saveRoot + str(val_ind) + '.png', dpi=300)


    def train_parent_epoch(self, epoch):

        loss_epoch = []
        iou_epoch = []
        acc_epoch = []
        dice_coeff_epoch = []
        dice_coeff_all_epoch = []

        num_iter = len(self.train_parent_loader)

        for batch_idx, sample in enumerate(self.train_parent_loader):
            start = time.time()
            img, labels, raw_img = sample['image'], sample['label'], sample['raw_image']
            img, labels = img.float().to(self.device), labels.float().to(self.device)

            #ipdb.set_trace()
            output = self.model(img)
            loss_batch = 0


            if self.class_balance == 'Batch_size':

                Voxnum = float(self.paras.train_batch_size * labels.shape[1] * labels.shape[2])
                if self.all_label:
                    weights = [Voxnum/float(torch.sum(labels == 0)), Voxnum/float(torch.sum(labels == 1)), Voxnum/float(torch.sum(labels == 2))
                           ,Voxnum/float(torch.sum(labels == 3)), Voxnum/float(torch.sum(labels == 4))]
                    #weights = [5, 100,20,30,100]
                else:
                    weights = [Voxnum / float(torch.sum(labels == 0)), Voxnum / float(torch.sum(labels == 1)),
                               Voxnum / float(torch.sum(labels == 2))]
                weights_norm = torch.FloatTensor(weights)/torch.FloatTensor(weights).sum()

                class_weights = torch.FloatTensor(weights_norm).to(self.device)

                loss = F.nll_loss(F.log_softmax(output), labels.long(),class_weights)


                #loss_middle[middle_idx] = class_balanced_cross_entropy_loss(output=output, label=labels)
                #loss_middle[middle_idx] = F.nll_loss(F.log_softmax(output), labels.long())


                loss_batch += loss


            bin_mask = torch.argmax(torch.softmax(output, 1), dim = 1)

            iou_all, iou_mean, acc, dice_coeff = evaluate_iou(bin_mask, labels, self.all_label)

            te = time.time() - start

            if self.use_visdom:
                vis_image = raw_img[0]
                vis_mask = bin_mask[0].data.cpu().numpy() * 50
                vis_gt = labels[0].data.cpu().numpy() * 50
                vis_train = np.concatenate((vis_gt, vis_mask), axis=0)
                #vis_train = np.concatenate((vis_gt, vis_mask, vis_image), axis=1)

                self.vis.show_feature_maps(features=vis_train, datatype='Training sample')
                self.vis.show_img(img=vis_image, datatype='Training Raw Image')



            loss_epoch.append(loss_batch.data.cpu().numpy())
            iou_epoch.append(iou_mean)
            acc_epoch.append(acc)
            dice_coeff_epoch.append(dice_coeff.mean())
            #All dice
            dice_coeff_all_epoch.append(dice_coeff)


            self.opt.zero_grad()
            loss_batch.backward()
            self.opt.step()

            print("{0}: [{1}][{2}/{3}] Time {batch_time:.3f} Loss {loss:.4f} IOU {iou:.4f} Acc {acc:.4f} dice {dice_coeff:.4f}".format(
                'Training', epoch, batch_idx, num_iter, batch_time=te, loss=loss_batch.data[0], iou=iou_mean,
                 acc = acc, dice_coeff = dice_coeff.mean()))
            if self.all_label:
                print("IOU: {:.4f} {:.4f} {:.4f} {:.4f}".format(iou_all[0], iou_all[1], iou_all[2], iou_all[3]))
                print("DICE: {:.4f} {:.4f} {:.4f} {:.4f}".format(dice_coeff[0], dice_coeff[1], dice_coeff[2], dice_coeff[3]))

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
        dice_coeff_each = dice_coeff_all_epoch.mean(axis = 0)

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
                output = self.model(img)


                if self.class_balance == 'Batch_size':

                    Voxnum = float(self.paras.train_batch_size * labels.shape[1] * labels.shape[2])
                    if self.all_label:
                        weights = [Voxnum / float(torch.sum(labels == 0)), Voxnum / float(torch.sum(labels == 1)),
                                   Voxnum / float(torch.sum(labels == 2))
                            , Voxnum / float(torch.sum(labels == 3)), Voxnum / float(torch.sum(labels == 4))]
                    else:
                        weights = [Voxnum / float(torch.sum(labels == 0)), Voxnum / float(torch.sum(labels == 1)),
                                   Voxnum / float(torch.sum(labels == 2))]
                    weights_norm = torch.FloatTensor(weights) / torch.FloatTensor(weights).sum()

                    class_weights = torch.FloatTensor(weights_norm).to(self.device)


                    loss = F.nll_loss(F.log_softmax(output), labels.long(), class_weights)



                    loss_batch += loss


                te = time.time() - start
                bin_mask = torch.argmax(torch.softmax(output,1), dim = 1)

                iou_all, iou_mean, acc, dice_coeff = evaluate_iou(bin_mask, labels, self.all_label)

                if self.use_visdom:
                    vis_image = raw_img[0]
                    vis_mask = bin_mask[0].data.cpu().numpy() * 50
                    vis_gt = labels[0].data.cpu().numpy() * 50
                    vis_val = np.concatenate((vis_gt, vis_mask), axis=0)
                    #vis_val = np.concatenate((vis_gt, vis_mask, vis_image), axis=1)

                    self.vis.show_feature_maps(features=vis_val, datatype='Validation sample')
                    self.vis.show_img(img=vis_image, datatype='Validation Raw Image')

                    self.vis_plot(vis_mask, vis_gt, vis_image, epoch, val_ind)
                    val_ind += 1

                loss_epoch.append(loss_batch.data.cpu().numpy())
                iou_epoch.append(iou_mean)
                acc_epoch.append(acc)
                dice_coeff_epoch.append(dice_coeff.mean())
                # All dice
                dice_coeff_all_epoch.append(dice_coeff)

                print(
                    "{0}: [{1}][{2}/{3}] Time {batch_time:.3f} Loss {loss:.4f} IOU {iou:.4f} Acc {acc:.4f} dice {dice_coeff:.4f}".format(
                        'Validation', epoch, batch_idx, num_iter, batch_time=te, loss=loss_batch.data[0], iou=iou_mean,
                        acc=acc, dice_coeff = dice_coeff.mean()))
                if self.all_label:
                    print("IOU: {:.4f} {:.4f} {:.4f} {:.4f}".format(iou_all[0], iou_all[1], iou_all[2], iou_all[3]))
                    print("DICE: {:.4f} {:.4f} {:.4f} {:.4f}".format(dice_coeff[0], dice_coeff[1], dice_coeff[2], dice_coeff[3]))

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
        dice_coeff_each = dice_coeff_all_epoch.mean(axis = 0) #Four means


        return  loss, iou, acc, dice_coeff, dice_coeff_each




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

        self.opt = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.paras.lr*0.01}], lr=self.paras.lr,
                                     weight_decay=self.paras.weight_decay)


        for epoch in range(self.paras.epoch, self.paras.num_epochs):

            start = time.time()
            #ipdb.set_trace()



            loss_train_epoch, iou_train_epoch, acc_train_epoch, dice_coeff_train_epoch, dice_coeff_each_train_epoch \
                = self.train_parent_epoch(epoch=epoch)

            loss_val_epoch, iou_val_epoch, acc_val_epoch, dice_coeff_val_epoch, dice_coeff_each_val_epoch \
                = self.val_parent_epoch(epoch=epoch)


            self.adjust_lr(optimizer=self.opt, init_lr=self.paras.lr, epoch=epoch, step=10)

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


            print ('epoch:%d, epoch_time:%.4f, loss_train:%.4f, loss_val:%.4f, iou_train:%.4f, iou_val:%.4f, acc_train:%.4f, '
                   'acc_val:%.4f, dice_train:%.4f, dice_val:%.4f'%(
                epoch, te, loss_train_epoch, loss_val_epoch, iou_train_epoch, iou_val_epoch, acc_train_epoch, acc_val_epoch,
                dice_coeff_train_epoch, dice_coeff_val_epoch))

            self.paras.epoch = epoch + 1

            # save model
            if dice_coeff_val_epoch > (dice_coeff_each_val_epoch[0] + dice_coeff_each_val_epoch[3]) / 2:
                print("Saving checkpoints.")
                self.paras.best_val_dice = (dice_coeff_each_val_epoch[0] + dice_coeff_each_val_epoch[3]) / 2

                save_checkpoint_lite(checkpoint_dir=self.checkpoint_dir,
                                     model=self.model,
                                     optim=self.opt,
                                     paras=self.paras)

            if self.use_visdom:

                self.vis.show_curve(train_data=np.array(loss_train_epochs), val_data=np.array(loss_val_epochs),
                                    datatype='loss_epoch')
                self.vis.show_curve(train_data=np.array(iou_train_epochs), val_data=np.array(iou_val_epochs),
                                    datatype='iou_epoch')
                self.vis.show_curve(train_data=np.array(acc_train_epochs), val_data=np.array(acc_val_epochs),
                                    datatype='acc_epoch')
                self.vis.show_curve(train_data=np.array(dice_coeff_train_epochs), val_data=np.array(dice_coeff_val_epochs),
                                    datatype='dice_epoch')
                if self.all_label:
                    self.vis.show_curve(train_data=np.array(dice_coeff_each_train_epochs)[:,0],
                                        val_data=np.array(dice_coeff_each_val_epochs)[:,0],
                                        datatype='dice_epoch_mito')

                    self.vis.show_curve(train_data=np.array(dice_coeff_each_train_epochs)[:,1],
                                        val_data=np.array(dice_coeff_each_val_epochs)[:,1],
                                        datatype='dice_epoch_mem')

                    self.vis.show_curve(train_data=np.array(dice_coeff_each_train_epochs)[:,2],
                                        val_data=np.array(dice_coeff_each_val_epochs)[:,2],
                                        datatype='dice_epoch_nu')

                    self.vis.show_curve(train_data=np.array(dice_coeff_each_train_epochs)[:,3],
                                        val_data=np.array(dice_coeff_each_val_epochs)[:,3],
                                        datatype='dice_epoch_gr')
                else:
                    self.vis.show_curve(train_data=np.array(dice_coeff_each_train_epochs)[:, 0],
                                        val_data=np.array(dice_coeff_each_val_epochs)[:, 0],
                                        datatype='dice_epoch_mito')


                    self.vis.show_curve(train_data=np.array(dice_coeff_each_train_epochs)[:, 1],
                                        val_data=np.array(dice_coeff_each_val_epochs)[:, 1],
                                        datatype='dice_epoch_gr')

if __name__ == '__main__':


    model = DavisParent(model_name='0723_psp_lr1e-3_Adam_decay10_03')
    model.init_net()
    model.main()