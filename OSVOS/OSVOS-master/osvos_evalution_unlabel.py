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

import warnings
import math
from operator import mul
from functools import reduce
import matplotlib as mpl

import torch
import matplotlib.pyplot as plt


class DAVIS2017(Dataset):

    def __init__(self, split, db_root_dir=None, transforms=None, file= None):
        self.split = split
        self.db_root_dir = db_root_dir
        self.transforms = None
        self.all_label = True
        self.crop = False
        self.channel3 = False
        self.denoise = False
        self.interval = 0
        self.evalution = True
        self.cla = False
        self.filter = False

        images_list = []
        labels_list = []

        if self.split == 'train':
            fname = 'train'
        elif self.split == 'val':
            fname = 'train' #We merge valset to the trainset
        elif self.split == 'test':
            fname = 'test'
        else:
            raise Exception('Only support train, val, test!')
        if self.crop:
            frame_image = 'crop_images/'
            frame_label = 'crop_labels/'
        elif self.denoise:
            frame_image = '../../share_data/denoise_images/images/'
            frame_label = 'labels/'
        elif self.evalution:
            frame_image = 'images_new/'
            frame_label = 'labels_file/'
        elif self.filter:
            frame_image = 'images_new/'
        #     frame_label = 'labels_filter/'
        # else:
        #     frame_image = 'images_new/'
        #     #frame_label = 'labels/'

        images = os.listdir(os.path.join(db_root_dir, frame_image, file))
        images.sort()
        # print(images)

        images_path = list(map(lambda x: os.path.join(db_root_dir, frame_image, file, x), images))
        images_list.extend(images_path)

        #labels = os.listdir(os.path.join(db_root_dir, frame_label, fname, file))
        #labels.sort()
        # print(labels)

        #labels_path = list(map(lambda x: os.path.join(db_root_dir, frame_label, fname, file, x), labels))
        #labels_list.extend(labels_path)

        # ipdb.set_trace()
        print(len(images_list))
        #assert (len(labels_list) == len(images_list))

        self.images_list = images_list
        #self.labels_list = labels_list

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
            image[:, :, 0] = clahe.apply(image[:, :, 0].astype(np.uint8))
            image[:, :, 1] = clahe.apply(image[:, :, 1].astype(np.uint8))
            image[:, :, 2] = clahe.apply(image[:, :, 2].astype(np.uint8))






        image = cv2.resize(image, (400, 400), cv2.INTER_CUBIC)

        #label = cv2.imread(os.path.join(self.labels_list[index]), 0)
        #label = cv2.resize(label, (400, 400), cv2.INTER_NEAREST)


        # if self.all_label == False:
        #
        #     label[label == 1] = 2
        #     label[label == 4] = 2
        #     label[label == 2] = 1
        #     label[label == 3] = 2

        image = np.transpose(image, axes=[2, 0, 1])

        sample = {'image': image}

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
        self.paras.lr = 1e-4
        self.paras.momentum = 0.99
        self.paras.weight_decay = 1e-4

        self.paras.num_epochs = 120

        # data
        self.paras.train_batch_size = 8
        self.paras.test_batch_size = 8
        self.train_parent_loader = None
        self.val_parent_loader = None

        # loss
        self.paras.best_val_iou = 0
        self.model_name = model_name

        self.db_root_dir = 'OSVOS/data/'
        self.basepath = '/p300/segmentation/'

        self.num_workers = 16
        self.device = torch.device("cuda")
        self.resume = True
        self.checkpoint_dir = None
        self.model = None

        self.opt = None
        self.criterion = class_balanced_cross_entropy_loss

        # visualization
        self.use_visdom = False
        #elf.vis = Dashboard(env=self.model_name, server='http://10.10.10.100', port=31370)
        self.adjust_lr = adjust_learning_rate

        self.all_label = True
        self.class_balance = 'Batch_size'
        self.loss_exclude = False
        self.cross_entropy = False
        self.loss_epoch = True
        self.evalution = True
        self.split = 'test'




    def init_dataloader(self, file):

        self.data_loader = {}


        val_transform = Compose([tf.ToTensor(),
                                 tf.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])

        #ipdb.set_trace()
        val_dataset = DAVIS2017(split=self.split, db_root_dir=self.db_root_dir,
                                transforms=val_transform, file = file)

        self.val_parent_loader = data.DataLoader(val_dataset, batch_size=self.paras.test_batch_size,
                                                 shuffle=False, num_workers=self.num_workers)

    def init_net(self):
        # self.checkpoint_dir = os.path.join(self.basepath, 'checkpoints', self.model_name,
        #                                    datetime.datetime.now().strftime("%Y-%m-%d-%X"))
        self.checkpoint_dir = os.path.join(self.basepath, 'checkpoints', self.model_name)

        if self.resume and os.path.exists(self.checkpoint_dir):
            print('Resume from {}'.format(self.checkpoint_dir))

            state_dict, optim_dict= load_checkpoint_lite(self.checkpoint_dir)


            self.model = ResUNet2().to(self.device)
            model_state = self.model.state_dict()

            model_pretrained = {}
            for k,v in state_dict.items():

                #ipdb.set_trace()
                k = k.replace('module.', '')
                if k in model_state.keys():
                    print(k)
                    model_pretrained[k] = v
            model_state.update(model_pretrained)

            self.model.load_state_dict(state_dict=model_state)
            #self.model.load_state_dict(state_dict)

        else:
            self.checkpoint_dir = os.path.join(self.basepath, 'checkpoints', self.model_name,
                                               datetime.datetime.now().strftime("%Y-%m-%d-%X"))

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            self.model = ResUNet2().to(self.device)

        self.model = torch.nn.DataParallel(self.model)
        return self

    def vis_plot(self, vis_mask,  vis_image, file, val_ind):
        colors = ['white', 'blue', 'cyan', 'Lime', 'yellow']
        cmap = mpl.colors.ListedColormap(colors)
        print(file, val_ind)
        #ipdb.set_trace()
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        plt.subplot(1, 2, 1)
        plt.title('Prediction')
        plt.imshow(vis_mask, cmap=cmap)
        plt.axis('off')

        # plt.subplot(1, 3, 2)
        # plt.title('GT')
        # plt.imshow(vis_gt, cmap="gray")
        # plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Raw image')
        plt.imshow(vis_image[0], cmap="gray")
        plt.axis('off')

        saveRoot = os.path.join('OSVOS/data/Output/Unlabel/', self.model_name, self.split, file, 'Prediction/')
        if not os.path.exists(saveRoot):
            os.makedirs(saveRoot)
        plt.savefig(saveRoot + '/' +  str(val_ind) + '.png')



    def evalution_parent_epoch(self, epoch, file):
        print('*' * 30, 'Start evalute file: ' , file, 'here!', '*'*30)

        loss_epoch = []
        iou_epoch = []
        acc_epoch = []
        dice_coeff_epoch = []
        dice_coeff_all_epoch = []
        bin_mask_file = []
        labels_file = []

        num_iter = len(self.val_parent_loader)
        val_ind = 1
        with torch.no_grad():

            for batch_idx, sample in enumerate(self.val_parent_loader):
                start = time.time()
                img = sample['image']
                img = img.float().to(self.device)
                loss_batch = 0
                outputs = self.model(img)

                te = time.time() - start
                output = outputs[-1]
                bin_mask = torch.argmax(torch.softmax(output, 1), dim=1)


                if batch_idx == 0:
                    bin_mask_file = bin_mask
                else:

                    bin_mask_file = torch.tensor(np.vstack((bin_mask_file, bin_mask)))

                # if batch_idx == 0:
                #     labels_file = labels
                # else:
                #     labels_file = torch.tensor(np.vstack((labels_file, labels)))

                #Todo
                # iou_all, iou_mean, acc, dice_coeff = evaluate_iou(bin_mask, labels, self.all_label)
                #ipdb.set_trace()

                for img_idx in range(img.shape[0]):
                    vis_image = img[img_idx]
                    vis_mask = bin_mask[img_idx].data.cpu().numpy()
                    self.vis_plot(vis_mask, vis_image, file, val_ind)
                    val_ind += 1
                    #vis_gt = labels[img_idx].data.cpu().numpy() * 50
                    #vis_val = np.concatenate((vis_gt, vis_mask), axis=0)
                    # vis_val = np.concatenate((vis_gt, vis_mask, vis_image), axis=1)

                    #self.vis.show_feature_maps(features=vis_val, datatype='Validation sample')
                    #self.vis.show_img(img=vis_image, datatype='Validation Raw Image')
                    # if np.mod(val_ind, 10) == 0:
                    #     self.vis_plot(vis_mask, vis_image, file, val_ind)
                    #     val_ind += 1

                #loss_epoch.append(loss_batch.data.cpu().numpy())



        #loss_epoch = np.array(loss_epoch)
        #loss = loss_epoch.mean()

        import scipy.io
        saveRoot = os.path.join('OSVOS/data/Output/Unlabel/', self.model_name, self.split, file)
        if not os.path.exists(saveRoot):
            os.makedirs(saveRoot)
        # #ipdb.set_trace()
        scipy.io.savemat(saveRoot + '/' + file + '.mat', {'bin_mask_file': bin_mask_file.data.cpu().numpy()})
        #scipy.io.savemat('2.mat', {'bin_mask_file': bin_mask_file})




        # iou_all, iou_mean, acc, dice_coeff = evaluate_iou(bin_mask_file, labels_file, self.all_label)
        #
        # print(
        #     "{0}: [{1}][{2}/{3}] Time {batch_time:.3f}  IOU {iou:.4f} Acc {acc:.4f} dice {dice_coeff:.4f}".format(
        #         'Validation', epoch, batch_idx, num_iter, batch_time=te, iou=iou_mean,
        #         acc=acc, dice_coeff=dice_coeff.mean()))
        # if self.all_label:
        #     print("IOU: {:.4f} {:.4f} {:.4f} {:.4f}".format(iou_all[0], iou_all[1], iou_all[2], iou_all[3]))
        #     print("DICE: {:.4f} {:.4f} {:.4f} {:.4f}".format(dice_coeff[0], dice_coeff[1], dice_coeff[2],
        #                                                      dice_coeff[3]))
        #
        # else:
        #     print("IOU: {:.4f} {:.4f}".format(iou_all[0], iou_all[1]))
        #     print("DICE: {:.4f} {:.4f}".format(dice_coeff[0], dice_coeff[1]))

        #return acc, dice_coeff #each file



    def main(self):
        #print(self.paras)
        loss_val_files = []

        acc_val_files = []

        dice_mito = []
        dice_mem = []
        dice_nu = []
        dice_gr = []
        testset = ['784_5', '766_8', '842_17']
        valset =  ['783_5', '766_5', '842_12']
        #unlabelset = ['785_1', '785_2']
        unlabelset = os.listdir(os.path.join('OSVOS/data/images_new/'))
        print(unlabelset)

        set = unlabelset

        print('*' * 30, self.split, '*' * 30)

        for file in set:
            if file[0] == '.':
                continue
            self.init_dataloader(file)

            torch.cuda.synchronize()



            epoch = 0

            start = time.time()

            #Each file

            self.evalution_parent_epoch(epoch=epoch,file = file) #each file
            #ipdb.set_trace()

            te = time.time() - start
            #loss_val_files.append(loss_val_file)

        #     acc_val_files.append(acc_val_file)
        #
        #
        #     #ipdb.set_trace()
        #     dice_mito.append(dice_coeff_val_file[0])
        #     dice_mem.append(dice_coeff_val_file[1])
        #     dice_nu.append(dice_coeff_val_file[2])
        #     dice_gr.append(dice_coeff_val_file[3])
        #
        # #print('Mean: loss {loss:.4f} acc {acc:.4f} '.format(loss = np.array(loss_val_files).mean(), acc= np.array(acc_val_files).mean()) )
        #
        # #ipdb.set_trace()
        # print("DICE: {:.4f} {:.4f} {:.4f} {:.4f}".format(np.array(dice_mito).mean(), np.array(dice_mem).mean(),
        #                                                  np.array(dice_nu).mean(),
        #                                                  np.array(dice_gr).mean()))

            # ipdb.set_trace()
            # print(
            #     'file:%s, epoch_time:%.4f, loss_val:%.4f,acc_val:%.4f, dice_val:%.4f' % (
            #         file, te, loss_val_epoch,
            #         acc_val_epoch, dice_coeff_val_epoch))



            # if self.use_visdom:
            #     self.vis.show_curve(val_data=np.array(loss_val_epochs),
            #                         datatype='loss_epoch')
            #     self.vis.show_curve(val_data=np.array(acc_val_epochs),
            #                         datatype='acc_epoch')
            #     self.vis.show_curve(
            #                         val_data=np.array(dice_coeff_val_epochs),
            #                         datatype='dice_epoch')
            #
            #     self.vis.show_curve(
            #                         val_data=dice_coeff_val_epochs[0][0],
            #                         datatype='dice_epoch_mito')
            #
            #     self.vis.show_curve(
            #                         val_data=dice_coeff_val_epochs[0][1],
            #                         datatype='dice_epoch_mem')
            #
            #     self.vis.show_curve(
            #                         val_data=dice_coeff_val_epochs[0][2],
            #                         datatype='dice_epoch_nu')
            #
            #     self.vis.show_curve(
            #                         val_data=dice_coeff_val_epochs[0][3],
            #                         datatype='dice_epoch_gr')


if __name__ == '__main__':
    model = DavisParent(model_name='0719_resnet101_lr1e-4_adam_decay30_bs8')
    model.init_net()
    model.main()