import os.path as osp
from torch.utils.data import Dataset
from scipy.misc import imresize
import os
import sys
sys.path.append(os.path.abspath('./'))
import cv2
import numpy as np
from torch.utils import data
from torchvision.transforms import Compose
import transforms as tf
import ipdb

class DAVIS2017Train(Dataset):

    def __init__(self, split, db_root_dir='/home/youngfly/Disk/data/DAVIS', transforms=None):
        self.split = split
        self.db_root_dir = db_root_dir
        self.transforms = transforms

        if self.split == 'train':
            fname = 'train'
        elif self.split == 'val':
            fname = 'val'
        else:
            raise Exception('Only support train and val!')

        # Initialize the original DAVIS splits for training the parent network
        with open(os.path.join(db_root_dir, 'ImageSets/2017', fname + '.txt')) as f:
            seqs = f.readlines()
            img_list = []
            labels = []
            for seq in seqs:
                if seq == '\n':
                    continue
                images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip())))
                images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images))
                img_list.extend(images_path)
                lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip())))
                lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
                labels.extend(lab_path)

        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels

        print('Done initializing ' + fname + ' Dataset')

    def __getitem__(self, index):

        ipdb.set_trace()
        # load image
        image = cv2.imread(os.path.join(self.db_root_dir, self.img_list[index]))

        label = cv2.imread(os.path.join(self.db_root_dir, self.labels[index]), 0)

        h, w, c = image.shape

        if h != 480 or w != 854:

            image = cv2.resize(image, (854, 480), cv2.INTER_NEAREST)
            label = cv2.resize(label, (854, 480), cv2.INTER_NEAREST)


        label = label / 255.0
        loc = np.where((label>0))
        label = np.zeros_like(label)
        label[loc[0], loc[1]] = 1

        sample = {'image': image, 'label': label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.img_list)


class DAVIS2017Val(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='./dataloaders/DAVIS/',
                 transform=None,
                 seq_name=None,
                 is_multi_object=False):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """

        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        # self.meanval = meanval
        self.seq_name = seq_name
        self.is_multi_object = is_multi_object

        self.train = train
        fname = 'val_seqs'

        # Initialize the per sequence images for online training
        names_img = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', str(seq_name))))
        img_list = list(map(lambda x: os.path.join('JPEGImages/480p/', str(seq_name), x), names_img))
        name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', str(seq_name))))
        labels = [os.path.join('Annotations/480p/', str(seq_name), name_label[i]) for i in range(len(name_label))]
        # labels.extend([None]*(len(names_img)-1))
        if self.train:
            img_list = [img_list[0]]
            labels = [labels[0]]
        else:
            img_list = img_list[1:]
            labels = labels[1:]

        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels

        print('Done initializing ' + fname + ' Dataset')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img, gt = self.make_img_gt_pair(idx)

        sample = {'image': img, 'gt': gt}

        if self.seq_name is not None:
            fname = os.path.join(self.seq_name, "%05d" % idx)
            sample['fname'] = fname

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        gt = None
        label = None
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        if self.labels[idx] is not None:
            label = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), 0)
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)

        if self.inputRes is not None:
            print (self.inputRes)
            img = imresize(img, self.inputRes)
            if self.labels[idx] is not None:
                label = imresize(label, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        if self.labels[idx] is not None:
            gt = np.array(label, dtype=np.float32)

            if self.is_multi_object:
                h, w = gt.shape
                gt_multi = {}
                gt_idx_set = list(set(gt.reshape(h * w, -1).squeeze().tolist()))[1:]
                for idx_value in gt_idx_set:
                    id = np.where((gt == idx_value))
                    gt_multi_sig = np.zeros_like(gt)
                    gt_multi_sig[id[0], id[1]] = 1
                    gt_multi[idx_value] = gt_multi_sig
                gt = gt_multi
            else:
                gt = gt / np.max([gt.max(), 1e-8])

        return img, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])

if __name__ == '__main__':

    train_transform = Compose([tf.RandomHorizontalFlip(),
                               tf.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                               tf.ToTensor(),
                               tf.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])

    val_transform = Compose([tf.ToTensor(),
                             tf.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])])
    num_workers = 4
    batch_size = 9


    db_root_dir = '/home/youngfly/Disk/data/DAVIS'
    train_dataset = DAVIS2017Train(split='train', db_root_dir=db_root_dir,
                                   transforms=train_transform)
    train_parent_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)

    val_dataset = DAVIS2017Train(split='val', db_root_dir=db_root_dir,
                                 transforms=val_transform)
    val_parent_loader = data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)

    for idx, sample in enumerate(val_parent_loader):

        img, label = sample['image'], sample['label']
        print (idx, img.size(), label.size())