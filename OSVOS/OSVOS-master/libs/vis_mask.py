import cv2
import pickle
import os
import numpy as np
import os.path as osp
import random
import colorsys


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / float(N), 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    colors = [tuple(int(c * 255) for c in color) for color in colors]
    return colors


def generate_mask(seq_name):

    """
    Args:
        seq_name: the corresponding image name
        results: dictionary list
        data: dictionary, 'image', 'mask', for all object in training image
    """

    dir_path = '/home/youngfly/Disk/Projects/pytorch/OSVOS-PyTorch/'
    db_root_dir = 'dataloaders/DAVIS/'
    result_root_dir = 'models/results/'
    names_img = np.sort(os.listdir(os.path.join(dir_path, db_root_dir, 'JPEGImages/480p/', str(seq_name))))[1:]
    img_list = list(map(lambda x: os.path.join(dir_path, db_root_dir, 'JPEGImages/480p/', str(seq_name), x), names_img))
    name_label = np.sort(os.listdir(os.path.join(dir_path, result_root_dir, str(seq_name))))
    labels = [os.path.join(dir_path, result_root_dir, str(seq_name), name_label[i]) for i in range(len(name_label))]
    results = []


    for img_name in img_list:

        data = {}

        mask_list = []
        baseName = os.path.basename(img_name)
        baseName = '%.5d.jpg'% (int(baseName[0:5]) - 1)
        img = cv2.imread(img_name)

        img = np.array(img, np.float64)

        for mask_path in labels:
            # print seq_name, mask_path
            mask_name = osp.join(mask_path, baseName).replace('jpg', 'png')
            maskProb = cv2.imread(mask_name, 0)

            if maskProb is not None:
                mask = np.zeros_like(img)[:,:,0]
                one_idx = np.where((maskProb>=0.5))
                mask[one_idx[0], one_idx[1]]  = 1
                mask = mask[:,:, None]
                mask_list.append(mask)
            else:
                mask = np.zeros_like(img)[:,:, 0][:,:, None]
                mask_list.append(mask)
        mask = np.concatenate(tuple(mask_list), axis=2)
        print (img.shape, mask.shape)
        data['image'] = img
        data['masks'] = mask
        results.append(data)

    return results


def vis_mask():

    seq_names = ['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel',
                 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog',
                 'dogs-jump', 'drift-chicane', 'drift-straight', 'goat', 'gold-fish',
                 'horsejump-high', 'india', 'judo', 'kite-surf', 'lab-coat', 'libby',
                 'loading', 'mbike-trick', 'motocross-jump', 'paragliding-launch', 'parkour',
                 'breakdance', 'scooter-black', 'shooting', 'soapbox']

    h, w = 480, 854
    video_writer = cv2.VideoWriter('osvos.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (w, h))

    for seq_name in seq_names:

        results = generate_mask(seq_name)
        num_instance = results[0]['masks'].shape[2]

        for i in range(num_instance):
            color = random_colors(10)[0]
            for data in results:
                img = data['image']
                mask = data['masks'][:, :, i]
                masked_image = img.copy()
                mask_color = np.stack([np.ones_like(mask) * c for c in color],
                                      axis=2)
                masked_image[mask == 1] = masked_image[mask == 1] * 0.5 + mask_color[mask == 1] * 0.5
                masked_image = cv2.resize(masked_image, (w, h)).astype(np.uint8)
                video_writer.write(masked_image)

    video_writer.release()


if __name__ == '__main__':

    vis_mask()