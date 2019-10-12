import os
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np


def parse(pathname=None):

    with open('./logs/' + pathname) as f:

        document = f.readlines()

    train_loss = []
    val_loss = []
    train_iou = []
    val_iou = []

    for idx, line in enumerate(document):
        if idx > 5:
            line = line.strip().split(',')
            if len(line) > 5:
                train_loss.append(float(line[2].split(':')[1]))
                val_loss.append(float(line[3].split(':')[1]))
                train_iou.append(float(line[4].split(':')[1]))
                val_iou.append(float(line[5].split(':')[1]))

    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    train_iou = np.array(train_iou)
    val_iou = np.array(val_iou)

    ## visualize

    best_val_iou = val_iou.max()
    best_train_iou = train_iou.max()
    epoch_train = list(train_iou).index(best_train_iou)
    epoch_val = list(val_iou).index(best_val_iou)

    print (pathname)
    print('training:', epoch_train, best_train_iou)
    print('val:', epoch_val, best_val_iou)


    plot_figure(train_iou=train_iou, val_iou=val_iou, train_loss=train_loss, val_loss=val_loss, pathname=pathname)
    return train_loss, val_loss, train_iou, val_iou


def plot_figure(train_loss, val_loss, train_iou, val_iou, pathname):

    plt.figure()
    plt.subplot(121)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'val'])

    plt.subplot(122)
    plt.plot(train_iou)
    plt.plot(val_iou)
    plt.xlabel('epoch')
    plt.ylabel('IOU')
    plt.legend(['train', 'val'])
    plt.savefig('./logs/' + pathname + '.png')
    plt.close()


if __name__ == '__main__':
    # parse(pathname='vis_osvos_adam_residual.out.node03')
    # parse(pathname='vis_osvos_adam.out.node06')

    # parse(pathname='vis_osvos.out.node04')
    parse(pathname='vis_osvos_full_residual17.out.node03')
