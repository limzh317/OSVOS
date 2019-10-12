import os
import pickle
import numpy as np
from collections import OrderedDict
# from dice_loss import dice_coeff
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def check_parallel(encoder_dict, decoder_dict):
    # check if the model was trained using multiple gpus
    trained_parallel = False
    for k, v in encoder_dict.items():
        if k[:7] == "module.":
            trained_parallel = True
        break
    if trained_parallel:
        # create new OrderedDict that does not contain "module."
        new_encoder_state_dict = OrderedDict()
        new_decoder_state_dict = OrderedDict()
        for k, v in encoder_dict.items():
            name = k[7:]  # remove "module."
            new_encoder_state_dict[name] = v
        for k, v in decoder_dict.items():
            name = k[7:]  # remove "module."
            new_decoder_state_dict[name] = v
        encoder_dict = new_encoder_state_dict
        decoder_dict = new_decoder_state_dict

    return encoder_dict, decoder_dict


def get_base_params(base_model, model):
    b = []
    if 'vgg' in base_model:
        b.append(model.base.features)
    else:
        b.append(model.base.conv1)
        b.append(model.base.bn1)
        b.append(model.base.layer1)
        b.append(model.base.layer2)
        b.append(model.base.layer3)
        b.append(model.base.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_skip_params(model):
    b = []
    b.append(model.sk1.parameters())
    b.append(model.sk2.parameters())
    b.append(model.sk3.parameters())
    b.append(model.sk4.parameters())
    b.append(model.sk5.parameters())
    b.append(model.bn1.parameters())
    b.append(model.bn2.parameters())
    b.append(model.bn3.parameters())
    b.append(model.bn4.parameters())
    b.append(model.bn5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


def get_skip_dims(model_name):
    if model_name == 'resnet50' or model_name == 'resnet101':
        skip_dims_in = [2048, 1024, 512, 256, 64]
    elif model_name == 'resnet34':
        skip_dims_in = [512, 256, 128, 64, 64]
    elif model_name == 'vgg16':
        skip_dims_in = [512, 512, 256, 128, 64]

    return skip_dims_in


def center_crop(x, height=480, width=854):
    crop_h = torch.FloatTensor([x.size()[2]]).sub(height).div(-2)
    crop_w = torch.FloatTensor([x.size()[3]]).sub(width).div(-2)

    return F.pad(x, [
        crop_w.ceil().int()[0], crop_w.floor().int()[0],
        crop_h.ceil().int()[0], crop_h.floor().int()[0],
    ])


def get_optimizer(optim_name, lr, parameters, weight_decay=0, momentum=0.9):
    if optim_name == 'sgd':
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, parameters),
                              lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optim_name == 'adam':
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=lr, weight_decay=weight_decay)
    elif optim_name == 'rmsprop':
        opt = torch.optim.RMSprop(filter(lambda p: p.requires_grad, parameters), lr=lr, weight_decay=weight_decay)
    return opt


def save_checkpoint(checkpoint_dir, encoder, decoder, enc_opt, dec_opt, paras):
    torch.save(encoder.state_dict(), os.path.join(checkpoint_dir, 'encoder.pt'))
    torch.save(decoder.state_dict(), os.path.join(checkpoint_dir, 'decoder.pt'))
    torch.save(enc_opt.state_dict(), os.path.join(checkpoint_dir, 'enc_opt.pt'))
    torch.save(dec_opt.state_dict(), os.path.join(checkpoint_dir, 'dec_opt.pt'))

    # save parameters for future use
    pickle.dump(paras, open(os.path.join(checkpoint_dir, 'paras.pkl'), 'wb'))


def save_checkpoint_lite(checkpoint_dir, model, optim, paras):
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model.pt'))
    torch.save(optim.state_dict(), os.path.join(checkpoint_dir, 'optim.pt'))
    # save parameters for future use
    pickle.dump(paras, open(os.path.join(checkpoint_dir, 'paras.pkl'), 'wb'))


def load_checkpoint(checkpoint_dir):
    encoder_dict = torch.load(os.path.join(checkpoint_dir, 'encoder.pt'))
    decoder_dict = torch.load(os.path.join(checkpoint_dir, 'decoder.pt'))
    enc_opt_dict = torch.load(os.path.join(checkpoint_dir, 'enc_opt.pt'))
    dec_opt_dict = torch.load(os.path.join(checkpoint_dir, 'dec_opt.pt'))

    # load parameters
    paras = pickle.load(open(os.path.join(checkpoint_dir, 'paras.pkl'), 'rb'))

    return encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, paras


def load_checkpoint_lite(checkpoint_dir):
    state_dict = torch.load(os.path.join(checkpoint_dir, 'model.pt'))
    optim_dict = torch.load(os.path.join(checkpoint_dir, 'optim.pt'))

    return state_dict, optim_dict


def init_visdom(viz):
    mviz_pred = viz.image(np.zeros((480, 854)), opts=dict(title='Pred mask'))
    mviz_true = viz.image(np.zeros((480, 854)), opts=dict(title='True mask'))

    lot = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,)).cpu(),
        opts=dict(
            xlabel='Iteration',
            ylabel='Loss',
            title='Running Loss',
            legend=['loss']
        )
    )

    elot = {}
    # epoch iou
    elot['iou'] = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 2)).cpu(),
        opts=dict(
            xlabel='Epoch',
            ylabel='IoU',
            title='IoU',
            legend=['train', 'val']
        )
    )

    # epoch loss
    elot['loss'] = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 2)).cpu(),
        opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title='Total Loss',
            legend=['train', 'val']
        )
    )

    # text
    text = viz.text(text='start visdom')

    return lot, elot, mviz_pred, mviz_true, text


def batch_to_var(sample, mode='train'):
    """
    Turns the output of DataLoader into data and ground truth to be fed
    during training
    """

    for key, value in sample.items():
        sample[key] = Variable(sample[key], volatile=mode == 'val').cuda()
    return sample


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    # imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass - 1, range=(1, numClass - 1))

    # Compute area union:
    # We dont need background
    (area_pred, _) = np.histogram(imPred, bins=numClass - 1, range=(1, numClass - 1))  # 1, 2, 3, 4
    (area_lab, _) = np.histogram(imLab, bins=numClass - 1, range=(1, numClass - 1))
    area_union = area_pred + area_lab - area_intersection

    IOU = area_intersection / area_union
    dice_coeff = 2 * area_intersection / (area_pred + area_lab)

    return IOU, dice_coeff


def dice_coeff_f(input, target):
    inter = torch.dot(input.view(-1), target.view(-1)) + 0.0001
    union = torch.sum(input) + torch.sum(target) + 0.0001

    t = 2 * inter.float() / union.float()
    return t


def accuracy(preds, label):
    valid = (label > 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (float(valid_sum) + 1e-10)
    return acc


def evaluate_iou(x, y, all_label):
    """
    :param x: tensor (B, 1, H, W) {0, 1} float
    :param y: tensor (B, 1, H, W) {0, 1} float
    :return: IOU: float
    """
    # ipdb.set_trace()

    batch_size, h, w = x.size()

    x = x.view(batch_size, -1).long()
    y = y.view(batch_size, -1).long()
    acc = accuracy(x, y)
    if all_label:
        num_class = 5
    else:
        num_class = 3
    IOU, dice_coeff = intersectionAndUnion(x, y, num_class)
    # dice_coeff = dice_coeff_f(x, y)

    return IOU, IOU.mean(), acc, dice_coeff


def class_balanced_cross_entropy_loss(outputs, label, size_average=True, batch_average=True):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = label.float()

    batch_size = label.shape[0]

    for bid in range(batch_size):
        label = labels[bid]
        output = outputs[bid]
        label_map_bak = (label == 0).float()
        label_map_one = (label == 1).float()
        label_map_two = (label == 2).float()
        label_map_thr = (label == 3).float()
        label_map_for = (label == 4).float()
        num_back = torch.sum(label_map_bak)
        num_one = torch.sum(label == 1)
        num_ = torch.sum(label == 2)
        num_one = torch.sum(label == 3)
        num_one = torch.sum(label == 4)

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

    loss_pos = torch.sum(-torch.mul(labels, loss_val))
    loss_neg = torch.sum(-torch.mul(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= int(np.prod(label.size()))
    elif batch_average:
        final_loss /= int(label.size()[0])

    return final_loss


def adjust_learning_rate(optimizer, init_lr, epoch, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = init_lr * (0.1 ** (epoch // step))
    if np.mod(epoch + 1, step) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
