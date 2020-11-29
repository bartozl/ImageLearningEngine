import requests
import os
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
from models import EmptyLayer, YoloLayer
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import matplotlib.patches as patches

URL_CFG = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
URL_WEIGHTS = "https://pjreddie.com/media/files/yolov3.weights"


def get_pretrained_weights(version, config, module_list):
    dest_path = f'{os.getcwd()}/{version}.weights'
    if not os.path.exists(dest_path):
        print(f"downloading {URL_WEIGHTS}")
        r = requests.get(URL_CFG, allow_redirects=True)
        with open(dest_path, 'wb') as f:
            f.write(r.content)
        print(f"saved in {dest_path}")

    with open(dest_path, 'rb') as f:
        header = np.fromfile(f, dtype=np.int32, count=5)
        pret_weights = np.fromfile(f, dtype=np.float32)

    idx = 0
    for o, (block, mod) in enumerate(zip(config, module_list)):
        if block['layer_type'] == 'convolutional':
            conv = mod[0]
            if 'batch_normalize' in block:
                bn = mod[1]

                bn_b = torch.from_numpy(pret_weights[idx:(idx + bn.bias.numel())]).view(bn.bias.shape)
                idx += bn.bias.numel()

                bn_w = torch.from_numpy(pret_weights[idx:(idx + bn.weight.numel())]).view(bn.weight.shape)
                idx += bn.weight.numel()

                bn_rm = torch.from_numpy(pret_weights[idx:(idx + bn.running_mean.numel())]).view(bn.running_mean.shape)
                idx += bn.running_mean.numel()

                bn_rv = torch.from_numpy(pret_weights[idx:(idx + bn.running_var.numel())]).view(bn.running_var.shape)
                idx += bn.running_var.numel()

                bn.bias.data.copy_(bn_b)
                bn.weight.data.copy_(bn_w)
                bn.running_mean.data.copy_(bn_rm)
                bn.running_var.data.copy_(bn_rv)
            else:
                bias = torch.from_numpy(pret_weights[idx:(idx + conv.bias.numel())]).view(conv.bias.shape)
                conv.bias.data.copy_(bias)
                idx += conv.bias.numel()
            weights = torch.from_numpy(pret_weights[idx:(idx + conv.weight.numel())]).view(conv.weight.shape)
            conv.weight.data.copy_(weights)
            idx += conv.weight.numel()


def download_cfg(dest_path):
    """
    Download the yolov3 configuration file
    :return: None
    """
    print(f"downloading {URL_CFG}")
    r = requests.get(URL_CFG, allow_redirects=True)
    with open(dest_path, 'wb') as f:
        f.write(r.content)
    print(f"saved in {dest_path}")


def rescale_anchors(anchors, stride):
    """
    The anchors are computed w.r.t. the input image (eg 416*416) then
    we need to rescale them in the current feature map size (e.g. 13 * 13)
    """
    num_anchors = len(anchors)
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    anchors = FloatTensor(np.asarray(anchors) / stride)
    anchors_w = anchors[..., 0].view(1, num_anchors, 1, 1)
    anchors_h = anchors[..., 1].view(1, num_anchors, 1, 1)
    return anchors_w, anchors_h


def create_grid(size_x):
    c_x = torch.ones((1, 1, size_x, size_x)) * torch.arange(size_x)
    c_y = (torch.ones((1, 1, size_x, size_x)) * torch.arange(size_x)).permute(0, 1, 3, 2)
    return c_x, c_y


def center2diag(bboxes):
    c_x, c_y, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    bboxes[..., 0] = c_x - w / 2  # x_1
    bboxes[..., 1] = c_y - h / 2  # y_1
    bboxes[..., 2] = c_x + w / 2  # x_2
    bboxes[..., 3] = c_y + h / 2  # y_2
    return bboxes


def get_pad(shape):
    h, w = shape
    pad = [0, 0, 0, 0]  # left, right, top, bottom
    q = [abs(h - w) // 2] * 2
    if h > w:
        pad[:2] = q
    elif h < w:
        pad[2:] = q
    return pad


def add_pad(img, resize=None):
    pad = get_pad(img.shape[1:])
    img = F.pad(img, pad=pad, mode='constant', value=0).unsqueeze(0)
    if resize is not None:
        img = F.interpolate(img, size=resize, mode="nearest")
    return img


def scale_bboxes(bboxes, orig_shape, curr_shape):
    curr_shape = torch.tensor(curr_shape)
    orig_shape = torch.tensor(orig_shape)
    scale = torch.min(curr_shape / orig_shape)

    # compute the pad ratio in the (resized) current image
    scaled_pad = torch.tensor(get_pad(orig_shape)) * scale  # [left, right, top, bottom]
    scaled_pad = [torch.sum(scaled_pad[2:]), torch.sum(scaled_pad[:2])]  # [width, height]

    # compute the measure of the unpadded current image
    curr_unpadded = [curr_shape[0] - scaled_pad[1], curr_shape[1] - scaled_pad[0]]

    # rescale the current bboxes according to the new size:

    for i in range(4):
        idx = (i + 1) % 2  # 0 = width ; 1 = height
        bboxes[:, i] = ((bboxes[:, i] - scaled_pad[idx] // 2) / curr_unpadded[idx]) * orig_shape[i % 2]

    return bboxes


def get_input(cap, resize=None):
    ret, frame = cap.read()
    img_orig = transforms.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_curr = add_pad(img_orig, resize=resize)
    return img_curr, img_orig


def get_labels(dir_path='labels.names'):
    with open(dir_path) as f:
        names = f.readlines()
    labels = {i: names[i].rstrip() for i in range(len(names))}
    return labels


def preprocess_nms(pred, conf_thresh=0.8):
    # pred.shape = [10647, 85] = [n_boxes, c_x | c_y | w | h | obj_score | classes(80)]
    pred = pred[pred[:, 4] > conf_thresh]
    if pred.shape[0] == 0:
        return None
    pred_new = torch.empty(pred.shape[0], 7)  # [n_boxes, c_x | c_y | w | h | score | class_idx | class_score]
    # transform bboxes coordinates: xywh --> x1y1x2y2
    c_x, c_y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1, y1 = c_x - w / 2, c_y - h / 2
    x2, y2 = c_x + w / 2, c_y + h / 2
    bboxes = torch.cat((x1.unsqueeze(-1), y1.unsqueeze(-1), x2.unsqueeze(-1), y2.unsqueeze(-1)), dim=-1)
    pred_new[:, :4] = bboxes

    # score = obj_score * max(class_score)
    max_class, idx_class = torch.max(pred[:, 5:], dim=-1)
    pred_new[:, 4] = pred[:, 4] * max_class
    pred_new[:, 5] = idx_class
    pred_new[:, 6] = max_class
    return pred_new


def cast_type(attr, val):
    """
    :param attr: attribute of the block
    :param val: (str) value corresponding to that attribute
    :return: val correctly casted
    """
    # integer attributes
    if attr in ['batch', 'subdivisions', 'width', 'height', 'channels', 'angle', 'burn_in',
                'max_batches', 'batch_normalize', 'filters', 'size', 'stride', 'pad',
                'from', 'classes', 'num', 'truth_thresh', 'random']:
        val = int(val)
    # float attributes
    elif attr in ['momentum', 'decay', 'saturation', 'exposure',
                  'hue', 'learning_rate', 'jitter', 'ignore_thresh']:
        val = float(val)
    # string attributes
    elif attr in ['policy', 'activation']:
        val = str(val)
    # list of int attributes
    elif attr in ['layers', 'mask', 'steps']:
        val = [int(v) for v in val.split(',')] if len(val) >= 2 else int(val)
    # list of float
    elif attr in ['scales']:
        val = [float(v) for v in val.split(',')] if len(val) >= 2 else float(val)
    # list of list attributes
    elif attr == 'anchors':
        val = (val[1:].split(',  '))
        val2 = []
        for x in val:
            val2.append([int(v) for v in x.split(',')])
        val = val2
    # an attribute has not been correctly parsed
    else:
        print('not classified', attr, val)
        exit(0)
    return attr, val


def parse_cfg(dest_path):
    """
    Parse the official configuration file 'yolov3.cfg'.
    :return: a list of dictionaries that represent each layer configuration
    """
    config = []
    if "yolov3.cfg" not in os.listdir(os.getcwd()):
        download_cfg(dest_path)
    with open(dest_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i][0] == '[':  # possible layers: ['convolutional', 'route', 'yolo', shortcut', 'upsample']
                layer_type = lines[i].rstrip()[1:-1]  # extract layer type info
                block = {"layer_type": layer_type}  # new dictionary for each new block
                count = 1  # count the line for each block
                while i + count < len(lines) and lines[i + count][0] != '[':  # parse new line until the block ends
                    if lines[i + count][0] not in ['#', '\n']:  # skip commented or white lines
                        attr, val = lines[i + count].rstrip().split('=')
                        attr, val = cast_type(attr.rstrip(), val)
                        block[attr] = val
                    count += 1
                i += count
                config.append(block)

    return config


def create_modules(config, hyperparams):
    module_list = nn.ModuleList()
    prev_filters = hyperparams['channels']  # RGB images --> first layer has 3 filters
    output_filters = [prev_filters]  # store the output filters for each layer
    for idx, block in enumerate(config):
        mod = nn.Sequential()
        if block['layer_type'] == 'convolutional':
            filters = block['filters']
            pad = (block['size'] - 1) // 2 if block['pad'] else 0
            conv = nn.Conv2d(in_channels=prev_filters,
                             out_channels=filters,
                             kernel_size=block['size'],
                             stride=block['stride'],
                             padding=pad,
                             bias=not int(block.get('batch_normalize', 0)))
            mod.add_module(f"conv_{idx}", conv)

            if 'batch_normalize' in block:
                mod.add_module(f"batch_norm_{idx}", nn.BatchNorm2d(filters, momentum=0.9))

            if block["activation"] == 'leaky':
                mod.add_module(f"leaky_{idx}", nn.LeakyReLU(0.1))

        elif block['layer_type'] == 'upsample':
            mod.add_module(f'upsample_{idx}', nn.Upsample(scale_factor=block["stride"], mode='bilinear'))

        elif block['layer_type'] == 'maxpool':
            if block['size'] == 2 and block['stride'] == 1:
                mod.add_module(f"_debug_padding_{idx}", nn.ZeroPad2d((0, 1, 0, 1)))
            mod.add_module(f'maxpool_{idx}', nn.MaxPool2d(kernel_size=block['size'], stride=block['stride'],
                                                          padding=int(block['size'] - 1) // 2))

        elif block['layer_type'] == 'route':
            filters: int = sum([output_filters[1:][i] for i in list(block['layers'])])
            mod.add_module(f'route_{idx}', EmptyLayer())  # its just a placeholder.

        elif block['layer_type'] == 'shortcut':
            mod.add_module(f'shortcut_{idx}', EmptyLayer())

        elif block['layer_type'] == 'yolo':
            img_dim = (hyperparams['width'], hyperparams['height'])
            anchors = [block['anchors'][i] for i in block['mask']]
            mod.add_module(f'yolo_{idx}', YoloLayer(anchors, block['classes'], img_dim))
        else:
            print("Unknown layer type")
            exit(0)

        module_list.append(mod)
        prev_filters = filters
        output_filters.append(filters)

    return module_list


def create_output_image(ax, detection, labels):
    for x1, y1, x2, y2, score, class_label, class_score in detection:
        box_w = x2 - x1
        box_h = y2 - y1
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, alpha=0.8, edgecolor='red', facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        ax.text(x1 + 3,
                y1 + 3,
                s=f'{class_score:.4f} {labels[int(class_label)]}',
                horizontalalignment='left',
                verticalalignment='top',
                bbox=dict(facecolor='red', alpha=0.4, edgecolor='red', boxstyle='round,pad=0.2')
                )
