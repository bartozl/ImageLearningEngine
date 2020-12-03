import os
import requests
import numpy as np
import tensorflow as tf
from tensorflow import keras
from models import YoloLayer

URL_CFG = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
URL_WEIGHTS = "https://pjreddie.com/media/files/yolov3.weights"


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


def parse_config(dest_path):
    """
        Parse the official configuration file 'yolov3.cfg'.
        :return: a list of dictionaries that represent each layer configuration
        """
    config = []
    if "yolov3.cfg" not in os.listdir(dest_path.split('/')[0]):
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


def add_pad(img, reshape=None):
    h_o, w_o = img.shape[1:3]  # original shape
    q = max(h_o-w_o, w_o-h_o)//2
    if h_o > w_o:  # (left, right) padding
        pad = ((0, 0), (q, q))
    else:  # (top, bottom) padding
        pad = ((q, q), (0, 0))
    padded_img = keras.layers.ZeroPadding2D(padding=pad)(img)
    if reshape is None:
        return padded_img
    else:
        h_i, w_i = reshape
        return tf.keras.layers.experimental.preprocessing.Resizing(h_i, w_i, interpolation="nearest")(padded_img)


def get_input(img, input_shape):
    img = (keras.preprocessing.image.img_to_array(img)/255)[np.newaxis, ...]  # shape=[1, H, W, C]
    original_shape = img.shape[1:3]
    img = add_pad(img, reshape=input_shape)
    return img, original_shape


def preprocess_nms(detections, conf_thresh=0.8):
    detections = detections[detections[:, 4] > conf_thresh]
    print(detections.shape)
    detections_new = (keras.backend.zeros((detections.shape[0], 7)))

    max_class_idx = keras.backend.argmax(detections[:, 5:])
    max_class_idx = keras.backend.cast(max_class_idx, dtype='float32')
    max_class_score = keras.backend.max(detections[:, 5:], axis=-1)
    print(max_class_score)

    detections_new[:, 0].assign(detections[:, 0] - detections[:, 2] // 2)
    detections_new[:, 1].assign(detections[:, 1] - detections[:, 3] // 2)
    detections_new[:, 2].assign(detections[:, 0] + detections[:, 2] // 2)
    detections_new[:, 3].assign(detections[:, 1] + detections[:, 3] // 2)
    detections_new[:, 4].assign(detections[:, 4] * max_class_score)
    detections_new[:, 5].assign(max_class_idx)
    detections_new[:, 6].assign(max_class_score)
    # print(detections_new[:, 4])
    return detections_new
