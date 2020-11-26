import utils
import os
from models import Darknet
import torch
from torchvision.ops import nms
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from matplotlib.animation import FuncAnimation

# input_image, orig_shape = utils.get_input('./samples/dog.jpg', resize=shape)

'''
ax1 = plt.subplot()
img = ax1.imshow(input_image)

ani = FuncAnimation(plt.gcf(), update, interval=10)
plt.show()

print(orig_shape)
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create label dictionary
labels = utils.get_labels()

# Create the Yolov3 model from the config file
config_path = f"{os.getcwd()}/yolov3.cfg"
config = utils.parse_cfg(config_path)
hyperparams = config.pop(0)
shape = (hyperparams['width'], hyperparams['height'])
module_list = utils.create_modules(config, hyperparams)
weights = utils.get_pretrained_weights(config, module_list)
model = Darknet(config, module_list, device)
model.eval()

cap = cv2.VideoCapture(0)
ax = plt.subplot()


def loop(i):

    # prepare the input of the network
    input_image, orig_image = utils.get_input(cap, shape)

    # make a prediction --> pred.shape = [n_bboxes, c_x + c_y + w + h + obj_score + classes] --> [10647, 85]
    pred = model(input_image).squeeze(0)

    # Non maximum suppression
    detection = utils.preprocess_nms(pred)
    selected_bboxes = nms(detection[:, :4], detection[:, 4], iou_threshold=0.4)
    detection = detection[selected_bboxes]

    # Rescale selected bboxes to the original shape
    detection[:, :4] = utils.scale_bboxes(detection[:, :4],
                                          orig_shape=orig_image.shape[1:],
                                          curr_shape=input_image.shape[2:])
    ax.clear()
    ax.imshow(orig_image.permute(1, 2, 0))
    utils.create_output_image(ax, detection, labels)
    return[ax]


ani = FuncAnimation(plt.gcf(), loop, interval=1, blit=True)
plt.show()
