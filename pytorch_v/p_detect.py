import utils
import os
from models import Darknet
import torch
from torchvision.ops import nms
import matplotlib.pyplot as plt
import cv2
from matplotlib.animation import FuncAnimation
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-tiny', action='store_true', default=False)
parser.add_argument('-camera', action='store_false', default=True)
parser.add_argument('--conf_thresh', type=float, default=0.8)
args = parser.parse_args()
version = 'yolov3' if not args.tiny else 'yolov3-tiny'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create label dictionary
labels = utils.get_labels()

# Create the Yolov3 model from the config file
config_path = f"{os.getcwd()}/{version}.cfg"
config = utils.parse_cfg()
hyperparams = config.pop(0)
shape = (hyperparams['width'], hyperparams['height'])
module_list = utils.create_modules(config, hyperparams)
weights = utils.get_pretrained_weights(version, config, module_list)
model = Darknet(config, module_list, device)
model.eval()

if args.camera:

    cap = cv2.VideoCapture(0)
    f, ax = plt.subplots(1, 1, figsize=(4, 4))

    def loop(i):

        # prepare the input of the network
        t1 = time.time()
        input_image, orig_image = utils.get_input(cap, shape)
        print(f'get input image: {time.time() - t1}')
        # make a prediction --> pred.shape = [n_bboxes, c_x + c_y + w + h + obj_score + classes] --> [10647, 85]
        t1 = time.time()
        with torch.no_grad():
            pred = model(input_image).squeeze(0)
        print(f'make prediction: {time.time() - t1}')

        # Non maximum suppression
        t1 = time.time()
        detection = utils.preprocess_nms(pred, conf_thresh=args.conf_thresh)
        if detection is None:
            return [ax]
        selected_bboxes = nms(detection[:, :4], detection[:, 4], iou_threshold=0.4)
        detection = detection[selected_bboxes]
        print(f'nms: {time.time() - t1}')

        # Rescale selected bboxes to the original shape
        detection[:, :4] = utils.scale_bboxes(detection[:, :4],
                                              orig_shape=orig_image.shape[1:],
                                              curr_shape=input_image.shape[2:])
        ax.clear()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        ax.imshow(orig_image.permute(1, 2, 0))
        t1 = time.time()
        utils.create_output_image(ax, detection, labels)
        print(f'create output: {time.time() - t1}')
        return [ax]


    ani = FuncAnimation(plt.gcf(), loop, interval=1, blit=True)
    plt.show()

'''
input_image, orig_image = utils.get_dummy_input(resize=shape)
pred = model(input_image).squeeze(0)
detection = utils.preprocess_nms(pred, conf_thresh=args.conf_thresh)
selected_bboxes = nms(detection[:, :4], detection[:, 4], iou_threshold=0.4)
'''