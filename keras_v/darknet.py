import utils
from PIL import Image
import models
import tensorflow as tf
from tensorflow import keras
import time
import cv2
import matplotlib.pyplot as plt


tf.config.run_functions_eagerly(True)
config = utils.parse_config('../yolov3.cfg')
hyperparams = config.pop(0)
shape = (hyperparams['height'], hyperparams['width'], hyperparams['channels'])
model = models.YoloNet(config, shape)
model.load_weights('../yolov3.h5')
# keras.utils.plot_model(model, show_shapes=True)

cap = cv2.VideoCapture(0)
f, ax = plt.subplots(1, 1, figsize=(4, 4))
TIMEIT = []

def loop(i):

    # prepare the input of the network
    t1 = time.time()
    input_image, orig_image = utils.get_input(cap, shape[:2])
    TIMEIT.append(f'get input image: {time.time() - t1}')

    t1 = time.time()
    yolo_outputs = model.predict(input_image)  # [(1, 507, 85) (1, 2028, 85) (1, 8112, 85)]
    TIMEIT.append(f'make prediction: {time.time() - t1}')

    detections = keras.backend.squeeze(keras.layers.Concatenate(axis=1)(yolo_outputs), axis=0)

    t1 = time.time()
    detections = utils.preprocess_nms(detections)
    boxes = detections[..., :4]
    score = detections[..., 4]
    selected_indices = tf.image.non_max_suppression(boxes, score, max_output_size=-1, iou_threshold=0.4)
    detections = tf.gather(detections, selected_indices)
    TIMEIT.append(f'nms: {time.time() - t1}')



    print('non maximum suppression done... ')



