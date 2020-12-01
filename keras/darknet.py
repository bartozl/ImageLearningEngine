import utils
from PIL import Image
import models
import tensorflow as tf
from tensorflow import keras

config = utils.parse_config('../yolov3.cfg')
hyperparams = config.pop(0)

# initialize the model

# create input for the network
img = Image.open('../samples/dog.jpg')
input_shape = hyperparams['height'], hyperparams['width']
input_img, original_shape = utils.prepare_input(img, input_shape)

model = models.YoloNet(config, input_img.shape[1:])
model.load_weights('../yolov3.h5')
keras.utils.plot_model(model, show_shapes=True)
# yolo_outputs = model.predict(input_img)  # [(1, 507, 85) (1, 2028, 85) (1, 8112, 85)]
# yolo_outputs = keras.layers.Concatenate(axis=1)(yolo_outputs)

# tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold)
