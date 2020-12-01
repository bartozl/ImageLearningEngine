from tensorflow import keras
from tensorflow.keras import layers
from collections import defaultdict


class YoloLayer(layers.Layer):
    def __init__(self, anchors, classes, img_dim):
        super(YoloLayer, self).__init__()
        """
        The anchors are the pre-defined bounding boxes, computed with k-means algorithm.
        Instead of computing the bounding box from scratch, we adjust the dimension
        of this already know bounding boxes
        """
        self.anchors = anchors
        self.classes = classes
        self.img_dim = img_dim

    def call(self, x, training=False):
        """
        layer 82 --> x.shape = [batch, 255, 13, 13]
        layer 94 --> x.shape = [batch, 255, 26, 26]
        layer 106 --> x.shape = [batch, 255, 52, 52]

        The input channels are 255, in the form:
        C_in = bbox_num * (bbox_coords_pred + obj_score + classes_scores) = 3 * (4 + 1 + 80)
        where: bbox_coords_pred = (t_x, t_y, t_w, t_h) <--
        with: (t_x, t_y) offset w.r.t. to a fixed grid ('placed over the input')
              (t_w, t_h) offset w.r.t. the anchor boxes

        The output channels will be 255, in the form:
        C_out = bbox_num * (bbox_coords + obj_score + classes_scores) = 3 * (4 + 1 + 80)

        The C_out bbox_coords are computed as follow:
        b_x = sigmoid(t_x) + c_x
        b_y = sigmoid(t_y) + c_y
        b_w = p_w * e^t_w
        b_h = p_h * e^t_h
        - c_x and c_y are the top left coordinate of the cells in the input grid
        - p_w and p_h are the scaled anchors width and height
        """
        return x


def YoloNet(config, input_shape):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    layers_output, yolo_outputs = [], []
    idx = defaultdict(int)
    for block in config:
        if block['layer_type'] == 'convolutional':
            x = keras.layers.Conv2D(filters=block['filters'],
                                    kernel_size=block['size'],
                                    strides=block['stride'],
                                    padding='same' if (block['pad'] and block['stride']) else 'valid',
                                    use_bias=not int(block.get('batch_normalize', 0)),
                                    name=f'{idx["conv"]}_Conv2D'
                                    )(x)
            idx['conv'] += 1
            if 'batch_normalize' in block:
                x = keras.layers.BatchNormalization(name=f'{idx["bn"]}_bn')(x)
                idx['bn'] += 1
            if block['activation'] == 'leaky':
                x = keras.layers.LeakyReLU(alpha=0.1, name=f'{idx["leaky"]}_leaky')(x)
                idx['leaky'] += 1

        elif block['layer_type'] == 'upsample':
            s = block['stride']
            x = keras.layers.UpSampling2D(size=(s, s), name=str(idx['upsample']))(x)
            idx['upsample'] += 1

        elif block['layer_type'] == 'shortcut':
            layer_A, layer_B = layers_output[-1], layers_output[block['from']]
            x = keras.layers.Add(name=f'{idx["shortcut"]}_shortcut')([layer_A, layer_B])
            idx['shortcut'] += 1

        elif block['layer_type'] == 'route':
            if len(block['layers']) > 1:
                x = keras.layers.Concatenate(name=f'{idx["route"]}_route')([layers_output[i] for i in block['layers']])
            else:
                x = layers_output[block['layers'][0]]
            idx['route'] += 1

        elif block['layer_type'] == 'yolo':
            img_dim = input_shape[:2]
            anchors = [block['anchors'][i] for i in block['mask']]
            x = YoloLayer(anchors, block['classes'], img_dim)(x)
            x._name = str(idx['yolo'])
            idx['yolo'] += 1
            yolo_outputs.append(keras.layers.Reshape(target_shape=(-1, 85))(x))

        layers_output.append(x)

    model = keras.Model(inputs=inputs, outputs=yolo_outputs, name='YoloNet')
    return model

