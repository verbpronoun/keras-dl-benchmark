import numpy as np

from keras.layers import Input, Dense, Activation, Flatten, concatenate
from keras.layers import Convolution2D, AveragePooling2D, BatchNormalization, MaxPooling2D, ZeroPadding2D
from keras.models import Model

from keras.utils import np_utils
from keras import backend as K

if (K.image_data_format() == 'channels_first'):
    bn_axis = 1
else:
    bn_axis = 3

def conv_layer(input, num_filters, kernel_size):
    x = BatchNormalization(axis=bn_axis)(input)
    x = Activation('relu')(x)
    x = Convolution2D(num_filters, (kernel_size, kernel_size), padding='same')(x)
    return x

def BottleneckBlock(input, growth_rate):
    # each 1x1 convolution reduces input to 4k feature-maps
    x = conv_layer(input, 4 * growth_rate, 1)
    x = conv_layer(x, growth_rate, 3)
    x = concatenate([input, x], axis=bn_axis)
    return x

def TransitionBlock(input, num_filters):
    # dense block has m feature maps -> transition gives (reduction*m) feature maps
    x = conv_layer(input, num_filters, 1)
    x = AveragePooling2D((2, 2), strides=2)(x)
    return x

def make_layer(x, n_blocks, growth_rate, num_transition_filters, last):
    for i in range(n_blocks):
        x = BottleneckBlock(x, growth_rate)
    if not last:
        x = TransitionBlock(x, num_transition_filters)
    return x

def DenseNet_builder(num_blocks, input_shape, num_classes, growth_rate, reduction):
    # initial layers modeled after resnet (32x32) 
    img_input = Input(shape=input_shape)
    x = Convolution2D(2*growth_rate, (3, 3), padding='same')(img_input)

    num_transition_filters = int(growth_rate * reduction)

    for i in range(3):
        x = make_layer(x, num_blocks[i], growth_rate, num_transition_filters, False)
        num_transition_filters = int(num_transition_filters * reduction)

    x = make_layer(x, num_blocks[3], growth_rate, num_transition_filters, True)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((4, 4), strides=4)(x)
    x = Flatten()(x)
    out = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=img_input, outputs=out)

def DenseNet121(input_shape, num_classes, growth_rate, reduction):
    return DenseNet_builder([6, 12, 24, 16], input_shape, num_classes, growth_rate, reduction)
#   print(model.summary())

# DenseNet121([32, 32, 3], 10, 32, 0.5)





