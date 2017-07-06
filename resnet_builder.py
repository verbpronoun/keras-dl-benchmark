import numpy as np

from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Convolution2D, AveragePooling2D, BatchNormalization, MaxPooling2D, ZeroPadding2D
from keras.models import Model
from keras import layers
from keras.utils import np_utils
from keras import backend as K

if (K.image_data_format() == 'channels_first'):
    bn_axis = 1
else:
    bn_axis = 3


def BasicBlock(input, numFilters, stride, isConvBlock):
    expansion = 1
    x = Convolution2D(numFilters, (3, 3), strides=stride, padding='same')(input)
    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.00001)(x)
    x = Activation('relu')(x)

    x = Convolution2D(numFilters, (3, 3), padding='same')(x)
    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.00001)(x)

    if isConvBlock:
        shortcut = Convolution2D(expansion * numFilters, (1, 1), strides = stride)(input)
        shortcut = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.00001)(x)
    else:
        shortcut = input

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    return x


def PreActBlock(input, numFilters, stride, isConvBlock):
    expansion = 1
    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.00001)(input)
    x = Activation('relu')(x)

    if isConvBlock:
        shortcut = Convolution2D(expansion * numFilters, (1, 1), strides = stride)(x)
    else:
        shortcut = x
    
    x = Convolution2D(numFilters, (3, 3), strides=stride, padding='same')(x)

    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.00001)(x)
    x = Activation('relu')(x)
    x = Convolution2D(numFilters, (3, 3), padding = 'same')(x)

    x = layers.add([x, shortcut])

    return x

def BottleneckBlock(input, numFilters, stride, isConvBlock):
    expansion = 4
    x = Convolution2D(numFilters, (1, 1), strides=stride)(input)
    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.00001)(x)
    x = Activation('relu')(x)

    x = Convolution2D(numFilters, (3, 3), padding='same')(x) 
    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.00001)(x)
    x = Activation('relu')(x)

    x = Convolution2D(4 * numFilters, (1, 1))(x)
    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.00001)(x)

    if isConvBlock:
        shortcut = Convolution2D(expansion * numFilters, (1, 1), strides=stride)(input)
        shortcut = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.00001)(shortcut)
    else:
        shortcut = input

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    return x
    

def PreActBottleneck(input, numFilters, stride, isConvBlock):
    expansion = 4
    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.00001)(input)
    x = Activation('relu')(x)

    if isConvBlock:
        shortcut = Convolution2D(expansion * numFilters, (1, 1), strides = stride)(x)
    else:
        shortcut = x

    x = Convolution2D(numFilters, (1, 1))(x)

    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.00001)(x)
    x = Activation('relu')(x)
    x = Convolution2D(numFilters, (3, 3), padding='same')(x) 

    x = BatchNormalization(axis=bn_axis, momentum=0.1, epsilon=0.00001)(x)
    x = Activation('relu')(x)
    x = Convolution2D(4 * numFilters, (1, 1))(x)

    # did not finish...


def make_layer(block, input, numFilters, numBlocks, stride):
    x = block(input, numFilters, stride, True)
    for i in range(numBlocks - 1):
        x = block(x, numFilters, 1, False)
    return x

def ResNet_builder(block, num_blocks, input_shape, num_classes):
    img_input = Input(shape=input_shape)
    x = Convolution2D(64, (3, 3), padding='same')(img_input)
    x = BatchNormalization(axis = bn_axis, momentum=0.1, epsilon=0.00001)(x)
    x = Activation('relu')(x)
    
    x = make_layer(block, x, 64, num_blocks[0], 1)
    x = make_layer(block, x, 128, num_blocks[1], 2)
    x = make_layer(block, x, 256, num_blocks[2], 2)
    x = make_layer(block, x, 512, num_blocks[3], 2)
    
    x = AveragePooling2D((4, 4), strides=2)(x)
    x = Flatten()(x)
    out = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=img_input, outputs=out)
    
def ResNet18(input_shape, num_classes):
    return ResNet_builder(PreActBlock, [2, 2, 2, 2],input_shape, num_classes)

def ResNet34(input_shape, num_classes):
    return ResNet_builder(BasicBlock, [3, 4, 6 ,3],input_shape, num_classes)

def ResNet50(input_shape, num_classes):
    return ResNet_builder(BottleneckBlock, [3, 4, 6, 3], input_shape, num_classes)
        
        
