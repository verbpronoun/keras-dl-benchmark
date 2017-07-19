import numpy as np
np.random.seed(123)  # for reproducibility

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import six
import os
import time
import csv
import h5py

from collections import OrderedDict
from collections import Iterable
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback
from keras.datasets import cifar10
from keras import backend as K
import resnext_builder

batch_size = 128
num_epochs = 350
num_classes = 10
data_augmentation = True

def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.1
    epoch_drop = 100
    if epoch < 50:
        return initial_lrate
    lrate = initial_lrate * math.pow(drop, math.floor((epoch - 50)/epoch_drop))
    return lrate

class CSV_Logger(Callback):
    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
        super(CSV_Logger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a' + self.file_flags)
        else:
            self.csv_file = open(self.filename, 'w' + self.file_flags)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())

            # class CustomDialect(csv.excel):
            #     delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch', 'epoch time'] + self.keys, dialect='excel-tab')
            if self.append_header:
                self.writer.writeheader()

        end_time = time.time() - self.epoch_time_start
        row_dict = OrderedDict({'epoch': epoch, 'epoch time': end_time})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None

# Preprocess train and test set data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

if (K.image_data_format() == 'channels_first'):
    for i in range(x_train.shape[1]):
        mean = np.mean(x_train[:,i,:,:])
        std_dev = np.std(x_train[:,i,:,:])
        x_train[:,i,:,:] -= mean
        x_train[:,i,:,:] /= std_dev
        x_test[:,i,:,:] -= mean
        x_test[:,i,:,:] /= std_dev
else:
    for i in range(x_train.shape[3]):
        mean = np.mean(x_train[:,:,:,i])
        std_dev = np.std(x_train[:,:,:,i])
        x_train[:,:,:,i] -= mean
        x_train[:,:,:,i] /= std_dev
        x_test[:,:,:,i] -= mean
        x_test[:,:,:,i] /= std_dev

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

img_input = Input(shape=x_train.shape[1:])

model = resnext_builder.ResNext29(x_train.shape[1:], num_classes=num_classes, cardinality=32, bottleneck_width=4)

if (K.image_data_format() == 'channels_first'):
    csv_logger = CSV_Logger('train_cifar10_resnext29_th.log', append=True)
else:
    csv_logger = CSV_Logger('train_cifar10_resnext29_tf.log', append=True)
    
callbacks_list = [csv_logger]

datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode='constant',
            cval=0,
            horizontal_flip=True,
            vertical_flip=False)
datagen.fit(x_train)

for num_epochs, lr_rate in [(150, 0.1), (100, 0.01), (100, 0.001)]:
    sgd = SGD(lr=lr_rate, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    history = model.fit_generator(datagen.flow(x_train, y_train, 
                                               batch_size=batch_size),
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  epochs=num_epochs,
                                  callbacks=callbacks_list,
                                  verbose=1,
                                  validation_data=(x_test, y_test))


if (K.image_data_format() == 'channels_first'):
    model.save('cifar10-resnext29-th.h5')
else:
    model.save('cifar10-resnext29-tf.h5')

scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        
