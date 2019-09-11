# -*- coding: utf-8 -*-
"""TF 2.0 Fast-SCNN.ipynb
"""

# !pip install tensorflow-gpu==2.0.0-alpha0

import tensorflow as tf
import os
import random
import re
from PIL import Image
from dataset import data_gen
from matplotlib import pyplot as plt
import utils

#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from metrics.miou import MeanIoU
from keras import backend as K


import tensorflow.keras.backend.tensorflow_backend as ktf

"""
def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


ktf.set_session(get_session())
"""

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 8, 'The number of images in each batch during training.')

flags.DEFINE_string('procdata_dir', None, 'Where to write the event logs.')

flags.DEFINE_integer('no_of_epochs', None, 'The number of steps used for training.')

## num_classes needs to be reviewed
flags.DEFINE_integer('num_classes', 32, 'The number of classes in the dataset.')


NO_OF_TRAINING_IMAGES = len(os.listdir(FLAGS.procdata_dir + '/train_images/images'))
NO_OF_VAL_IMAGES = len(os.listdir(FLAGS.procdata_dir + '/val_images/images'))

logdir=FLAGS.procdata_dir + '/../export/tensorboard_logs'
tensorboard = TensorBoard(log_dir=logdir, histogram_freq=1,
    write_graph=True, write_images=False)

# Initialize mIoU metric
miou_metric = MeanIoU(FLAGS.num_classes)

""" Plots """
class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()

""" plots """
"""
# Model Architecture

"""


"""## Step 1: Learning to DownSample"""

# Input Layer
#[1242, 378]
nb_classes = 34
input_size_height = 512
input_size_width = 512
input_layer = tf.keras.layers.Input(shape=(input_size_height, input_size_width, 3), name = 'input_layer')
#input_layer = tf.keras.layers.Input(shape=(2048, 1024, 3), name = 'input_layer')

lds_layer = utils.conv_block(input_layer, 'conv', 32, (3, 3), strides = (2, 2))
lds_layer = utils.conv_block(lds_layer, 'ds', 48, (3, 3), strides = (2, 2))
lds_layer = utils.conv_block(lds_layer, 'ds', 64, (3, 3), strides = (2, 2))

"""## Step 2: Global Feature Extractor

"""

"""#### Assembling all the methods"""

gfe_layer = utils.bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3)
gfe_layer = utils.bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3)
gfe_layer = utils.bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3)
gfe_layer = utils.pyramid_pooling_block(gfe_layer, [2,4,6,8])

"""## Step 3: Feature Fusion"""

ff_layer1 = utils.conv_block(lds_layer, 'conv', 128, (1,1), padding='same', strides= (1,1), relu=False)

ff_layer2 = tf.keras.layers.UpSampling2D((4, 4))(gfe_layer)
ff_layer2 = tf.keras.layers.DepthwiseConv2D(128, strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)
ff_layer2 = tf.keras.layers.BatchNormalization()(ff_layer2)
ff_layer2 = tf.keras.activations.relu(ff_layer2)
ff_layer2 = tf.keras.layers.Conv2D(128, 1, 1, padding='same', activation=None)(ff_layer2)

#ff_layer2 = tf.keras.layers.Lambda(lambda ff_layer2: tf.image.resize(ff_layer2, (w,h)))(ff_layer2)

ff_final = tf.keras.layers.add([ff_layer1, ff_layer2])
ff_final = tf.keras.layers.BatchNormalization()(ff_final)
ff_final = tf.keras.activations.relu(ff_final)

"""## Step 4: Classifier"""

classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides = (1, 1), name = 'DSConv1_classifier')(ff_final)
classifier = tf.keras.layers.BatchNormalization()(classifier)
classifier = tf.keras.activations.relu(classifier)

classifier = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides = (1, 1), name = 'DSConv2_classifier')(classifier)
classifier = tf.keras.layers.BatchNormalization()(classifier)
classifier = tf.keras.activations.relu(classifier)


classifier = utils.conv_block(classifier, 'conv', nb_classes, (1, 1), strides=(1, 1), padding='same', relu=True)

classifier = tf.keras.layers.Dropout(0.3)(classifier)

classifier = tf.keras.layers.UpSampling2D((8, 8))(classifier)
classifier = tf.keras.activations.softmax(classifier)


### Model End ###

"""## DATA GEN"""
zipped_train = data_gen._get_pairs_from_paths(PROC_DATA = FLAGS.procdata_dir, gen='train')
zipped_val = data_gen._get_pairs_from_paths(PROC_DATA = FLAGS.procdata_dir, gen='val')

##TODO
# data_gen.image_segmentation_generator_1 doesn't work with do_argument=False

train_data_gen = data_gen.image_segmentation_generator_1(zipped=zipped_train, target_size = (input_size_width, input_size_height), batch_size=FLAGS.batch_size, do_augment=True)
val_data_gen = data_gen.image_segmentation_generator_1(zipped=zipped_val, target_size = (input_size_width, input_size_height), batch_size=FLAGS.batch_size, do_augment=True)

weights_path = FLAGS.procdata_dir + '../export/weights'
checkpoint = ModelCheckpoint(weights_path, monitor='METRI C_TO_MONITOR', 
                             verbose=1, save_best_only=True, mode='max')
csv_logger = CSVLogger('./log.out', append=True, separator=';')
earlystopping = EarlyStopping(monitor = 'METRIC_TO_MONITOR', verbose = 1,
                              min_delta = 0.01, patience = 3, mode = 'max')

callbacks_list = [checkpoint, csv_logger, earlystopping, tensorboard, plot_losses]
"""## Model Compilation"""

fast_scnn = tf.keras.Model(inputs = input_layer , outputs = classifier, name = 'Fast_SCNN')
optimizer = tf.keras.optimizers.SGD(momentum=0.9, lr=0.045)
fast_scnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', miou_metric.mean_iou])


# TODO
print("\n\n")
print(train_data_gen)
print("\n\n")


results = fast_scnn.fit_generator(generator = train_data_gen, epochs=FLAGS.no_of_epochs, 
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//FLAGS.batch_size),
                          validation_data=val_data_gen, 
                          validation_steps=(NO_OF_VAL_IMAGES//FLAGS.batch_size), 
                          callbacks=callbacks_list,
                          use_multiprocessing=False, 
                          shuffle=True, 
                          initial_epoch=0)