# -*- coding: utf-8 -*-
"""TF 2.0 Fast-SCNN.ipynb
"""

# !pip install tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf
import keras
from keras.layers import Softmax, ReLU
import os
import random
import re
from PIL import Image
from dataset import data_gen
from matplotlib import pyplot as plt
import utils
import math
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
from tensorflow.python.keras.optimizers import Adam
from keras import regularizers
from metrics.miou import MeanIoU
from keras.models import Model

import keras.backend.tensorflow_backend as K
cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
K.set_session(K.tf.Session(config=cfg))
"""
import tensorflow.keras.backend.tensorflow_backend as ktf

def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


ktf.set_session(get_session())
"""
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 32} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
#flags = tf.compat.v1.app.flags
flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 8, 'The number of images in each batch during training.')

flags.DEFINE_string('procdata_dir', None, 'Where to write the event logs.')

flags.DEFINE_string('kittidata_dir', None, 'TODO')

flags.DEFINE_integer('no_of_epochs', None, 'The number of steps used for training.')

## num_classes needs to be reviewed
flags.DEFINE_integer('num_classes', 32, 'The number of classes in the dataset.')


NO_OF_TRAINING_IMAGES = len(os.listdir(FLAGS.procdata_dir + '/train_images/images'))
NO_OF_VAL_IMAGES = len(os.listdir(FLAGS.procdata_dir + '/val_images/images'))

logdir=FLAGS.procdata_dir + '/../../export/tensorboard_logs'
tensorboard = callbacks.TensorBoard(log_dir=logdir, histogram_freq=0,
    write_graph=True, write_images=False)

# Initialize mIoU metric
miou_metric = MeanIoU(FLAGS.num_classes)


   
""" plots """

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.fig1 = plt.figure(1)
        #self.fig2 = plt.figure(2)
        self.miou = []
        self.val_miou = []
        self.logs = []
        

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.miou.append(logs.get('mean_iou'))
        self.val_miou.append(logs.get('val_mean_iou'))
        self.i += 1
        #clear_output(wait=True)
        plt.plot(self.x, self.losses, 'C3', label="Training_loss")
        plt.plot(self.x, self.val_losses, 'C4', label="Val_loss")
        plt.plot(self.x, self.acc, 'C5', label="Training_acc")
        plt.plot(self.x, self.val_acc, 'C6', label="Val_acc")
        plt.plot(self.x, self.miou, 'C7', label="miou")
        plt.plot(self.x, self.val_miou, 'C8', label="val_miou")
        if self.i == 1:
            plt.legend()
        #plt.show(block=False)
        plt.savefig(logdir+'/../plots/fast_scnn.png')
        plt.close

def step_decay(epoch, lr):
    base_lrate = 0.045
    power = 0.9
    epochs = FLAGS.no_of_epochs
    updated_lrate = base_lrate * math.pow(1.0 - (float(epoch)/epochs),power)
    if(lr!=updated_lrate):
        print("\nlearning rate updated from {:.6f} to {:.6f} ".format(lr,updated_lrate))
    return updated_lrate

class PolylrPolicy(keras.callbacks.LearningRateScheduler):
    def __init__(self):
        super(PolylrPolicy, self).__init__(step_decay)

    #def on_epoch_begin(self, epoch):
    #    super(PolylrPolicy, self).on_epoch_begin(self, epoch)

# if tenor board is not available        
plot_losses = PlotLosses()
polylr_policy = PolylrPolicy()

"""## Step 1: Learning to DownSample"""

# Input Layer
#input_size = [378, 1248]
#original image size = [375, 1242]
nb_classes = 34
input_size = [384, 1248]
input_layer = keras.layers.Input(shape=(input_size[0],input_size[1],3),  name = 'input_layer', dtype = 'float32' )


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

ff_layer2 = keras.layers.UpSampling2D((4, 4))(gfe_layer)
ff_layer2 = keras.layers.DepthwiseConv2D(128, strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)
ff_layer2 = ReLU()(ff_layer2) ## as some authors suggest BN after activation layer sounds more logical in statistical POV
ff_layer2 = keras.layers.BatchNormalization()(ff_layer2)
ff_layer2 = keras.layers.Conv2D(128, kernel_size = 1, strides=1, padding='same', activation=None, kernel_regularizer=keras.regularizers.l2(0.00004))(ff_layer2)

ff_final = keras.layers.add([ff_layer1, ff_layer2])
ff_final = keras.layers.BatchNormalization()(ff_final)
ff_final = ReLU()(ff_final)

"""## Step 4: Classifier"""

classifier = keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides = (1, 1), name = 'DSConv1_classifier')(ff_final)
classifier = ReLU()(classifier)
classifier = keras.layers.BatchNormalization()(classifier)


classifier = keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides = (1, 1), name = 'DSConv2_classifier')(classifier)
classifier = ReLU()(classifier)
classifier = keras.layers.BatchNormalization()(classifier)


classifier = utils.conv_block(classifier, 'conv', nb_classes, (1, 1), strides=(1, 1), padding='same', relu=True)

classifier = keras.layers.Dropout(0.3)(classifier)

classifier = keras.layers.UpSampling2D((8, 8))(classifier)
classifier = Softmax(axis=-1)(classifier)

classifier = keras.backend.argmax(classifier, axis=-1)

### Model End ###

"""## DATA GEN"""
zipped_train = data_gen.get_pairs_from_paths(PROC_DATA = FLAGS.procdata_dir, gen='train')
zipped_val   = data_gen.get_pairs_from_paths(PROC_DATA = FLAGS.procdata_dir, gen='val')


train_data_gen = data_gen.image_segmentation_generator_categorical(zipped=zipped_train, target_size = input_size, batch_size=FLAGS.batch_size, nb_cls=nb_classes, do_augment=True)
val_data_gen = data_gen.image_segmentation_generator_categorical(zipped=zipped_val, target_size = input_size, batch_size=FLAGS.batch_size, nb_cls=nb_classes, do_augment=True)

weights_path = FLAGS.kittidata_dir + '/export/weights/model_weights-{epoch:02d}-{mean_iou:.2f}.h5'
model_checkpoint = callbacks.ModelCheckpoint(filepath=weights_path, monitor='mean_iou', 
                             verbose=1, save_best_only=True, save_weights_only=True, mode='max', period=1)
csv_logger = callbacks.CSVLogger(FLAGS.procdata_dir + '/../../expor t/CSV_LOGGER/training.log', append=True, separator=';')
earlystopping = callbacks.EarlyStopping(monitor = 'loss', verbose = 1,
                              min_delta = 0.001, patience = 20, mode = 'min')

#callbacks_list = [model_checkpoint, csv_logger, earlystopping, tensorboard, plot_losses, polylr_policy]
callbacks_list = [ csv_logger, tensorboard, model_checkpoint, polylr_policy]



"""## Model Compilation"""
fast_scnn = Model(inputs = input_layer , outputs = classifier, name = 'Fast_SCNN')
optimizer = keras.optimizers.SGD(momentum=0.9, lr=0.045, nesterov = True)
fast_scnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', miou_metric.mean_iou])
print(fast_scnn.metrics_names)
## loss = 'sparse_categorical_crossentropy' - ground truth dimension : (?,?,1)
## loss = 'categorical_crossentropy'        - ground truth dimension : (?,?,nb_classes)

results = fast_scnn.fit_generator(generator = train_data_gen, steps_per_epoch = (NO_OF_TRAINING_IMAGES//FLAGS.batch_size),
                          epochs=FLAGS.no_of_epochs,  verbose=2, callbacks=callbacks_list,
                          validation_data=val_data_gen, 
                          validation_steps=(NO_OF_VAL_IMAGES//FLAGS.batch_size),
                          use_multiprocessing=False, 
                          shuffle=True, 
                          initial_epoch=0)

model_path = FLAGS.kittidata_dir + '/export/weights/model.h5'
fast_scnn.save(model_path)

