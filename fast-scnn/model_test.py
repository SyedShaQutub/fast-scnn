# -*- coding: utf-8 -*-
"""TF 2.0 Fast-SCNN.ipynb
"""

# !pip install tensorflow-gpu==2.0.0-alpha0

from keras import backend as K
import keras
from keras.layers import Softmax, ReLU
#import tensorflow as tf

"""
# Model Architecture
#### Custom function for conv2d: conv_block
"""
def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding='same', relu=True):
  
  if(conv_type == 'ds'):
    x = keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)
  else:
    x = keras.layers.Conv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)  
  
  x = keras.layers.BatchNormalization()(x)
  
  if (relu):
    x = ReLU()(x)
  
  return x

"""## Step 1: Learning to DownSample"""

# Input Layer
input_layer = keras.layers.Input(shape=(1024, 1024, 3), name = 'input_layer')

lds_layer = conv_block(input_layer, 'conv', 32, (3, 3), strides = (2, 2))
lds_layer = conv_block(lds_layer, 'ds', 48, (3, 3), strides = (2, 2))
lds_layer = conv_block(lds_layer, 'ds', 64, (3, 3), strides = (2, 2))

"""## Step 2: Global Feature Extractor
#### residual custom method
"""

def _res_bottleneck(inputs, filters, kernel, t, s, r=False):
    
    
    tchannel = keras.backend.int_shape(inputs)[-1] * t

    x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))

    x = keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    #x = keras.activations.relu(x)

    x = conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)

    if r:
        x = keras.layers.add([x, inputs])
    return x

"""#### Bottleneck custom method"""

def bottleneck_block(inputs, filters, kernel, t, strides, n):
  x = _res_bottleneck(inputs, filters, kernel, t, strides)
  
  for i in range(1, n):
    x = _res_bottleneck(x, filters, kernel, t, 1, True)

  return x

"""#### PPM Method"""

def pyramid_pooling_block(input_tensor, bin_sizes):
  concat_list = [input_tensor]
  w = 32
  h = 32

  for bin_size in bin_sizes:
    x = keras.layers.AveragePooling2D(pool_size=(w//bin_size, h//bin_size), strides=(w//bin_size, h//bin_size))(input_tensor)
    x = keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = keras.layers.Lambda(lambda x: K.resize_images(x, (h//x.shape[1]), (w//x.shape[2]), data_format='channels_last', interpolation='bilinear'))(x)

    concat_list.append(x)

  return keras.layers.concatenate(concat_list)

"""#### Assembling all the methods"""

gfe_layer = bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3)
gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3)
gfe_layer = bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3)
gfe_layer = pyramid_pooling_block(gfe_layer, [2,4,8])

"""## Step 3: Feature Fusion"""

ff_layer1 = conv_block(lds_layer, 'conv', 128, (1,1), padding='same', strides= (1,1), relu=False)

ff_layer2 = keras.layers.UpSampling2D((4, 4))(gfe_layer)
ff_layer2 = keras.layers.DepthwiseConv2D(128, strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)
ff_layer2 = keras.layers.BatchNormalization()(ff_layer2)
ff_layer2 = ReLU()(ff_layer2)
ff_layer2 = keras.layers.Conv2D(128, kernel_size=1, strides=1, padding='same', activation=None)(ff_layer2)

ff_final = keras.layers.add([ff_layer1, ff_layer2])
ff_final = keras.layers.BatchNormalization()(ff_final)
ff_final = ReLU()(ff_final)

"""## Step 4: Classifier"""

classifier = keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides = (1, 1), name = 'DSConv1_classifier')(ff_final)
classifier = keras.layers.BatchNormalization()(classifier)
classifier = ReLU()(classifier)

classifier = keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides = (1, 1), name = 'DSConv2_classifier')(classifier)
classifier = keras.layers.BatchNormalization()(classifier)
classifier = ReLU()(classifier)


classifier = conv_block(classifier, 'conv', 19, (1, 1), strides=(1, 1), padding='same', relu=True)

classifier = keras.layers.Dropout(0.3)(classifier)

classifier = keras.layers.UpSampling2D((8, 8))(classifier)
classifier = Softmax()(classifier)

"""## Model Compilation"""

fast_scnn = keras.Model(inputs = input_layer , outputs = classifier, name = 'Fast_SCNN')
optimizer = keras.optimizers.SGD(momentum=0.9, lr=0.045)
fast_scnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#fast_scnn.summary()

#keras.utils.plot_model(fast_scnn, show_layer_names=True, show_shapes=True)

print("\n\n\n")
print("Model is Valid")
print("\n\n\n")