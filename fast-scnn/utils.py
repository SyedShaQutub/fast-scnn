import tensorflow as tf
from keras import backend as K
import keras
import numpy as np
from keras.models import Model
from keras.layers import ReLU
#### Custom function for conv2d: conv_block

def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding='same', relu=True):
  
  if(conv_type == 'ds'):
    x = keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)
  else:
    x = keras.layers.Conv2D(kernel, kernel_size, padding=padding, strides = strides, kernel_regularizer=keras.regularizers.l2(0.00004))(inputs)  
  
  x = keras.layers.BatchNormalization()(x)
  
  if (relu):
    x = ReLU()(x)
  
  return x


#### residual custom method

def _res_bottleneck(inputs, filters, kernel, t, s, r=False):
    
    
    tchannel = keras.backend.int_shape(inputs)[-1] * t

    x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))

    x = keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = ReLU()(x)

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

def pyramid_pooling_block(input_tensor, Bin_sizes):
  concat_list = [input_tensor]
  #w and h are size of input_tensor
  size = input_tensor.shape
  h = size[1]
  w = size[2]
  # mapping compatible bin_sizes
  factor = np.gcd(h, w)
  bin_sizes = [x if ((h%x==0) and (w%x==0)) else 0 for x in Bin_sizes ]
  bin_sizes = list(filter((0).__ne__, bin_sizes))
  if (1 not in bin_sizes):
    bin_sizes.insert(0,1)
  ## TODO: need to add algo to add the next level compatible pyramid factor
  #bin_sizes.append(bin_sizes[-1]+factor)
  #while(len(bin_sizes)!=len(Bin_sizes)):

  for bin_size in bin_sizes:
    #TODO
    print("\n\n")
    print(size)
    print(w)
    print(h)
    print("\n\n")
    x = keras.layers.AveragePooling2D(pool_size=(h//bin_size, w//bin_size), strides=(h//bin_size, w//bin_size))(input_tensor)
    x = keras.layers.Conv2D(128, 3, strides=1, padding='same', kernel_regularizer=keras.regularizers.l2(0.00004))(x)
    x = keras.layers.BatchNormalization()(x)
    x = ReLU()(x)
    x = keras.layers.Lambda(lambda x: K.resize_images(x, (h//x.shape[1]),(w//x.shape[2]), data_format='channels_last', interpolation='bilinear'))(x)
  
    concat_list.append(x)

  return keras.layers.concatenate(concat_list)