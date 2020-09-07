"""
Copyright 2018
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import tensorflow.keras

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding1D
from tensorflow.keras.layers import MaxPooling1D, Lambda
from tensorflow.keras.layers import UpSampling1D, Conv1D, Cropping1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import MaxPooling1D, concatenate, Add
import tensorflow.keras.backend as K


import tensorflow.keras.utils
from tensorflow.keras.layers import Layer
class Bias(Layer):
    def build(self, input_shape):
        self.bias = self.add_weight(shape=(input_shape[-1],),
                                        initializer="zeros",
                                        name='bias',
                                        regularizer=None,
                                        constraint=None)
        super(Bias, self).build(input_shape)
        
    def call(self, inputs):
        outputs = K.bias_add(
                inputs,
                self.bias)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape


tensorflow.keras.layers.Bias = Bias

def partial_convolution(input_, mask, filters, shape, stride, activation):
    convolution_layer = Conv1D(filters, shape, strides=stride, use_bias=False,
                               padding="same")
    
    mask_sum_layer = Conv1D(filters, shape, strides=stride, 
                                  padding="same", 
                                  weights=[np.ones((shape[0], input_.shape[-1], filters)),
                                           np.zeros((filters,))])
    
    mask_sum_layer.trainable = False
    
    mask_sum = mask_sum_layer(mask)
    
    new_mask = Lambda(lambda x: K.clip(x, 0, 1))(mask_sum)
    
    output = convolution_layer(tensorflow.keras.layers.multiply([mask, input_]))
    
    inv_sum = Lambda(lambda x: filters * shape[0] / (.0001 + x))(mask_sum) 
    
    output = tensorflow.keras.layers.multiply([output, inv_sum])
    
    output = Bias()(output)
    
    output = activation(output)
    
    return output, new_mask

def normal_convolution(input_, mask, filters, shape, stride, activation):
    convolution_layer = Conv1D(filters, shape, strides=stride,
                               padding="same")
    
    output = convolution_layer(input_)
    
    output = activation(output)
    
    return output, mask

def residual_block(input_, mask):
    output, new_mask = normal_convolution(input_, mask, 512, (3,), 1, Activation("relu"))
    output, new_mask = normal_convolution(output, new_mask, 512, (3,), 1, Activation("relu"))
    output, new_mask = normal_convolution(output, new_mask, int(input_.shape[-1]), (3,), 1, Activation("relu"))
    output = tensorflow.keras.layers.Add()([output, input_])
    new_mask = tensorflow.keras.layers.Add()([mask, new_mask])
    new_mask = Lambda(lambda x: K.clip(x, 0, 1))(new_mask)
    return output, new_mask
    

def sequence_unet(patch_size=128, input_dim=37, output_dim=33):
    
    global sequence_unet_input
    global sequence_unet_input_mask
    input_ = Input((patch_size, input_dim))
    sequence_unet_input = input_
    input_mask = Input((patch_size, input_dim))
    sequence_unet_input_mask = input_mask
    skips = []
    output = input_
    mask = input_mask
    for shape, filters in zip([7, 5, 5, 3, 3, 3, 3], [64, 128, 256, 256, 256, 256, 256]):
        skips.append((output, mask))
        print(output.shape)
        output, mask = partial_convolution(output, mask, filters, (shape,), 2,
                                           Activation("relu"))
        if shape != 7:
            output = BatchNormalization()(output)
    for shape, filters in zip([4, 4, 4, 4, 4, 4, 4], [256, 256, 256, 256, 128, 64, output_dim]):
        output = UpSampling1D()(output)
        mask = UpSampling1D()(mask)
        skip_output, skip_mask = skips.pop()
        output = concatenate([output, skip_output], axis=2)
        mask = concatenate([mask, skip_mask], axis=2)
        
        if filters != output_dim:
            activation = tensorflow.keras.layers.LeakyReLU(.2)
        else:
            activation = Activation("linear")
        output, mask = partial_convolution(output, mask, filters, (shape,), 1, activation)
        #output, mask = residual_block(output, mask)
        if filters != output_dim:
            output = BatchNormalization()(output)
    assert len(skips) == 0
    return Model([input_, input_mask], [output])


