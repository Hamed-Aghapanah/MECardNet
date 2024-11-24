"""   Created on Sat May 18 14:01:24 2024

@author       :   Dr Hamed Aghapanah  , PhD bio-electrics

@affiliation  :  Isfahan University of Medical Sciences

"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, concatenate, Multiply, Add, Activation, Softmax
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.models import Model

def conv_block(inputs, filters):
    conv = Conv2D(filters, (3, 3), padding='same', activation='relu')(inputs)
    return conv

def upsample_block(inputs, skip_connection, filters):
    upsample = UpSampling2D((2, 2))(inputs)
    merge = concatenate([upsample, skip_connection])
    conv = conv_block(merge, filters)
    return conv

def attention_block(inputs, filters):
    conv = Conv2D(filters, (3, 3), padding='same', activation='sigmoid')(inputs)
    return conv

def mcme_block(inputs):
    experts = []
    for i in range(1, 7):
        expert = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        experts.append(expert)
    
    attention_maps = [attention_block(expert, 64) for expert in experts]
    combined = Add()([Multiply()([expert, attention]) for expert, attention in zip(experts, attention_maps)])
    
    return combined

def create_mecardnet(input_shape):
    inputs = Input(input_shape)
    
    # EfficientNetV2L Backbone
    backbone = EfficientNetV2L(include_top=False, input_tensor=inputs)
    backbone_outputs = [backbone.get_layer(name).output for name in ['block1d_drop', 'block2g_drop', 'block4a_expand_bn', 'block5r_expand_activation']]

    # Bottleneck
    bottleneck = backbone.get_layer('top_activation').output

    # Decoder Block
    d5 = upsample_block(bottleneck, backbone_outputs[3], 512)
    d4 = upsample_block(d5, backbone_outputs[2], 256)
    d3 = upsample_block(d4, backbone_outputs[1], 128)
    d2 = upsample_block(d3, backbone_outputs[0], 64)
    
    # Multi-Scale Convolutional Mixture of Expert (MCME) Block
    mcme_output = mcme_block(d2)
    
    # Adaptive Deep Supervision Block
    output_layers = []
    for i in range(6):
        aux_output = Conv2D(3, (1, 1), activation='sigmoid')(mcme_output)
        output_layers.append(aux_output)
    
    combined_output = Add()(output_layers)
    
    model = Model(inputs, combined_output)
    
    return model

# ایجاد مدل
input_shape = (256, 256, 3)
model = create_mecardnet(input_shape)

# خلاصه مدل
model.summary()
model.save('MM.h5')

# کامپایل مدل
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

