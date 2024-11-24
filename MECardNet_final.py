
# MECardNet Model Implementation based on the paper description

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2L

# Multi-Scale Convolutional Mixture of Experts Block (MCME)
def MCME_block(inputs, filters):
    conv1 = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(inputs)
    conv2 = layers.Conv2D(filters, (5, 5), padding='same', activation='relu')(inputs)
    conv3 = layers.Conv2D(filters, (7, 7), padding='same', activation='relu')(inputs)
    
    output = layers.Add()([conv1, conv2, conv3])
    output = layers.BatchNormalization()(output)
    
    return output

# Adaptive Deep Supervision Block (ADS)
def ADS_block(inputs, filters, num_classes):
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid')(x)
    return x

# U-Net Decoder Block
def decoder_block(inputs, skip_features, filters):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    return x

# Cross-Additive Attention Mechanism
def cross_additive_attention(x):
    attention = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
    x = layers.Multiply()([x, attention])
    return x

# Post-Processing Block
def post_processing_block(segmentation_output):
    smooth_edges = tf.nn.conv2d(segmentation_output, filters=tf.ones((5, 5, 1, 1)), strides=[1, 1, 1, 1], padding='SAME')
    return smooth_edges

# MECardNet Model
def MECardNet(input_shape, num_classes):
    # Input Layer
    inputs = layers.Input(input_shape)
    
    # Backbone - EfficientNetV2L
    backbone = EfficientNetV2L(include_top=False, weights='imagenet', input_tensor=inputs)

    # Encoder features from EfficientNet backbone
    skip1 = backbone.get_layer('block2b_add').output  # 128x128
    skip2 = backbone.get_layer('block3b_add').output  # 64x64
    skip3 = backbone.get_layer('block4b_add').output  # 32x32
    skip4 = backbone.get_layer('block6d_add').output  # 16x16
    
    x = backbone.output  # Final bottleneck feature at 8x8

    # MCME Block applied on bottleneck
    x = MCME_block(x, 512)
    
    # Cross-Additive Attention Mechanism
    x = cross_additive_attention(x)
    
    # Decoder path
    x = decoder_block(x, skip4, 256)
    x = decoder_block(x, skip3, 128)
    x = decoder_block(x, skip2, 64)
    x = decoder_block(x, skip1, 32)
    
    # Final Convolution for segmentation
    output = layers.Conv2D(num_classes, (1, 1), padding="same", activation="sigmoid")(x)
    
    # Adaptive Deep Supervision
    output1 = ADS_block(skip2, 64, num_classes)
    output2 = ADS_block(skip3, 128, num_classes)
    
    # Post-processing the output to refine segmentation results
    post_processed_output = post_processing_block(output)
    
    # Model with final segmentation output and auxiliary supervision outputs
    model = Model(inputs, [post_processed_output, output1, output2])
    
    return model

# Compile and display the model
input_shape = (256, 256, 3)  # Example input shape
num_classes = 3  # Number of segmentation classes (e.g., LV, RV, Myo)

model = MECardNet(input_shape, num_classes)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
