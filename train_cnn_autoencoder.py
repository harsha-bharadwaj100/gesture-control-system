import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_1d_cnn_autoencoder(input_shape=(1500, 1), latent_dim=32, num_classes=5):
    """
    Builds a 1D Convolutional Autoencoder and Classifier model.
    Stage 1: Encoder compresses 1500-sample window into latent_dim.
    Stage 2: Classifier maps latent embeddings to gesture classes.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv1D(32, 7, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(16, 5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Flatten()(x)
    bottleneck = layers.Dense(latent_dim, activation='relu', name='bottleneck')(x)
    
    encoder = models.Model(inputs, bottleneck, name='encoder')
    
    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    feat_len = input_shape[0] // 4
    x = layers.Dense(feat_len * 16, activation='relu')(latent_inputs)
    x = layers.Reshape((feat_len, 16))(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    decoded = layers.Conv1D(1, 7, activation='linear', padding='same')(x)
    
    decoder = models.Model(latent_inputs, decoded, name='decoder')
    
    # Combined Autoencoder
    autoencoder_outputs = decoder(encoder(inputs))
    autoencoder = models.Model(inputs, autoencoder_outputs, name='autoencoder')
    
    # Classifier Head connected to Encoder Bottleneck
    clf_x = layers.Dense(64, activation='relu')(bottleneck)
    clf_x = layers.Dropout(0.3)(clf_x)
    clf_outputs = layers.Dense(num_classes, activation='softmax', name='classification')(clf_x)
    
    classifier_model = models.Model(inputs, clf_outputs, name='cnn_autoencoder_classifier')
    
    return autoencoder, classifier_model

def get_optimizer(name='adam', learning_rate=0.001):
    name = name.lower()
    if name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif name == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
