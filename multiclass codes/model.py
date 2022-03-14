import os
import requests
import numpy as np
import random as rn
import tensorflow as tf
from keras import layers, Sequential
from parameters import *

rn.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


def build_model():
    if PRE_TRAINED_MODEL_PATH not in os.listdir('./'):
        r = requests.get(PRE_TRAINED_MODEL_URL, allow_redirects=True)
        with open(PRE_TRAINED_MODEL_PATH, 'wb') as f:
            f.write(r.content)

    effnet = tf.keras.applications.efficientnet_v2.EfficientNetV2L(weights=PRE_TRAINED_MODEL_PATH,
                                                                   include_top=False,
                                                                   input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS))

    effnet.trainable = False

    model = Sequential([
        effnet,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(5, activation='softmax')
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1,
        decay_steps=5,
        decay_rate=0.5)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(
                      learning_rate=lr_schedule),
                  metrics=['categorical_accuracy'])

    return model


def unfreeze_last_block(model):
    for layer in model.layers[-21:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=5,
        decay_rate=0.5)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(
                      learning_rate=lr_schedule),
                  metrics=['categorical_accuracy'])

    return model
