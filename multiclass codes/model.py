import os
import random as rn

import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras import Sequential, callbacks, layers, optimizers, mixed_precision

from parameters import *

rn.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


def build_model():
    model_path = os.path.join(MODEL_FOLDER_PATH, PRE_TRAINED_MODEL_NAME)
    if PRE_TRAINED_MODEL_NAME not in os.listdir(MODEL_FOLDER_PATH):
        r = requests.get(PRE_TRAINED_MODEL_URL, allow_redirects=True)
        with open(model_path, 'wb') as f:
            f.write(r.content)

    input_0 = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    input_0 = layers.RandomFlip(seed=SEED)(input_0)
    input_0 = layers.RandomRotation(factor=(-0.45, 0.45), seed=SEED)(input_0)

    model_path = os.path.join(MODEL_FOLDER_PATH, PRE_TRAINED_MODEL_NAME)
    effnet = tf.keras.applications.efficientnet_v2.EfficientNetV2L(weights=model_path,
                                                                   include_top=False,
                                                                   input_tensor=input_0,
                                                                   input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
    effnet.trainable = False

    def build_top(effnet_output, suffix, output_num, activation):
        conv = layers.Conv2D.from_config(effnet.layers[-3].get_config())
        conv._name = f'top_conv_{suffix}'
        top_component = conv(effnet_output)
        bn = layers.BatchNormalization.from_config(
            effnet.layers[-2].get_config())
        bn._name = f'top_bn_{suffix}'
        top_component = bn(top_component)
        act = layers.Activation.from_config(effnet.layers[-1].get_config())
        act._name = f'top_activation_{suffix}'
        top_component = act(top_component)

        top_component = layers.GlobalAveragePooling2D()(top_component)
        top_component = layers.BatchNormalization()(top_component)
        top_component = layers.Dropout(0.5)(top_component)

        top_component = layers.Dense(
            output_num, activation=activation, name=f'output_{suffix}')(top_component)
        return top_component

    effnet_no_top_block = effnet.layers[-4].output

    top_output_1 = build_top(effnet_no_top_block, 1, 5, 'softmax')
    top_output_2 = build_top(effnet_no_top_block, 2, 2, 'softmax')

    model = tf.keras.Model(inputs=effnet.input, outputs=[
                           top_output_1, top_output_2])

    for i in [-3, -2, -1]:
        model.layers[2*i-8].set_weights(effnet.layers[i].get_weights())
        model.layers[2*i-7].set_weights(effnet.layers[i].get_weights())

    model.build((None, IMG_SIZE, IMG_SIZE, CHANNELS))

    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1,
        decay_steps=5,
        decay_rate=0.5)

    losses = {
        'output_1': 'categorical_crossentropy',
        'output_2': 'binary_crossentropy',
    }

    model.compile(loss=losses,
                  optimizer=mixed_precision.LossScaleOptimizer(optimizers.Adam(
                      learning_rate=lr_schedule)),
                  metrics=['accuracy'])

    return model


earlystop = callbacks.EarlyStopping(patience=15,
                                    monitor='val_loss',
                                    min_delta=0.0005)
tensorboard = callbacks.TensorBoard(log_dir="./logs/freezed",
                                    update_freq=100,
                                    profile_batch='1, 49')


def fit_data(train_dataset, validation_dataset, model, epoch, batch_size, model_path):

    checkpoint = callbacks.ModelCheckpoint(filepath=model_path,
                                           monitor='val_loss',
                                           save_best_only=True,
                                           mode='min')

    return model.fit(train_dataset,
                     batch_size=batch_size,
                     epochs=epoch,
                     validation_data=validation_dataset,
                     validation_batch_size=batch_size,
                     callbacks=[earlystop, checkpoint, tensorboard])


def unfreeze_last_block(model):
    for layer in model.layers[-29:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=5,
        decay_rate=0.5)

    losses = {
        'output_1': 'categorical_crossentropy',
        'output_2': 'binary_crossentropy',
    }

    model.compile(loss=losses,
                  optimizer=mixed_precision.LossScaleOptimizer(optimizers.Adam(
                      learning_rate=lr_schedule)),
                  metrics=['accuracy'])
    return model
