import os
import random as rn

import numpy as np
import pandas as pd
import tensorflow as tf

from model import Model
from parameters import *
from sklearn.utils import shuffle

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

print(tf.config.list_physical_devices('GPU'))

tf.config.optimizer.set_jit('autoclustering')
print(tf.config.optimizer.get_jit())

tf.keras.mixed_precision.set_global_policy('mixed_float16')

for i in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(i, True)

rn.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

mirrored_strategy = tf.distribute.MirroredStrategy()

parallel_batch_size = BATCH_SIZE * \
    mirrored_strategy.num_replicas_in_sync

print(f'parallel batch size: {parallel_batch_size}')

def input_pipeline():
    dr_2019_X = np.load(DR_2019_X_PATH)
    dr_2019_y = np.load(DR_2019_Y_PATH)
    dr_2019_y1 = pd.get_dummies(dr_2019_y).to_numpy()
    dr_2019_y2 = np.array([[1, 0] for _ in range(len(dr_2019_y1))])

    print(dr_2019_X.shape, dr_2019_y.shape)

    glaucoma_X = np.load(GLAUCOMA_X_PATH)
    glaucoma_y = np.load(GLAUCOMA_Y_PATH)
    glaucoma_y2 = pd.get_dummies(glaucoma_y).to_numpy().tolist()
    glaucoma_y1 = np.array([[1, 0, 0, 0, 0] for _ in range(len(glaucoma_y2))])

    print(glaucoma_X.shape, glaucoma_y.shape)

    X = np.concatenate((dr_2019_X, glaucoma_X), axis=0)
    y1 = np.concatenate((dr_2019_y1, glaucoma_y1), axis=0)
    y2 = np.concatenate((dr_2019_y2, glaucoma_y2), axis=0)
    print(X.shape, y1.shape, y2.shape)

    X, y1, y2 = shuffle(X, y1, y2, random_state=SEED)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X[:int(TRAINING_RATIO * len(X))], {"output_1": y1[:int(TRAINING_RATIO * len(X))], "output_2": y2[:int(TRAINING_RATIO * len(X))]}))
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (X[int(TRAINING_RATIO * len(X)):], {"output_1": y1[int(TRAINING_RATIO * len(X)):], "output_2": y2[int(TRAINING_RATIO * len(X)):]}))

    return X, y1, y2, set_options(train_dataset), set_options(validation_dataset)


def set_options(dataset):
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    options.experimental_optimization.apply_default_optimizations = True
    return dataset.with_options(options)

X, y1, y2, train_dataset, validation_dataset = input_pipeline()

with mirrored_strategy.scope():
    model = Model()
    model.build()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1, profile_batch=(1, 50))

history = model.fit(train_dataset.batch(parallel_batch_size, num_parallel_calls=tf.data.AUTOTUNE,
                                        deterministic=True).prefetch(8).cache(),
                   validation_dataset.batch(parallel_batch_size, num_parallel_calls=tf.data.AUTOTUNE,
                                        deterministic=True).prefetch(8).cache(),
                   FREEZED_EPOCH,
                   parallel_batch_size,
                   tensorboard_callback)


with mirrored_strategy.scope():
    model.unfreeze()

history = model.fit(train_dataset.batch(parallel_batch_size, num_parallel_calls=tf.data.AUTOTUNE,
                                        deterministic=True).prefetch(8).cache(),
                   validation_dataset.batch(parallel_batch_size, num_parallel_calls=tf.data.AUTOTUNE,
                                        deterministic=True).prefetch(8).cache(),
                   UNFREEZED_EPOCH,
                   parallel_batch_size,
                   tensorboard_callback)

dataset = tf.data.Dataset.from_tensor_slices((X, {"output_1": y1, "output_2": y2})) \
                                    .batch(parallel_batch_size, num_parallel_calls=tf.data.AUTOTUNE,
                                        deterministic=True).prefetch(8).cache()
dataset = set_options(dataset)

print(f'Accuracy: {model.evaluate(dataset, parallel_batch_size)[-2:]}')
