import os
import random as rn

import numpy as np
import pandas as pd
import tensorflow as tf

from model import build_model, fit_data, unfreeze_last_block
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

freezed_model_path = os.path.join(MODEL_FOLDER_PATH, FREEZED_SAVED_MODEL_NAME)
unfreezed_model_path = os.path.join(
    MODEL_FOLDER_PATH, UNFREEZED_SAVED_MODEL_NAME)

mirrored_strategy = tf.distribute.MirroredStrategy()

parallel_freezed_batch_size = FREEZED_BATCH_SIZE * \
    mirrored_strategy.num_replicas_in_sync
parallel_unfreezed_batch_size = UNFREEZED_BATCH_SIZE * \
    mirrored_strategy.num_replicas_in_sync


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

    cleaned_X = np.load(CLEANED_X_PATH)
    cleaned_Y = np.load(CLEANED_Y_PATH)
    cleaned_y1 = pd.get_dummies(cleaned_Y).to_numpy()
    cleaned_y2 = np.array([[1, 0] for _ in range(len(cleaned_y1))])
    print(cleaned_X.shape, cleaned_Y.shape)

    X = np.concatenate((dr_2019_X, glaucoma_X, cleaned_X), axis=0)
    y1 = np.concatenate((dr_2019_y1, glaucoma_y1, cleaned_y1), axis=0)
    y2 = np.concatenate((dr_2019_y2, glaucoma_y2, cleaned_y2), axis=0)
    print(X.shape, y1.shape, y2.shape)

    X, y1, y2 = shuffle(X, y1, y2, random_state=SEED)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X[:int(TRAINING_RATIO * len(X))], {"output_1": y1[:int(TRAINING_RATIO * len(X))], "output_2": y2[:int(TRAINING_RATIO * len(X))]}))
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (X[int(TRAINING_RATIO * len(X)):], {"output_1": y1[int(TRAINING_RATIO * len(X)):], "output_2": y2[int(TRAINING_RATIO * len(X)):]}))

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset = train_dataset.with_options(options)
    validation_dataset = validation_dataset.with_options(options)

    return X, y1, y2, train_dataset, validation_dataset


X, y1, y2, train_dataset, validation_dataset = input_pipeline()

with mirrored_strategy.scope():
    model = build_model()

history = fit_data(train_dataset.batch(FREEZED_BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True).prefetch(FREEZED_BATCH_SIZE * 2).cache(),
                   validation_dataset.batch(FREEZED_BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE,
                                            deterministic=True).prefetch(FREEZED_BATCH_SIZE * 2).cache(), model,
                   FREEZED_EPOCH, parallel_freezed_batch_size)


with mirrored_strategy.scope():
    model = unfreeze_last_block(model)

history = fit_data(train_dataset.batch(UNFREEZED_BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True).prefetch(UNFREEZED_BATCH_SIZE * 2).cache(),
                   validation_dataset.batch(UNFREEZED_BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE,
                                            deterministic=True).prefetch(UNFREEZED_BATCH_SIZE * 2).cache(), model,
                   UNFREEZED_EPOCH, parallel_unfreezed_batch_size)

dataset = tf.data.Dataset.from_tensor_slices(
    (X, {"output_1": y1, "output_2": y2})).batch(UNFREEZED_BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE,
                                                 deterministic=True).prefetch(UNFREEZED_BATCH_SIZE * 2).cache()
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
dataset = dataset.with_options(options)

print(
    f'Accuracy: {model.evaluate(dataset, batch_size=UNFREEZED_BATCH_SIZE)[-2:]}')
