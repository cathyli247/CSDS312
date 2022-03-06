import os

import cv2
import numpy as np
import random as rn
import pandas as pd
from IPython.display import display
import tensorflow as tf

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import cohen_kappa_score

IMG_WIDTH = 224
IMG_HEIGHT = 224
CHANNELS = 3

TRAIN_DF_PATH = "./Data/train.csv"
TEST_DF_PATH = './Data/test.csv'
TRAIN_IMG_PATH = "./Data/train_images/"
TEST_IMG_PATH = './Data/test_images/'
SAVED_MODEL_NAME = 'effnet_modelB5.h5'

seed = 1234
rn.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

options = [0, 4]
print("Image IDs and Labels (TRAIN)")
train_df = pd.read_csv(TRAIN_DF_PATH)
train_df = train_df[train_df['diagnosis'].isin(options)]
# Add extension to id_code
train_df['id_code'] = train_df['id_code'] + ".png"
print(f"Training images: {train_df.shape[0]}")
display(train_df.head())
print("Image IDs (TEST)")
test_df = pd.read_csv(TEST_DF_PATH)
# Add extension to id_code
test_df['id_code'] = test_df['id_code'] + ".png"
print(f"Testing Images: {test_df.shape[0]}")
display(test_df.head())

BATCH_SIZE = 64

def get_preds_and_labels(model, generator):
    """
    Get predictions and labels from the generator

    :param model: A Keras model object
    :param generator: A Keras ImageDataGenerator object

    :return: A tuple with two Numpy Arrays. One containing the predictions
    and one containing the labels
    """
    preds = []
    labels = []
    for _ in range(int(np.ceil(generator.samples / BATCH_SIZE))):
        x, y = next(generator)
        preds.append(model.predict(x))
        labels.append(y)
    # Flatten list of numpy arrays
    return np.concatenate(preds).ravel(), np.concatenate(labels).ravel()



def crop_image_from_gray(img, tol=7):
    """
    Applies masks to the orignal image and
    returns the a preprocessed image with
    3 channels

    :param img: A NumPy Array that will be cropped
    :param tol: The tolerance used for masking

    :return: A NumPy array containing the cropped image
    """
    # If for some reason we only have two channels
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]

    # If we have a normal RGB images
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def preprocess_image(image, sigmaX=10):
    """
    The whole preprocessing pipeline:
    1. Read in image
    2. Apply masks
    3. Resize image to desired size
    4. Add Gaussian noise to increase Robustness

    :param img: A NumPy Array that will be cropped
    :param sigmaX: Value used for add GaussianBlur to the image

    :return: A NumPy array containing the preprocessed image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)
    return image

train_datagen = ImageDataGenerator(rotation_range=45,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   validation_split=0.15,
                                   preprocessing_function=preprocess_image)

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    x_col='id_code',
                                                    y_col='diagnosis',
                                                    directory = TRAIN_IMG_PATH,
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='raw',
                                                    subset='training')

val_generator = train_datagen.flow_from_dataframe(train_df,
                                                  x_col='id_code',
                                                  y_col='diagnosis',
                                                  directory = TRAIN_IMG_PATH,
                                                  target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='raw',
                                                  subset='validation')

effnet = tf.keras.applications.efficientnet.EfficientNetB0(weights="imagenet",
                                                           include_top=False,
                                                           input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS))

# effnet = EfficientNetB5(weights=None,
#                             include_top=False,
#                             input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS))

# effnet = tf.keras.applications.EfficientNetV2L(weights="imagenet",
#                         include_top=False,
#                         input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS))

# effnet.load_weights('efficientnet-b5_imagenet_1000_notop.h5')

effnet.trainable = False

model = tf.keras.Sequential([
    effnet,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(5, activation=elu),
    tf.keras.layers.Dense(1, activation="linear"),
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['binary_accuracy'])

checkpoint = tf.keras.callbacks.ModelCheckpoint(SAVED_MODEL_NAME, monitor='val_loss', save_best_only=True, mode='min')
earlystop = EarlyStopping(patience=5, monitor='val_loss', min_delta=0.0005)

model.summary()
#
history = model.fit(train_generator,
                    batch_size=BATCH_SIZE,
                    epochs=25,
                    validation_data=val_generator,
                    validation_batch_size=BATCH_SIZE,
                    callbacks=[earlystop, checkpoint])

model.load_weights(SAVED_MODEL_NAME)

for layer in model.layers[-20:]:
    if not isinstance(layer, BatchNormalization):
        layer.trainable = True

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=['binary_accuracy'])

history = model.fit(train_generator,
                    batch_size=BATCH_SIZE,
                    epochs=35,
                    validation_data=val_generator,
                    validation_batch_size=BATCH_SIZE,
                    callbacks=[earlystop, checkpoint])

model.load_weights(SAVED_MODEL_NAME)

# Calculate QWK on train set
y_train_preds, train_labels = get_preds_and_labels(model, train_generator)
y_train_preds = np.rint(y_train_preds).astype(np.uint8).clip(0, 4)

# Calculate score
train_score = cohen_kappa_score(train_labels, y_train_preds, weights="quadratic")

# Calculate QWK on validation set
y_val_preds, val_labels = get_preds_and_labels(model, val_generator)
y_val_preds = np.rint(y_val_preds).astype(np.uint8).clip(0, 4)

# Calculate score
val_score = cohen_kappa_score(val_labels, y_val_preds, weights="quadratic")

print(f"The Training Cohen Kappa Score is: {round(train_score, 5)}")
print(f"The Validation Cohen Kappa Score is: {round(val_score, 5)}")
