import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from parameters import IMG_SIZE


def crop_image_from_gray(img, tol=7):
    """
    Applies masks to the original image and
    returns the a preprocessed image with
    3 channels

    :param img: A NumPy Array that will be cropped
    :param tol: The tolerance used for masking

    :return: A NumPy array containing the cropped image
    """
    # If for some reason we only have two channels
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    # If we have a normal RGB images
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray_img>tol   
    img_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape
    if (img_shape[0] != 0):
        image = []
        for i in range(img.ndim):
            image.append(img[:,:,i][np.ix_(mask.any(1),mask.any(0))])
        img = np.stack(image,axis=-1)
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
    path = 'train_images/' + id + '.png'
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img)    
    cropped_size = (IMG_SIZE, IMG_SIZE)
    img = cv2.resize(img, cropped_size)
    center = (int(IMG_SIZE/2), int(IMG_SIZE/2))
    radius = np.min(center)

    cir = np.zeros(cropped_size, dtype=np.uint8)
    cir = cv2.circle(img=cir, center=center, radius=radius, color=(255,255,255), thickness=-1)

    gauss_blur = cv2.GaussianBlur(img, (0, 0), radius/10)
    img = cv2.addWeighted(img, 4, gauss_blur, -4, 128)
    img = cv2.bitwise_and(img, img, mask=cir)
    return img 



def img_aug():
    train_datagen = ImageDataGenerator(rotation_range=60,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       validation_split=0.15,
                                       preprocessing_function=preprocess_image)
    return train_datagen
