IMG_SIZE = 380
CHANNELS = 3
SEED = 12345

DATA_FOLDER_PATH = '/home/yxz1648/project/Data/raw/dr_2019'
MODEL_FOLDER_PATH = './models'
LOGS_FOLDER_PATH = './logs/freezed'

PRE_TRAINED_MODEL_URL = 'http://storage.googleapis.com/tensorflow/keras-applications/efficientnet_v2/efficientnetv2-l_notop.h5'

PRE_TRAINED_MODEL_NAME = 'efficientnetv2-l_notop.h5'
FREEZED_SAVED_MODEL_NAME = 'freezed_model.h5'
UNFREEZED_SAVED_MODEL_NAME = 'unfreezed_model.h5'

FREEZED_BATCH_SIZE = 64
UNFREEZED_BATCH_SIZE = 8
BUFFER_SIZE = 128
FREEZED_EPOCH = 1
UNFREEZED_EPOCH = 1

TRAINING_RATIO = 0.85

DR_2019_X_PATH = "/home/yxz1648/project/Data/preprocessed/2019/2019train_X.npy"
DR_2019_Y_PATH = "/home/yxz1648/project/Data/preprocessed/2019/2019train_y.npy"

GLAUCOMA_X_PATH = "/home/yxz1648/project/Data/preprocessed/glaucoma/glaucoma_X.npy"
GLAUCOMA_Y_PATH = "/home/yxz1648/project/Data/preprocessed/glaucoma/glaucoma_y.npy"
