from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import UpSampling2D
from keras.layers import Dropout
from keras.layers import concatenate
from all_params import IMG_ROWS, IMG_COLS

def get_model(input_shape=(IMG_ROWS, IMG_COLS, 1), train=True):
    layers = {}
    layers['inputs'] = Input(shape=input_shape, name='inputs')

    layers['conv1_1'] = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_1')(layers['inputs'])
    layers['conv1_2'] = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_2')(layers['conv1_1'])
    layers['pool_1'] = MaxPool2D(pool_size=(2, 2), name='pool_1')(layers['conv1_2'])
    if train == True:
        layers['dropout_1'] = Dropout(0.25, name='dropout_1')(layers['pool_1'])
        layers['conv2_1'] = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_1')(layers['dropout_1'])
    else:
        layers['conv2_1'] = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_1')(layers['pool_1'])
    layers['conv2_2'] = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_2')(layers['conv2_1'])
    layers['pool_2'] = MaxPool2D(pool_size=(2, 2), name='pool_2')(layers['conv2_2'])
    if train == True:
        layers['dropout_2'] = Dropout(0.25, name='dropout_2')(layers['pool_2'])
        layers['conv3_1'] = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_1')(layers['dropout_2'])
    else:
        layers['conv3_1'] = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_1')(layers['pool_2'])
    layers['conv3_2'] = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_2')(layers['conv3_1'])
    layers['pool_3'] = MaxPool2D(pool_size=(2, 2), name='pool_3')(layers['conv3_2'])
    if train == True:
        layers['dropout_3'] = Dropout(0.25, name='dropout_3')(layers['pool_3'])
        layers['conv4_1'] = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_1')(layers['dropout_3'])
    else:
        layers['conv4_1'] = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_1')(layers['pool_3'])
    layers['conv4_2'] = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_2')(layers['conv4_1'])
    layers['pool_4'] = MaxPool2D(pool_size=(2, 2), name='pool_4')(layers['conv4_2'])
    if train == True:
        layers['dropout_4'] = Dropout(0.25, name='dropout_4')(layers['pool_4'])
        layers['conv5_1'] = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_1')(layers['dropout_4'])
    else:
        layers['conv5_1'] = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_1')(layers['pool_4'])
    layers['conv5_2'] = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_2')(layers['conv5_1'])

    layers['upsample_1'] = UpSampling2D(size=(2, 2), name='upsample_1')(layers['conv5_2'])
    layers['concat_1'] = concatenate([layers['upsample_1'], layers['conv4_2']], name='concat_1')
    layers['conv6_1'] = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv6_1')(layers['concat_1'])
    layers['conv6_2'] = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv6_2')(layers['conv6_1'])
    if train == True:
        layers['dropout_6'] = Dropout(0.25, name='dropout_6')(layers['conv6_2'])
        layers['upsample_2'] = UpSampling2D(size=(2, 2), name='upsample_2')(layers['dropout_6'])
    else:
        layers['upsample_2'] = UpSampling2D(size=(2, 2), name='upsample_2')(layers['conv6_2'])
    layers['concat_2'] = concatenate([layers['upsample_2'], layers['conv3_2']], name='concat_2')
    layers['conv7_1'] = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv7_1')(layers['concat_2'])
    layers['conv7_2'] = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv7_2')(layers['conv7_1'])
    if train == True:
        layers['dropout_7'] = Dropout(0.25, name='dropout_7')(layers['conv7_2'])
        layers['upsample_3'] = UpSampling2D(size=(2, 2), name='upsample_3')(layers['dropout_7'])
    else:
        layers['upsample_3'] = UpSampling2D(size=(2, 2), name='upsample_3')(layers['conv7_2'])
    layers['concat_3'] = concatenate([layers['upsample_3'], layers['conv2_2']], name='concat_3')
    layers['conv8_1'] = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv8_1')(layers['concat_3'])
    layers['conv8_2'] = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv8_2')(layers['conv8_1'])
    if train == True:
        layers['dropout_8'] = Dropout(0.25, name='dropout_8')(layers['conv8_2'])
        layers['upsample_4'] = UpSampling2D(size=(2, 2), name='upsample_4')(layers['dropout_8'])
    else:
        layers['upsample_4'] = UpSampling2D(size=(2, 2), name='upsample_4')(layers['conv8_2'])
    layers['concat_4'] = concatenate([layers['upsample_4'], layers['conv1_2']], name='concat_4')
    layers['conv9_1'] = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv9_1')(layers['concat_4'])
    layers['conv9_2'] = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv9_2')(layers['conv9_1'])
    if train == True:
        layers['dropout_9'] = Dropout(0.25, name='dropout_9')(layers['conv9_2'])
        layers['outputs'] = Conv2D(1, (1, 1), activation='sigmoid', name='outputs')(layers['dropout_9'])
    else:
        layers['outputs'] = Conv2D(1, (1, 1), activation='sigmoid', name='outputs')(layers['conv9_2'])

    model = Model(inputs=layers['inputs'], outputs=layers['outputs'])

    return model
