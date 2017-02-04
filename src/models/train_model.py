from keras.models import Model, Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, \
    ConvLSTM2D, TimeDistributed, Flatten, LSTM, Reshape, Convolution3D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import numpy as np
import glob
from scipy import misc

data_dir = '../../data/processed/'

def get_unet():

    kernels = 64
    seq = Sequential()

    seq.add(ConvLSTM2D(nb_filter=32, nb_row=3, nb_col=3,
                       input_shape=(None, 128, 128, 3),
                       border_mode='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(Activation('relu'))
    seq.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))

    seq.add(ConvLSTM2D(nb_filter=32, nb_row=3, nb_col=3,
                       input_shape=(None, 128, 128, 3),
                       border_mode='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(Activation('relu'))
    seq.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))

    seq.add(ConvLSTM2D(nb_filter=32, nb_row=3, nb_col=3,
                       input_shape=(None, 128, 128, 3),
                       border_mode='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(Activation('relu'))

    seq.add(TimeDistributed(UpSampling2D(size=(2,2))))
    seq.add(ConvLSTM2D(nb_filter=32, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(Activation('relu'))

    seq.add(TimeDistributed(UpSampling2D(size=(2, 2))))
    seq.add(ConvLSTM2D(nb_filter=32, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(Activation('relu'))

    seq.add(Convolution3D(nb_filter=1, kernel_dim1=1, kernel_dim2=3,
                          kernel_dim3=3, activation='sigmoid',
                          border_mode='same', dim_ordering='tf'))

    print(seq.output_shape)




    seq.compile(optimizer=Adam(), loss="binary_crossentropy")

    return seq


def get_data():
    x_train = np.load('../../data/split/x_train.npy')
    y_train = np.load('../../data/split/y_train.npy')



    return x_train,y_train


if __name__ == '__main__':
    x_train, y_train = get_data()



    model = get_unet()
    model_checkpoint = ModelCheckpoint('lstm_unet.hdf5', monitor='loss', save_best_only=True)
    model.fit(x_train,y_train, batch_size=16, verbose=1,nb_epoch=1000, callbacks=[model_checkpoint], shuffle=True)

    model.save('../../models/lstm')
