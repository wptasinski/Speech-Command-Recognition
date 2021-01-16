
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')

from functools import partial
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from preprocessing import get_MFCC, normalize_array, invert_encode



def get_model(input_shape, loss="categorical_crossentropy", learning_rate=0.005,num_outputs = 4):
    """Build neural network using keras.
    :param input_shape (tuple): Shape of array representing a sample train. E.g.: (44, 13, 1)
    :param loss (str): Loss function to use
    :param learning_rate (float):
    :return model: TensorFlow model
    """

    # build network architecture using convolutional layers
    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2), padding='same'))

    # flatten output and feed into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    tf.keras.layers.Dropout(0.3)

    # softmax output layer
    model.add(tf.keras.layers.Dense(num_outputs, activation='softmax'))

    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)

    # compile model
    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])

    # print model parameters on console
    # model.summary()

    return model


def init_encoder():
    commands = ['go', 'left', 'right', 'stop']
    label_encoder = LabelEncoder()
    label_encoder.fit(commands)
    return partial(invert_encode, label_encoder=label_encoder)

partial_mffc = partial(get_MFCC,sample_rate=22050,num_mfcc=13, n_fft=2048, hop_length=512, scaled = False)
class SoundCommandClf(object):
    def __init__(self,input_shape=(13, 44, 1), model_path='model.hdf5', recDuration=1, partial_mffc=partial_mffc):
        self.input_shape = input_shape
        self.calculate_mfcc = partial_mffc
        self.encode_pred = init_encoder()
        try:
            self.load_model(model_path)
        except:
            self.model=None
            print('model.hdf5 not found, change model_path or train a new model.')

    def load_model(self,model_path):
        self.model = get_model(input_shape=self.input_shape)
        self.model.load_weights(model_path)
        print('Model {} has been loaded successfully!'.format(model_path))
        
    def train(X_train,y_train,epochs=10,batch_size=64,validation_data=None):
        pass

    def preprocess_rec(self, rec):
        mfcc = self.calculate_mfcc(rec)
        normalized_mfcc = normalize_array(mfcc)
        reshaped_mfcc = normalized_mfcc[np.newaxis,:,:,np.newaxis]
        return reshaped_mfcc

    def classify_rec(self, rec):
        x = self.preprocess_rec(rec)
        y_hat = self.model.predict(x)
        predicted_class = self.encode_pred(y_hat) 
        return predicted_class