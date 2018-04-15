"""
Bilal DINC 150113008
Selen PARLAR 150113049
2018

Class for ANN model and preprocessing described in the  "Human-level control through deep reinforcement learning".
"""


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import keras
from keras.layers import Conv2D, Flatten, Lambda
import scipy.misc
import tensorflow as tf

HUBER_LOSS_DELTA = 1.0

class Atari_Model:
    def __init__(self, learning_rate, action_size):
        self.action_size = action_size
        self.learning_rate = learning_rate
#%%
    def build_model(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0, input_shape=(84, 84, 4)))
        model.add(Conv2D(32, (8, 8), padding='same', activation='relu', strides=(4, 4)))
        model.add(Conv2D(64, (4, 4), padding='same', activation='relu', strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', strides=(1, 1)))
        model.add(Flatten()) # converts vectors to one dimension.
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        rmsprop = keras.optimizers.RMSprop(lr=self.learning_rate,rho=0.95, epsilon=0.01)
        model.compile(loss=self.huber_loss, optimizer=rmsprop)

        model.summary()
        return model

#%%
    def preprocess(self, state, reward, done, last_k_history):
        # clip rewards -1 to 1
        if reward < 0:
            reward = -1
        elif reward > 0:
            reward = 1
        else:
            reward = 0

        # to grayscale
        state = np.mean(state, axis=2).astype(np.uint8)
        # between [0 1]
        # state = np.divide(state, 255.0).astype(np.float16)
        # down sample to 105x80
        # state = state[::2, ::2]
        state = scipy.misc.imresize(state, (84,84)).astype(np.uint8)

        # concatenate with last 4 history
        last_k_history.append(state)
        temp_list = list(last_k_history)

        if len(temp_list) == 1:
            state = np.stack((temp_list[0],temp_list[0],temp_list[0],temp_list[0]), axis=2)
        elif len(temp_list) == 2:
            state = np.stack((temp_list[0],temp_list[0],temp_list[0],temp_list[1]), axis=2)
        elif len(temp_list) == 3:
            state = np.stack((temp_list[0],temp_list[0],temp_list[1],temp_list[2]), axis=2)
        else:
            state = np.stack((temp_list[0],temp_list[1],temp_list[2],temp_list[3]), axis=2)

        # dimension adjust
        # state = np.expand_dims(state, axis=0)

        return state,reward
#%%

    def huber_loss(self, y_true, y_pred):
        err = y_true - y_pred
    
        cond = K.abs(err) < HUBER_LOSS_DELTA
        L2 = 0.5 * K.square(err)
        L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)
    
        loss = tf.where(cond, L2, L1)
    
        return K.mean(loss)

