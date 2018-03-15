"""
Bilal DINC 150113008
Selen PARLAR 150113049
2018
"""

import random
import gym
import numpy as np
import sys
import time
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# ANN model and preprocessing described in the  "Human-level control through deep reinforcement learning"

class Cart_Pole_Model:
    def __init__(self, learning_rate, action_size):
        self.action_size = action_size
        self.learning_rate = learning_rate

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=4, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self.huber_loss, optimizer=Adam(lr=self.learning_rate))

        model.summary()
        return model


    def preprocess(self, state, reward, done, last_k_history):
        # clip rewards -1 to 1
        reward = reward if not done else -10

        # dimension adjust
        state = np.expand_dims(state, axis=0)

        return state,reward

    def huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
