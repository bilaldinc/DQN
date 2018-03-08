import random
import gym
import numpy as np
import sys
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

EPISODES = 5000
ATARI_SHAPE = (105, 80, 4)
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
# TO do list

# add linear epsilon decay OKKKK
# add update frequency parameter (this should speed up 4 times) OKKKK
# add no-op max parameter OKKK
# check complexity of numpy operations. maybe extending dimension everytime is costly? OKKK (they are not)
# also random acces queue does not change much OKKK

# tesorflow gpu nun hepsini kullaniyor mu? okk
# number of prediction by mask?(en son)
# look minibatch update okkkk

# get parameters as an argumant
# get history_size as an argumant
# get model as an argumant
# train based on total step not episode

# chekk fit again make predictions at ones maybe it can be speed-up
# epsilon_decrease is in wrong place

class DQN:
    def __init__(self,env):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.env = env
        self.memory = deque(maxlen=1000000)
        self.last_k_history = deque(maxlen=4)
        self.update_frequency = 4
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.final_exploration = 1000000
        self.learning_rate = 0.00025
        self.no_op_max = 6 # maybe 30
        self.batch_size = 32
        self.C_steps = 10000 # target network update frequency
        self.replay_start_size = 50 # before learning starts play randomly SHOULD BE 50000
        self.save_network_frequence = 2000000
        self.prediction_model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()

        model.add(Conv2D(32, (8, 8), padding='same', activation='relu', input_shape=ATARI_SHAPE, strides=(4, 4)))
        model.add(Conv2D(64, (4, 4), padding='same', activation='relu', strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', strides=(1, 1)))
        model.add(Flatten()) # converts vectors to one dimension. I am not sure this is the right place (other peaple did after last conv layer)
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        rmsprop = keras.optimizers.RMSprop(lr=self.learning_rate,rho=0.95, epsilon=0.01)
        model.compile(loss=self.huber_loss, optimizer=rmsprop) # loss function ?? maybe try with mse

        # model.summary()
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.prediction_model.get_weights())

    def save_to_experience_pool(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def load(self, name):
        self.prediction_model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        self.prediction_model.save_weights(name)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
             # act random
            return random.randrange(self.action_size)
        # Predict the best action values
        act_values = self.prediction_model.predict(state)
        # return index of best action
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        # Sample minibatch from the memory
        minibatch = random.sample(self.memory, batch_size)

        #---------------------------------------------------------
        # prepare inputs for prediction
        state_batch = np.zeros(minibatch[0][0].shape)
        next_state_batch = np.zeros(minibatch[0][0].shape)
        for state, action, reward, next_state, done in minibatch:
            state_batch = np.append(state_batch, state,axis=0)
            next_state_batch = np.append(next_state_batch, state, axis=0)

        # predict necessary informations to produce targets
        prediction_model_state_predictions = self.prediction_model.predict(state_batch[1:, ...])
        target_model_next_state_predictions = self.target_model.predict(next_state_batch[1:, ...])

        #---------------------------------------------------------
        # prepare targets and inputs for trainning
        minibatch_targets = prediction_model_state_predictions
        minibatch_inputs = np.zeros(minibatch[0][0].shape)
        for index, sample in zip(range(batch_size),minibatch):
            state, action, reward, next_state, done = sample

            # append input state
            minibatch_inputs = np.append(minibatch_inputs,state,axis=0)

            # calculate target
            target = reward
            if not done:
                # target should produced from target model
                target = (reward + self.gamma * np.amax(target_model_next_state_predictions[index]))

            # replace desired action with target action
            prediction_model_state_predictions[index][action] = target


        # train minibatch
        self.prediction_model.fit(minibatch_inputs[1:,...], minibatch_targets, epochs=1, verbose=0)


        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon - self.epsilon_min) / self.final_exploration


    def huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def preprocess(self, state, reward):
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
        state = np.divide(state, 255.0).astype(np.float16)
        # down sample to 105x80
        state = state[::2, ::2]

        # concatenate with last 4 history
        self.last_k_history.append(state)
        temp_list = list(self.last_k_history)

        if len(temp_list) == 1:
            state = np.stack((temp_list[0],temp_list[0],temp_list[0],temp_list[0]), axis=2)
        elif len(temp_list) == 2:
            state = np.stack((temp_list[1],temp_list[0],temp_list[0],temp_list[0]), axis=2)
        elif len(temp_list) == 3:
            state = np.stack((temp_list[2],temp_list[1],temp_list[0],temp_list[0]), axis=2)
        else:
            state = np.stack((temp_list[3],temp_list[2],temp_list[1],temp_list[0]), axis=2)

        # dimension adjust
        state = np.expand_dims(state, axis=0)

        return state,reward

    def learn(self, max_step):
        total_steps = 0;
        total_episode = 0;
        # self.load("network_weights_13")

        while total_steps < max_step:
            # reset state in the beginning of each game
            state = env.reset()
            self.last_k_history.clear()
            state, reward = self.preprocess(state, 0)
            done = False
            totalreward = 0
            step_in_episode = 0

            # every time step
            while True:
                env.render()
                # Decide action
                if step_in_episode < self.no_op_max:
                    # do not move first k move every episode
                    action = 0;
                else:
                    if total_steps < self.replay_start_size:
                        action = random.randrange(self.action_size)
                    else:
                        action = self.act(state)

                # Apply action
                next_state, reward, done, _ = env.step(action)
                totalreward += reward
                # print( "episode:" + str(e) + " step:" + str(step_in_episode) + " reward:" + str(reward))

                # Preprocess state and reward
                next_state, reward = self.preprocess(next_state, reward)

                # save to the experience pool the previous state, action, reward, and done
                self.save_to_experience_pool(state, action, reward, next_state, done)

                # make next_state the new current state for the next frame.
                state = next_state

                if (total_steps >= self.replay_start_size) and (len(self.memory) > self.batch_size) and (total_steps % self.update_frequency == 0):
                    # Perform SGD
                    self.replay(self.batch_size)

                # Update target model
                if (total_steps % self.C_steps) == 0:
                    self.update_target_model()

                # save model
                if (total_steps % self.save_network_frequence) == 0:
                    self.save("network_weights_" + str(total_steps))

                step_in_episode += 1
                total_steps += 1
                if done:
                    print("episode: " + str(total_episode) + " total_reward:" + str(totalreward) + " total_steps:" + str(total_steps))
                    break

            total_episode += 1

        # save model
        self.save("network_weights_final" + str(total_steps))


env = gym.make('BreakoutDeterministic-v4')
agent1 = DQN(env)
agent1.learn(10000000)
