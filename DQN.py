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
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

ATARI_SHAPE = (105, 80, 4)

# TO do list

class DQN:
    def __init__(self,environment,experience_pool_size, update_frequency, gamma,epsilon_start,
        epsilon_min, final_exploration,learning_rate, batch_size,target_network_update_frequency,
        replay_start_size, do_nothing_actions,save_network_frequency):

        self.experience_pool = deque(maxlen=experience_pool_size)
        self.update_frequency = update_frequency
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.final_exploration = final_exploration
        self.learning_rate = learning_rate
        self.do_nothing_actions = do_nothing_actions
        self.batch_size = batch_size
        self.target_network_update_frequency = target_network_update_frequency
        self.replay_start_size = replay_start_size
        self.save_network_frequence = save_network_frequency
        self.last_k_history = deque(maxlen=4)

        self.environment = environment
        self.state_size = environment.observation_space.shape[0]
        self.action_size = environment.action_space.n

        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (self.final_exploration - self.replay_start_size)
        self.prediction_model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.prediction_model.get_weights())

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), padding='same', activation='relu', input_shape=ATARI_SHAPE, strides=(4, 4)))
        model.add(Conv2D(64, (4, 4), padding='same', activation='relu', strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', strides=(1, 1)))
        model.add(Flatten()) # converts vectors to one dimension.
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        rmsprop = keras.optimizers.RMSprop(lr=self.learning_rate,rho=0.95, epsilon=0.01)
        model.compile(loss=self.huber_loss, optimizer=rmsprop)

        return model

    def huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def load(self, name):
        self.prediction_model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        self.prediction_model.save_weights(name)

    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
             # act random
            return random.randrange(self.action_size)
        # Predict the best action values
        act_values = self.prediction_model.predict(state)
        # return index of best action
        return np.argmax(act_values[0])

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

    def replay(self, batch_size):
        # Sample minibatch from the experience pool
        minibatch = random.sample(self.experience_pool, batch_size)

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
        self.prediction_model.fit(minibatch_inputs[1:,...], prediction_model_state_predictions, epochs=1, verbose=0)

    def learn(self, max_step):
        total_steps = 1;
        total_episode = 1;

        while total_steps < max_step:
            # reset state in the beginning of each game
            state = self.environment.reset()
            self.last_k_history.clear()
            state, reward = self.preprocess(state, 0)
            done = False
            totalreward = 0
            step_in_episode = 1

            # every time step
            while not done:
                self.environment.render()

                # Decide action
                if step_in_episode < self.do_nothing_actions:
                    # do not move first k move every episode
                    action = 0;
                else:
                    if total_steps < self.replay_start_size:
                        action = random.randrange(self.action_size)
                    else:
                        action = self.act(state,self.epsilon)

                # Apply action
                next_state, reward, done, _ =  self.environment.step(action)
                totalreward += reward

                # Preprocess state and reward
                next_state, reward = self.preprocess(next_state, reward)

                # save to the experience pool the previous state, action, reward, and done
                self.experience_pool.append((state, action, reward, next_state, done))

                # make next_state the new current state for the next frame.
                state = next_state

                if (total_steps >= self.replay_start_size) and (len(self.experience_pool) > self.batch_size) and (total_steps % self.update_frequency == 0):
                    # Perform SGD
                    self.replay(self.batch_size)

                # Update target model
                if (total_steps % self.target_network_update_frequency) == 0:
                    self.target_model.set_weights(self.prediction_model.get_weights())
                    print("target model is updated")

                # save model
                if (total_steps % self.save_network_frequence) == 0:
                    self.save("network_weights_" + str(total_steps))
                    print("network is saved to the file network_weights_" + str(total_steps))

                # linear epsilon decay
                if (total_steps >= self.replay_start_size) and (self.epsilon > self.epsilon_min):
                    self.epsilon -= self.epsilon_decay

                step_in_episode += 1
                total_steps += 1

            print("episode: " + str(total_episode) + " total_reward:" + str(totalreward) + " total_steps:" + str(total_steps) + " epsilon:" + str(self.epsilon))
            total_episode += 1

        # save model
        self.save("network_weights_final" + str(total_steps))

    def play(self,max_episode,epsilon,wait):
        total_episode = 1;

        while total_episode < max_episode:
            # reset state in the beginning of each game
            state =  self.environment.reset()
            self.last_k_history.clear()
            state, reward = self.preprocess(state, 0)
            done = False
            totalreward = 0

            # every time step
            while not done:
                self.environment.render()

                # Decide action
                action = self.act(state,epsilon)

                # Apply action
                next_state, reward, done, _ =  self.environment.step(action)
                totalreward += reward

                # Preprocess state and reward
                next_state, reward = self.preprocess(next_state, reward)

                # wait
                time.sleep(wait)


            print("episode: " + str(total_episode) + " reward:" + str(totalreward))
            total_episode += 1
