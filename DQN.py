"""
Bilal DINC 150113008
Selen PARLAR 150113049
2018
"""

import random
import numpy as np
import time
from collections import deque
from keras import backend as K
import keras
import pickle
from keras.models import model_from_json

# TO do list
# control frameskip from bookmark in the home
# log the experiment for plots

class DQN:
    def __init__(self,environment,experience_pool_size, update_frequency, gamma,epsilon_start,
        epsilon_min, final_exploration, batch_size,target_network_update_frequency,
        replay_start_size, do_nothing_actions,save_network_frequency,last_k_history,preprocess,prediction_model,target_model,file_name,action_repeat, consecutive_max):

        self.experience_pool = deque(maxlen=experience_pool_size)
        self.update_frequency = update_frequency
        self.gamma = gamma
        # add step to pickle
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.final_exploration = final_exploration
        self.do_nothing_actions = do_nothing_actions
        self.batch_size = batch_size
        self.target_network_update_frequency = target_network_update_frequency
        self.replay_start_size = replay_start_size
        self.save_network_frequence = save_network_frequency
        self.last_k_history = deque(maxlen=last_k_history)
        self.preprocess = preprocess
        self.file_name = file_name

        self.consecutive_max = consecutive_max
        self.action_repeat = action_repeat

        self.environment = environment
        self.state_size = environment.observation_space.shape[0]
        self.action_size = environment.action_space.n
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (self.final_exploration - self.replay_start_size)

        self.prediction_model = prediction_model
        self.target_model = target_model
        self.target_model.set_weights(self.prediction_model.get_weights())

        # counters
        self.total_episode = 1
        self.total_steps = 1

#%%
    def load1(self, name):
        self.prediction_model.load_weights(name)
        self.target_model.load_weights(name)
#%%
    def load_all1(self, network_weights, exp_pool):
        self.load(network_weights)
        with open(exp_pool, 'rb') as f:
            self.experience_pool, self.total_episode, self.total_steps, self.epsilon = pickle.load(f)
#%%
    def save1(self, name):
        self.prediction_model.save_weights(name)

#%% 
    def load(self, namej, nameh):
        # load json and create model
        json_file = open(namej, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.prediction_model = model_from_json(loaded_model_json)
        self.target_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.prediction_model.load_weights(nameh)       
        self.target_model.load_weights(nameh)
        rmsprop = keras.optimizers.RMSprop(lr=0.00025,rho=0.95, epsilon=0.01)
        self.prediction_model.compile(loss=self.huber_loss, optimizer=rmsprop)
        self.target_model.compile(loss=self.huber_loss, optimizer=rmsprop)


#%%
    def load_all(self, network_weights, exp_pool):
        self.load(network_weights)
        with open(exp_pool, 'rb') as f:
            self.experience_pool, self.total_episode, self.total_steps, self.epsilon = pickle.load(f)
#%%
    def save(self, name, model):
        # serialize model to JSON
        model_json = model.to_json()
        with open(name+ ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(name+"w.h5")
#%%
    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
             # act random
            return random.randrange(self.action_size)
        # Predict the best action values
        act_values = self.prediction_model.predict(state)
        # return index of best action
        return np.argmax(act_values[0])
#%%
    def act(self, action):
        state = None
        reward = 0
        done = False
        total_reward = 0
        for i in range(self.action_repeat):
            previous_state = state
            state, reward, done, _ = self.environment.step(action)
            total_reward += reward
            if done:
                if i == 0 and previous_state is None:
                    previous_state = state
                break

        if self.consecutive_max and self.action_repeat > 1:
            state = np.maximum(state, previous_state)
        return state, total_reward, done, _
#%%
    def random_start(self):
        self.environment.reset()
        for i in range(random.randint(4, self.do_nothing_actions)):
            next_state, reward, done, _ = self.environment.step(0)
            self.environment.render()
            state, reward = self.preprocess(next_state, reward, done, self.last_k_history)

        return state, reward
#%%
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
#%%
    def learn(self, max_step):

        max_reward = 0
        while self.total_steps < max_step:
            # reset state in the beginning of each game
            self.last_k_history.clear()
            # state = self.environment.reset()
            state, reward = self.random_start()
            done = False
            # state, reward = self.preprocess(state, 0, done, self.last_k_history)
            totalreward = 0
            step_in_episode = 1

            # every time step
            while not done:
                self.environment.render()

                # Decide action
                if self.total_steps < self.replay_start_size:
                    action = random.randrange(self.action_size)
                else:
                    action = self.select_action(state,self.epsilon)


                # Apply action
                next_state, reward, done, _ =  self.environment.step(action)
                # next_state, reward, done, _ =  self.act(action)
                totalreward += reward

                # Preprocess state and reward
                next_state, reward = self.preprocess(next_state, done, reward,self.last_k_history)

                # save to the experience pool the previous state, action, reward, and done
                self.experience_pool.append((state, action, reward, next_state, done))

                # make next_state the new current state for the next frame.
                state = next_state

                if (self.total_steps >= self.replay_start_size) and (len(self.experience_pool) > self.batch_size) and (self.total_steps % self.update_frequency == 0):
                    # Perform SGD
                    self.replay(self.batch_size)

                # Update target model
                if (self.total_steps % self.target_network_update_frequency) == 0:
                    # self.target_model = self.copy_model(self.prediction_model)
                    self.target_model.set_weights(self.prediction_model.get_weights())
                    print("Target model is updated")

                # linear epsilon decay
                if (self.total_steps >= self.replay_start_size) and (self.epsilon > self.epsilon_min):
                    self.epsilon -= self.epsilon_decay

                step_in_episode += 1
                self.total_steps += 1

                # save model
                if (self.total_steps % self.save_network_frequence) == 0:
                    self.save(self.file_name + "_network_weights_" + str(self.total_steps),self.prediction_model)
                    print(self.file_name + "_network is saved to the file network_weights_" + str(self.total_steps))
                    with open(self.file_name + 'exp_pool.pkl', 'wb') as f:
                        pickle.dump((self.experience_pool, self.total_episode, self.total_steps, self.epsilon), f)
                    print("exp_pool is saved:")

            print("Episode:" + str(self.total_episode) + " Reward:" + str(totalreward) + " Step:" + str(step_in_episode) + " Total steps:" + str(self.total_steps) + " Epsilon:" + str(self.epsilon))

            if totalreward > max_reward:
                max_reward=totalreward

            if self.total_episode % 100 == 0:
                print("Highest reward in " + str(self.total_episode) + " episodes:" + str(max_reward))
            self.total_episode += 1

        # save model
        self.save(self.file_name + "_network_weights_final" + str(self.total_steps), self.prediction_model)
#%%
    def play(self,max_episode,epsilon,wait):
        total_episode = 1
        total_steps = 1

        while total_episode < max_episode:
            # reset state in the beginning of each game
            self.last_k_history.clear()
            # state = self.environment.reset()
            state, reward = self.random_start()
            done = False
            # state, reward = self.preprocess(state, 0, done, self.last_k_history)
            totalreward = 0
            step_in_episode = 1

            # every time step
            while not done:
                self.environment.render()

                # Decide action
                action = self.select_action(state,epsilon)

                # Apply action
                next_state, reward, done, _ =  self.environment.step(action)
                # next_state, reward, done, _ =  self.act(action)
                totalreward += reward

                # Preprocess state and reward
                next_state, reward = self.preprocess(next_state, reward, done, self.last_k_history)

                # make next_state the new current state for the next frame.
                state = next_state

                # wait
                time.sleep(wait)
                step_in_episode += 1
                total_steps += 1


            print("Episode:" + str(total_episode) + " Reward:" + str(totalreward) + " Step:" + str(step_in_episode))
            total_episode += 1
#%%
    def huber_loss(self, prediction, target):
        error = prediction - target
        MSE = error * error / 2.0
        MAE = abs(error) - 0.5
        condition = (abs(error) > 1.0)
        condition = K.cast(condition, 'float32')
        return condition * MAE + (1-condition) * MSE
