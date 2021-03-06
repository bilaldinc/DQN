"""
Bilal DINC 150113008
Selen PARLAR 150113049
2018

Class for testing any atari model.
"""

import gym
from DQN import DQN
from Atari_Model import Atari_Model
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

MAX_STEP = 50000000 
experience_pool_size = 1000000
update_frequency = 4
gamma = 0.99
epsilon_start = 1.0
epsilon_min = 0.1
final_exploration = 1000000 #1000000
learning_rate = 0.00025
batch_size = 32
target_network_update_frequency = 10000
replay_start_size = 3000
do_nothing_actions = 30
save_network_frequency = 300000 # 100,000 ~= 1 hour
last_k_history = 4
action_repeat = 4
consecutive_max = True
eval_freq = 100

# environment =  gym.make('BreakoutNoFrameskip-v4')
environment =  gym.make('BreakoutDeterministic-v4')
state_size = environment.observation_space.shape[0]
action_size = environment.action_space.n
file_name = "atari_cont"

atari_model = Atari_Model(learning_rate, action_size)

agent = DQN(environment,experience_pool_size, update_frequency, gamma,epsilon_start,
    epsilon_min, final_exploration, batch_size,target_network_update_frequency,
    replay_start_size, do_nothing_actions,save_network_frequency,last_k_history,
    atari_model.preprocess, atari_model.build_model(),atari_model.build_model(), file_name,action_repeat, consecutive_max,eval_freq)

# agent.load("atari_new_cont_network_weights_6000000")
# agent.load_all("atari_cont_network_weights", "atari_contexp_pool.pkl")
agent.learn(MAX_STEP)
#agent.play(100,0,0.02)
#agent.save("atari_new_cont_network_weights_6000000",agent.prediction_model)

#agent.load("atari_new_cont_network_weights_6000000.json", "atari_new_cont_network_weights_6000000.h5")
