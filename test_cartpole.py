import gym
from DQN import DQN
from Cart_Pole_Model import Cart_Pole_Model


MAX_STEP = 50000000
experience_pool_size = 2000
update_frequency = 50 # 10
gamma = 0.95
epsilon_start = 1.0
epsilon_min = 0.1
final_exploration = 225
learning_rate = 0.001
batch_size = 32
target_network_update_frequency = 50
replay_start_size = 1
do_nothing_actions = 1
save_network_frequency = 300000 # 100,000 ~= 1 hour
last_k_history = 1
environment = gym.make('CartPole-v1')
state_size = environment.observation_space.shape[0]
action_size = environment.action_space.n



cart_pole_model = Cart_Pole_Model(learning_rate, action_size)

agent = DQN(environment,experience_pool_size, update_frequency, gamma,epsilon_start,
    epsilon_min, final_exploration, batch_size,target_network_update_frequency,
    replay_start_size, do_nothing_actions,save_network_frequency,last_k_history,cart_pole_model.preprocess, cart_pole_model.build_model(),cart_pole_model.build_model())

agent.load("cartpole-ddqn.h5")
agent.learn(MAX_STEP)

# agent.play(100,0,0.02)
