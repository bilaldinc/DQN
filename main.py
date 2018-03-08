import gym
from DQN import DQN

MAX_STEP = 50000000

experience_pool_size = 1000000
update_frequency = 4
gamma = 0.99
epsilon_start = 1.0
epsilon_min = 0.1
final_exploration = 1000000
learning_rate = 0.00025
batch_size = 32
target_network_update_frequency = 10000
replay_start_size = 50000
do_nothing_actions = 6
save_network_frequency = 300000 # 100,000 ~= 1 hour
environment =  gym.make('BreakoutDeterministic-v4')


agent = DQN(environment,experience_pool_size, update_frequency, gamma,epsilon_start,
    epsilon_min, final_exploration,learning_rate, batch_size,target_network_update_frequency,
    replay_start_size, do_nothing_actions,save_network_frequency)

agent.learn(MAX_STEP)

# agent.load("weights")
# agent.play(100,0.5,0.02)
