import gym
import numpy as np
import math
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')
# Prepare the Q Table
# create one bucket for each of the 4 feature
# features are : [cart_position, cart_velocity, pole_position, pole_volecity]
# the first two features are less important and coded as 1
n_buckets = (1, 1, 6, 3)

# define the number of actions = 2; [left_move, right_move]
n_actions = env.action_space.n

# define the limits of the state space aslower and upper bound
l_bound = env.observation_space.low
u_bound = env.observation_space.high

state_bounds = np.column_stack((l_bound, u_bound))
state_bounds[1] = [-0.5, 0.5]
state_bounds[3] = [-np.radians(50), np.radians(50)]

# define Q table to store each [state,action] pairs' value
q_table = np.zeros(n_buckets + (n_actions,))

# define a learning schedule for random action parameter epsilon and learning rate
# values decrease over time
# the values have been floored: min is 1 and 0.5 respectively
# and bound at upper 0.01 for both
def learning_schedule(i, use_schedule):
    use_schedule = use_schedule
    if use_schedule == 0:
        epsilon = max(0.01, min( 1, 1.0 - np.log10(( i+1 ) / 25 )))
        learning_rate = max(0.01, min( 0.5, 1.0 - np.log10(( i+1 ) / 25 )))
    elif use_schedule == 1:
        epsilon = 1
        learning_rate = max(0.01, min( 0.5, 1.0 - np.log10(( i+1 ) / 25 )))
    elif use_schedule == 2:
        epsilon = 0.5
        learning_rate = max(0.01, min( 0.5, 1.0 - np.log10(( i+1 ) / 25 )))
    elif use_schedule == 3:
        epsilon = 0.1
        learning_rate = max(0.01, min( 0.5, 1.0 - np.log10(( i+1 ) / 25 )))
    return(epsilon, learning_rate, use_schedule)

# reward discount factor gamma
gamma = 0.99 

# initialize list for results for schedule learning on and off
learning_schedule_on = []
learning_schedule_off_1 = []
learning_schedule_off_2 = []
learning_schedule_off_3 = []

def pick_action(state, q_table, action_space, epsilon):
    # chance that a random action will be chosen
    if np.random.random_sample() < epsilon: 
        return action_space.sample() 
    # select the action based on the existing policy, that is, in the 
    # current state in the Q table, select the action with the largest Q value
    else: 
        return np.argmax(q_table[state]) 

def get_discrete_state(observation, n_buckets, state_bounds):
    # initalize state as length of observation vector
    state = np.zeros(len(observation), dtype = int) 
    for i, s in enumerate(observation):
    # lower bound, upper bound of each feature value
        lower = state_bounds[i][0]
        upper = state_bounds[i][1] 
        # below the lower bound or equal to assign as zero
        if s <= lower: 
            state[i] = 0
        # if its higher or equal to the upper bound assign max
        elif s >= upper: 
            state[i] = n_buckets[i] - 1
        # if within the bounds use a proportional distribution
        else: 
            state[i] = int(((s - lower) / (upper - lower)) * n_buckets[i])
    return tuple(state)

# Q-learning
n_episodes = 250
n_time_steps = 200
# initalize boolian to use learning schedule or not
schedule_switch = [0, 1, 2, 3]

for on_off in range(len(schedule_switch)):
    use_schedule = schedule_switch[on_off]
    for i_episode in range(n_episodes):
        epsilon = learning_schedule(i_episode, use_schedule)[0]
        learning_rate = learning_schedule(i_episode, use_schedule)[1]
        use_schedule = learning_schedule(i_episode, use_schedule)[2]
        observation = env.reset()
        rewards = 0
        # convert continious values to discrete values
        state = get_discrete_state(observation, n_buckets, state_bounds) 
        for t in range(n_time_steps):
            #env.render()

            action = pick_action(state, q_table, env.action_space, epsilon)
            observation, reward, done, info = env.step(action)

            rewards += reward
            next_state = get_discrete_state(observation, n_buckets, state_bounds)

            # update Q table
            # compute a given states next q value
            q_next_max = np.amax(q_table[next_state]) 
            # update Q values using update rule/equation
            q_table[state + (action,)] += learning_rate * (reward + gamma * q_next_max - q_table[state + (action,)])

            # move to next state
            state = next_state
            if done and use_schedule == 0:
                learning_schedule_on.append(rewards)
                print('Learning Schedule On: Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
                break
            elif done and use_schedule == 1:
                learning_schedule_off_1.append(rewards)
                print('Learning Schedule Off 1: Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
                break
            elif done and use_schedule == 2:
                learning_schedule_off_2.append(rewards)
                print('Learning Schedule Off 2: Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
                break
            elif done and use_schedule == 3:
                learning_schedule_off_3.append(rewards)
                print('Learning Schedule Off 3: Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
                break
    env.close()

plt.figure(figsize=(8, 6))
plt.plot(learning_schedule_on, linewidth=2)
plt.plot(learning_schedule_off_1, linewidth=2)
plt.plot(learning_schedule_off_2, linewidth=2)
plt.plot(learning_schedule_off_3, linewidth=2)
plt.title('Total Reward after each Episode', fontsize = 20)
plt.xlabel('Number of Episodes', fontsize = 18)
plt.ylabel('Total Reward', fontsize = 18)
plt.legend(['Model 1 ', 'Model 2 ', 'Model 3 ', 'Model 4', 'Model 5' ], 
            prop={'size': 18}, frameon=False)
plt.show()

