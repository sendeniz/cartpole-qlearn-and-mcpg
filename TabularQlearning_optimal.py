
'''    # DoF #
The system has four degrees of freedom: the x-location of the cart, the x-velocity of the cart, the angle the pole makes with the cart, and the rotational velocity of the pole.

This gives one logical way to save game states: a state containing these four variables.

Another option would be to convert the polar coordinates of the pole into one variable, for example the tip of the pole. However, this would result in multiple game state having similar values. After training, this would result in ambiguity between which state to choose, because of these similar values corresponding to very different states.


Observation_space contains four variables in this order: [cart_position, cart_velocity, pole_angle, pole_rotational_velocity]

Action_space contains two possible actions: 0 or 1, corresponding to starting movement in the left resp. right direction. Inbetween values are not an option.
    
'''

'''    # Discretization #
Cart pos/vel not as important as angle, (https://medium.com/swlh/using-q-learning-for-openais-cartpole-v1-4a216ef237df) so we start with 3 discrete values for cart pos/vel, and 5 discrete values for angle pos/vel.


possible positions: range(-4.8, 4.8)

discretized states (pos): np.linspace(-4.8, 4.8, 3)


possible velocities: range($-\infty, \infty$). Linspace from $-\infty$ to $\infty$ doesn't make much sense here, since most high velocities will never be reached. Therefore we cut off at $\pm$0.5 units/s

discretized states (vel): np.linspace(-0.5, 0.5, 3)


possible angles: range(-0.42, 0.42) # units Rad (=$\pm$ 24 deg)

discretized states (ang): np.linspace(-0.42, 0.42, 5)


possible angle velocities: range($-\infty, \infty$). Linspace from $-\infty$ to $\infty$ doesn't make much sense here, since most high velocities will never be reached. Therefore we cut off at $\pm$ 50 deg/s (=0.87 rad/s)

discretized states(ang vel): np.linspace(-0.87, 0.87, 5)
'''

import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

def discretize_state(state, discrete_cart_pos, discrete_cart_vel, discrete_pole_ang, discrete_pole_vel): # Given a certain state, return discretized state values, which is present in Q-table
    cart_pos, cart_vel, pole_ang, pole_vel = state
    disc_cart_pos = discrete_cart_pos[(np.abs(discrete_cart_pos - cart_pos)).argmin()]
    disc_cart_vel = discrete_cart_vel[(np.abs(discrete_cart_vel - cart_vel)).argmin()]
    disc_pole_ang = discrete_pole_ang[(np.abs(discrete_pole_ang - pole_ang)).argmin()]
    disc_pole_vel = discrete_pole_vel[(np.abs(discrete_pole_vel - pole_vel)).argmin()]
    discrete_state = [disc_cart_pos, disc_cart_vel, disc_pole_ang, disc_pole_vel]
    return discrete_state

def tabular_Q_learning(environment='CartPole-v0', n_epochs=250, buckets=[1,1,6,3], gamma=0.99, alpha=0.2, epsilon=0.1, cart_vel_threshold=0.5, pole_vel_threshold=0.87):
    env = gym.make(environment)

    discrete_cart_pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], buckets[0]) # 4.8 units
    discrete_cart_vel = np.linspace(-cart_vel_threshold, cart_vel_threshold, buckets[1]) # 0.5 units/s
    discrete_pole_ang = np.linspace(env.observation_space.low[2], env.observation_space.high[2], buckets[2]) # 24 deg
    discrete_pole_vel = np.linspace(-pole_vel_threshold, pole_vel_threshold, buckets[3]) # 50 deg/s
    
    # These two arrays make up the Q-table
    all_possible_states = np.array(np.meshgrid(discrete_cart_pos, discrete_cart_vel, discrete_pole_ang, discrete_pole_vel)).T.reshape(-1,4)
    Q_values = np.random.uniform(0, 1, ((len(all_possible_states), 2))) # Initialize randomly to encourage random moves at the start
    
    all_rewards = []
    for epoch in range(n_epochs):
        done = False
        total_reward = 0
        state = env.reset()
        epsilon = max(0.01, min( 1, 1.0 - np.log10(( epoch+1 ) / 25 )))
        alpha = max(0.01, min( 0.5, 1.0 - np.log10(( epoch+1 ) / 25 )))
        while not done:
            #env.render()
            disc_state = discretize_state(state, discrete_cart_pos, discrete_cart_vel, discrete_pole_ang, discrete_pole_vel)
            idx = np.where((all_possible_states==disc_state).sum(axis=1)==4)[0][0]
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore state space
            else:
                action = np.argmax(Q_values[idx]) # Exploit learned values
            next_state, reward, done, info = env.step(action) # invoke Gym
            disc_next_state = discretize_state(next_state, discrete_cart_pos, discrete_cart_vel, discrete_pole_ang, discrete_pole_vel)
            idx_next = np.where((all_possible_states==disc_next_state).sum(axis=1)==4)[0][0]
            next_max = np.max(Q_values[idx_next])
            old_value = Q_values[idx][action]
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            Q_values[idx][action] = new_value
            total_reward += reward
            state = next_state
        all_rewards.append(total_reward)
    #    if epoch % (n_epochs/10) == 0 or epoch==n_epochs-1:
    print("Epoch {} Total Reward: {}".format(epoch, total_reward))
    plt.plot(np.arange(epoch+1), all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.show()

    env.close()


tabular_Q_learning(environment='CartPole-v1', n_epochs=250, buckets=[1, 1, 6, 3])