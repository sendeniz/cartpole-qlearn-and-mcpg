import numpy as np
import torch
import gym
from matplotlib import pyplot as plt
import time
from IPython.display import clear_output

def running_mean(x):
    x = np.array(x)
    N=50
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y


learning_rate = 0.003
gamma = 0.85
HIDDEN_SIZE = 128
hidden2 = True # Controls whether there is a second hidden layer (with same amount of hidden nodes as h1)

env = gym.make('CartPole-v1')

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

# Generate model
if hidden2:
    model = torch.nn.Sequential(
        torch.nn.Linear(obs_size, HIDDEN_SIZE),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_SIZE, n_actions),
        torch.nn.Softmax(dim=0))
else:
    model = torch.nn.Sequential(
        torch.nn.Linear(obs_size, HIDDEN_SIZE),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_SIZE, n_actions),
        torch.nn.Softmax(dim=0))

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
Horizon = 500
MAX_TRAJECTORIES = 1000
score = []
start_time = time.time()
freeze = False
freeze_episode = 0

for trajectory in range(MAX_TRAJECTORIES):
    if trajectory%(MAX_TRAJECTORIES/10) == 0:
        print('{}% done'.format(trajectory/MAX_TRAJECTORIES*100))
        clear_output(wait=True)

    
    curr_state = env.reset()
    done = False
    transitions = []

    for t in range(Horizon):
        act_prob = model(torch.from_numpy(curr_state).float())
        action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy())
        prev_state = curr_state
        curr_state, _, done, info = env.step(action)
        transitions.append((prev_state, action, t+1))

        if done:
            break

    score.append(len(transitions))
    
    if score[-10:] == [500]*10:
        if freeze == False:
            print("Freezing model")
            freeze_episode = trajectory
            freeze = True
    
    if not freeze:
        reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,))

        batch_Gvals =[]
        for i in range(len(transitions)):
            new_Gval=0
            power=0

            for j in range(i,len(transitions)):
                new_Gval=new_Gval+ ((gamma**power)*reward_batch[j]).numpy()
                power+=1

            batch_Gvals.append(new_Gval)

        expected_returns_batch=torch.FloatTensor(batch_Gvals)
        expected_returns_batch /= expected_returns_batch.max()

        state_batch = torch.Tensor([s for (s,a,r) in transitions])
        action_batch = torch.Tensor([a for (s,a,r) in transitions])

        pred_batch = model(state_batch)
        prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze()

        loss= -torch.sum(torch.log(prob_batch)*expected_returns_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

end_time = time.time()
print("The model took {} seconds to run 1000 episodes".format(end_time-start_time))

print('Trajectory {}\tAverage Score: {:.2f}'.format(trajectory, np.mean(score[-50:-1])))
avg_score = running_mean(score)
plt.figure(figsize=(15,7))
plt.title(r'Monte Carlo Policy Gradient'+'\n'+r'Discount factor $\gamma$={}, learning rate $\alpha$={},'.format(gamma, learning_rate)+'\n'+r'{} hidden nodes, hidden_2 = {}'.format(HIDDEN_SIZE, hidden2))
plt.ylabel("Trajectory Duration",fontsize=12)
plt.xlabel("Training Epochs",fontsize=12)
plt.plot(score, color='gray' , linewidth=1)
plt.plot(avg_score, color='blue', linewidth=3)
if freeze_episode != 0:
    plt.vlines(freeze_episode, 0, 500, linestyle='--', color='r', label='Freezing moment')
plt.scatter(np.arange(np.array(score).shape[0]),score, color='green' , linewidth=0.3)
plt.legend()
plt.show()