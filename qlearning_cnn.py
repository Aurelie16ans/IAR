from gridworld import GridWorld
import collections
import random
import torch
import torch.nn as nn
import tqdm
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F

MEM_SIZE = 1000
T_MAX = 50
HIDDEN_DIM = 5
MAX_ITER = 10000
BATCH_SIZE = 32
DISCOUNT = 0.9
LEARNING_RATE = 0.0001
FREEZE_PERIOD = 100 # epoch


class Perceptron(nn.Module):
    def __init__(self,observation_space_size, hidden_dim, action_space_size):
        super().__init__()
        self.in_to_hidden = nn.Linear(observation_space_size, hidden_dim)
        self.hidden_to_out = nn.Linear(hidden_dim, action_space_size)

    def forward(self,observation):
        h = self.in_to_hidden(observation)
        h = nn.functional.relu(h)
        return self.hidden_to_out(h)

class CNNModel(nn.Module):

    def __init__(self,observation_space_size, hidden_dim, action_space_size, img_width, img_height, second_hidden_dim=16, third_hidden_dim = 120, conv_size=3):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim, conv_size)
        self.conv2 = nn.Conv2d(hidden_dim, second_hidden_dim, conv_size)
        self.fc1 = nn.Linear(img_width * img_height * second_hidden_dim, third_hidden_dim)
        self.fc2 = nn.Linear(third_hidden_dim, action_space_size)
        self.img_size = img_width * img_height
        self.scnd_dim = second_hidden_dim

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.img_size * self.scnd_dim)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

width = 4
height = 4
g=GridWorld(width,height)
g.add_start(1,1)
g.add_goal(3,3)
n_s = g.state_space_size
n_a = g.action_space_size
s = g.reset()

#Q_value = Perceptron(1, HIDDEN_DIM, n_a)
#target_Q_value = Perceptron(1, HIDDEN_DIM, n_a)
Q_value = CNNModel(1, HIDDEN_DIM, n_a, width+2, height+2)
target_Q_value = CNNModel(1, HIDDEN_DIM, n_a, width+2, height+2)
target_Q_value.load_state_dict(Q_value.state_dict()) # copie des param

optim = torch.optim.SGD(Q_value.parameters(),lr=LEARNING_RATE)


def calculate_epsilon(it):
    return (-np.arctan(20*(it-MAX_ITER/2)/MAX_ITER)+1.5)/3
    
def sample_action(env, z, Q_value, epsilon):
    print("z ",z)
    if random.random() < epsilon:
        return random.randint(0,n_a-1)
    else:
        return Q_value(torch.tensor(z, dtype=torch.float).unsqueeze(0)).argmax().item() # renvoit une action aleatoire


def sample_trajectory(replay_memory, Q_value, epsilon):
    z = g.reset() #observation
    cumul = 0
    for t in range(T_MAX):
        a = sample_action(g, z, Q_value, epsilon)
        next_z, r, done = g.step(a)
        cumul += r
        replay_memory.append((z,a,r,next_z,done))
        z=next_z
        if done:
            break
    return cumul

def train():
    replay_memory = collections.deque(maxlen=MEM_SIZE)
    epsilon = 1
    plot_tot_loss = []

    cumul_reward_epoch = 0
    average_reward_epoch = []
    cumul = 0
    with tqdm.trange(MAX_ITER) as progress_bar:
        for it in progress_bar:
            cumul = cumul+sample_trajectory(replay_memory, Q_value, epsilon)
            n = len(replay_memory)

            if  n >BATCH_SIZE:
                indices = list(range(n))
                random.shuffle(indices)
                tot_loss = 0
                for b in range(n // BATCH_SIZE):
                    batch_z, batch_a, batch_r, batch_nxt, batch_done = zip(
                            *(replay_memory[i] for i in indices[b * BATCH_SIZE:(b+1) * BATCH_SIZE]))

                    batch_z = torch.tensor(batch_z).float().unsqueeze(1)
                    batch_a = torch.tensor(batch_a).unsqueeze(1)
                    batch_r = torch.tensor(batch_r).unsqueeze(1).float()
                    batch_nxt = torch.tensor(batch_nxt).unsqueeze(1).float()
                    batch_done = torch.tensor(batch_done).unsqueeze(1)

                    batch_target = batch_r + DISCOUNT*target_Q_value(batch_nxt).max(1, keepdim=True)[0]
                    batch_target[batch_done] = 0

                    batch_qval = Q_value(batch_z).gather(1, batch_a)
                    loss = nn.functional.mse_loss(batch_qval, batch_target.detach())
                    tot_loss += loss.item()

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                cumul_reward_epoch += sample_trajectory([], Q_value, 0)
                plot_tot_loss.append(tot_loss / (n // BATCH_SIZE))
                progress_bar.set_postfix(loss = tot_loss / (n // BATCH_SIZE), cumul=cumul)

            if it % FREEZE_PERIOD == FREEZE_PERIOD - 1:
                average_reward_epoch.append(cumul_reward_epoch/FREEZE_PERIOD)
                cumul_reward_epoch = 0
                temp = target_Q_value.state_dict()
                target_Q_value.load_state_dict(Q_value.state_dict()) # copie des param
                Q_value.load_state_dict(temp)

            epsilon = calculate_epsilon(it)

    return plot_tot_loss, average_reward_epoch



# training
plot_tot_loss, average_reward_epoch = train()

# plot loss
plt.figure()
plt.plot(plot_tot_loss)
plt.savefig("Loss_neural_network.png")

# plot average_reward
plt.figure()
plt.plot(average_reward_epoch)
plt.savefig("average_reward_per_epoch.png")