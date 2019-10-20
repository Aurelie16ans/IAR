from gridworld_maj import GridWorld
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
MAX_ITER = 500
BATCH_SIZE = 32
DISCOUNT = 0.9
LEARNING_RATE = 0.0001
FREEZE_PERIOD = 50 # epoch


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

    def __init__(self, action_space_size, img_width, img_height):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        #self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        #self.bn2 = nn.BatchNorm2d(32)
        #self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)    #image trop petite
        #self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=2, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(img_width) #a modif selon le nbre de conv   conv2d_size_out(conv2d_size_out(img_width)) etc
        convh =conv2d_size_out(img_height)
        linear_input_size = convw * convh * 16
        self.head = nn.Linear(linear_input_size, action_space_size)

    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x = F.relu(x)
        #x = F.relu(self.bn2(self.conv2(x)))
        #x = F.relu(self.bn3(self.conv3(x)))
        #print(x.shape)
        #return self.head(x)
        return self.head(x.view(x.size(0), -1))
"""
width = 3
height = 3
g=GridWorld(width,height)
g.add_start(1,1)
g.add_goal(3,3)
n_a = 4
s = g.reset()
"""


#Q_value = Perceptron(1, HIDDEN_DIM, n_a)
#target_Q_value = Perceptron(1, HIDDEN_DIM, n_a)


def get_screen(env):
    screen=env.full_observation()
    return screen


def calculate_epsilon(it):
    return (-np.arctan(20*(it-MAX_ITER/2)/MAX_ITER)+1.5)/3

def sample_action(env, screen, Q_value, epsilon):
    if random.random() < epsilon:
        return random.randint(0,n_a-1)
    else:
        return Q_value(screen.unsqueeze(0).unsqueeze(0).float()).argmax().item() # renvoit une action aleatoire


def sample_trajectory(replay_memory, Q_value, epsilon, g):
    z = g.reset() #observation
    screen=get_screen(g)
    cumul = 0
    for t in range(T_MAX):
        a = sample_action(g, screen, Q_value, epsilon)
        next_z, r, done = g.step(a)
        next_screen=get_screen(g)
        cumul += r
        replay_memory.append((screen,a,r,next_screen,done))
        z=next_z
        screen=next_screen
        if done:
            break
    return cumul

def train(g):
    replay_memory = collections.deque(maxlen=MEM_SIZE)
    epsilon = 1
    plot_tot_loss = []

    cumul_reward_epoch = 0
    average_reward_epoch = []
    cumul = 0
    with tqdm.trange(MAX_ITER) as progress_bar:
        for it in progress_bar:
            cumul = cumul+sample_trajectory(replay_memory, Q_value, epsilon, g)
            n = len(replay_memory)

            if  n >BATCH_SIZE:
                indices = list(range(n))
                random.shuffle(indices)
                tot_loss = 0
                for b in range(n // BATCH_SIZE):
                    batch_z, batch_a, batch_r, batch_nxt, batch_done = zip(
                            *(replay_memory[i] for i in indices[b * BATCH_SIZE:(b+1) * BATCH_SIZE]))

                    batch_z = torch.stack(batch_z).float().unsqueeze(1)
                    batch_a = torch.tensor(batch_a).unsqueeze(1)
                    batch_r = torch.tensor(batch_r).unsqueeze(1).float()
                    batch_nxt = torch.stack(batch_nxt).float().unsqueeze(1)
                    #print("shape", batch_nxt.shape)
                    batch_done = torch.tensor(batch_done).unsqueeze(1)

                    batch_target = batch_r + DISCOUNT*target_Q_value(batch_nxt).max(1, keepdim=True)[0]
                    batch_target[batch_done] = 0

                    batch_qval = Q_value(batch_z).gather(1, batch_a)
                    loss = nn.functional.mse_loss(batch_qval, batch_target.detach())
                    tot_loss += loss.item()

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                cumul_reward_epoch += sample_trajectory([], Q_value, 0,g)
                plot_tot_loss.append(tot_loss / (n // BATCH_SIZE))
                progress_bar.set_postfix(loss = tot_loss / (n // BATCH_SIZE), cumul=cumul)
            if it % FREEZE_PERIOD == FREEZE_PERIOD - 1:
                average_reward_epoch.append(cumul_reward_epoch/FREEZE_PERIOD)
                print("average reward per epoch", average_reward_epoch)
                cumul_reward_epoch = 0
                temp = target_Q_value.state_dict()
                target_Q_value.load_state_dict(Q_value.state_dict()) # copie des param
                Q_value.load_state_dict(temp)

            epsilon = calculate_epsilon(it)

    return plot_tot_loss, average_reward_epoch


# training
width = 3
height = 3
grid_tab = []

for _ in range(3):
    h=GridWorld(width, height)
    h.add_start(1,1)
    h.add_goal(3,3)
    n_a = 4
    s = h.reset()

    Q_value = CNNModel(n_a, width+2, height+2)
    target_Q_value = CNNModel(n_a, width+2, height+2)
    target_Q_value.load_state_dict(Q_value.state_dict()) # copie des param
    optim = torch.optim.SGD(Q_value.parameters(),lr=LEARNING_RATE)

    plot_tot_loss, average_reward_epoch = train(h)
    width += 1
    height += 1


    # plot loss
    plt.figure()
    plt.plot(plot_tot_loss)
    plt.savefig("Loss_neural_network.png")

    # plot average_reward
    plt.figure()
    plt.plot(average_reward_epoch)
    plt.savefig("average_reward_per_epoch.png")
