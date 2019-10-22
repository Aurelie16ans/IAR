from gridworld_maj import GridWorld
import collections
import random
import torch
import torch.nn as nn
import tqdm
from matplotlib import pyplot as plt
import numpy as np
from cnn_model import CNNModel

MEM_SIZE = 1000
T_MAX = 50
HIDDEN_DIM = 5
MAX_ITER = 500
BATCH_SIZE = 32
DISCOUNT = 0.9
LEARNING_RATE = 0.0001
FREEZE_PERIOD = 50 # epoch

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
    epsilon = 0.01
    plot_tot_loss = []

    cumul_reward_epoch = 0
    average_reward_epoch = []
    cumul = 0
    #with tqdm.trange(MAX_ITER) as progress_bar:
    #    for it in progress_bar:
    for it in range(MAX_ITER):
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
            #progress_bar.set_postfix(loss = tot_loss / (n // BATCH_SIZE), cumul=cumul)
        if it % FREEZE_PERIOD == FREEZE_PERIOD - 1:
            average_reward_epoch.append(cumul_reward_epoch/FREEZE_PERIOD)
            print("average reward per epoch", average_reward_epoch)
            cumul_reward_epoch = 0
            temp = target_Q_value.state_dict()
            target_Q_value.load_state_dict(Q_value.state_dict()) # copie des param
            Q_value.load_state_dict(temp)

        #epsilon = calculate_epsilon(it)

    return plot_tot_loss, average_reward_epoch

def create_random_griworld(width,height):
    h = GridWorld(width, height)
    #wall (none, one horizontal or one vertical)
    type_of_wall = random.randint(0,2) # 0->none, 1->horizontal, 2->vertical
    if type_of_wall != 0:
        if type_of_wall == 1: # horizontal
            at_y = random.randint(1,height)
            x = [random.randint(1,width),random.randint(1,width)]
            x.sort()
            from_x = x[0]
            #verify all points can be reached
            if x[0]==1 and x[1]==width:
                to_x = width-1
            else:
                to_x = x[1]
            h.add_horizontal_wall(at_y,from_x,to_x)
        else: #vertical
            at_x = random.randint(1,width)
            y = [random.randint(1,height),random.randint(1,height)]
            y.sort()
            from_y = y[0]
            #verify all points can be reached
            if y[0]==1 and y[1]==height:
                to_y = height-1
            else:
                to_y = y[1]
            h.add_vertical_wall(at_x,from_y,to_y)
    #start
    start_w = random.randint(1,width)
    start_h = random.randint(1,height)
    h.add_start(start_w,start_h)
    #goal
    end_w = random.randint(1,width)
    end_h = random.randint(1,height)
    while end_w==start_w and start_h == end_h:
        end_w = random.randint(1,width)
        end_h = random.randint(1,height)
    h.add_goal(end_w,end_h)
    return h 

for _ in tqdm.trange(2): #train on 50 different grid for each grid size
    # training
    width = 3
    height = 3
    grid_tab = [] 
    for _ in range(3): # train for width=height from 3 to 5
        h = create_random_griworld(width,height)
        print(h)
        n_a = 4
        s = h.reset()
    
        Q_value = CNNModel(n_a, width+2, height+2)
        #load the q value calculated before
        Q_value.load_state_dict(torch.load("cnn_{}x{}.pt".format(width,height)).state_dict())
        target_Q_value = CNNModel(n_a, width+2, height+2)
        target_Q_value.load_state_dict(Q_value.state_dict()) # copie des param
        optim = torch.optim.SGD(Q_value.parameters(),lr=LEARNING_RATE)
        
        plot_tot_loss, average_reward_epoch = train(h)
        torch.save(Q_value,"cnn_{}x{}.pt".format(width,height))
        width += 1
        height += 1
    
    
        # plot loss
        plt.figure()
        plt.plot(plot_tot_loss)
        plt.savefig("Loss_neural_network_{}x{}.png".format(width-1,height-1))
    
        # plot average_reward
        plt.figure()
        plt.plot(average_reward_epoch)
        plt.savefig("average_reward_per_epoch_{}x{}.png".format(width-1,height-1))
