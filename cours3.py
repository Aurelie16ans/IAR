# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 08:10:04 2019

@author: MARGOT
"""
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html : DQN (deep q network) à partir de l'image
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py : reinforce avec policy gradient
import gym
import collections
import random
import torch
import torch.nn as nn
import tqdm

MEM_SIZE = 1000
T_MAX = 200
HIDDEN_DIM = 128
MAX_ITER = 1000
BATCH_SIZE = 64
DISCOUNT = 0.9
LEARNING_RATE = 0.0001
FREEZE_PERIOD = 30 # epoch


class Perceptron(nn.Module):
    def __init__(self,observation_space_size, hidden_dim, action_space_size):
        super().__init__()
        self.in_to_hidden = nn.Linear(observation_space_size, hidden_dim)
        self.hidden_to_out = nn.Linear(hidden_dim, action_space_size)
        
    def forward(self,observation):
        h = self.in_to_hidden(observation)
        h = nn.functional.relu(h)
        return self.hidden_to_out(h)
    


env = gym.make('CartPole-v1')
#env = gym.make('MountainCar-v0')

Q_value = Perceptron(env.observation_space.shape[0], HIDDEN_DIM, env.action_space.n)
target_Q_value = Perceptron(env.observation_space.shape[0], HIDDEN_DIM, env.action_space.n)
target_Q_value.load_state_dict(Q_value.state_dict()) #○ copie des param

optim = torch.optim.SGD(Q_value.parameters(),lr=LEARNING_RATE)
print(Q_value)


def sample_action(env, z, Q_value, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return Q_value(torch.tensor(z, dtype=torch.float).unsqueeze(0)).argmax(1).item() # renvoit une action aleatoire
    

def sample_trajectory(replay_memory, Q_value, epsilon):
    z = env.reset() #observation
    cumul = 0
    """
    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.low)
    print(env.observation_space.high)
    """
    for t in range(T_MAX):
        a = sample_action(env, z, Q_value, epsilon)
        next_z, r, done, _  = env.step(a)
        cumul += r
        env.render()
        replay_memory.append((z,a,r,next_z,done))
        if done:
            break
    return cumul
        
def train():
    replay_memory = collections.deque(maxlen=MEM_SIZE)
    epsilon = 1
    with tqdm.trange(MAX_ITER) as progress_bar:
        for it in progress_bar:
            cumul = sample_trajectory(replay_memory, Q_value, epsilon)
            
            n = len(replay_memory)
            if n < BATCH_SIZE:
                indices = list(range(n))
            
                random.shuffle(indices)
                tot_loss = 0
                for b in range(n // BATCH_SIZE):
                    batch_z, batch_a, batch_r, batch_nxt, batch_done = zip(
                            *(replay_memory[i] for i in indices[b * BATCH_SIZE:(b+1) * BATCH_SIZE])) 
                    # sans le zip, on a une liste de tuples (s,a,r,s',done), le fait de faire zip permet 
                    #d'avoir une liste de s, une liste de a, une liste de s', une liste de done...
                    # le * permet d'exploser les listes initiales pour que le zip fonctionne7
                    batch_z = torch.tensor(batch_z).float()
                    batch_a = torch.tensor(batch_a).unsqueeze(1)
                    batch_r = torch.tensor(batch_r).unsqueeze(1)
                    batch_nxt = torch.tensor(batch_nxt).float()
                    batch_done = torch.tensor(batch_done).unsqueeze(1)
                    """
                    print(batch_a)
                    print(batch_r)
                    print(batch_nxt)
                    print(batch_done)
                    """
                    batch_target = batch_r + DISCOUNT*target_Q_value(batch_nxt).max(1, keepdim=True)[0]
                    batch_target[batch_done] = 0
                    print(batch_target)
                    
                    batch_qval = Q_value(batch_z).gather(1, batch_a)
                    loss = nn.functional.mse_loss(batch_qval, batch_target.detach())
                    tot_loss += loss.item()
                    
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                progress_bar.set_postfix(loss = tot_loss / (n // BATCH_SIZE), cumul=cumul)
                    
            if it % FREEZE_PERIOD == FREEZE_PERIOD - 1:
                temp = target_Q_value.state_dict()
                target_Q_value.load_state_dict(Q_value.state_dict()) # copie des param
                Q_value.load_state_dict(temp)
                
            epsilon = 1 - it / MAX_ITER # pas optimisé
            
            
                

train()

env.close()