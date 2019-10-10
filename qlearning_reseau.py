# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:57:56 2019

@author: MARGOT
"""

import torch
import torch.nn as nn
import torch.nn.functional as func
import math
from gridworld import GridWorld
#from environment import Environment
import tqdm
import numpy as np

class MyMLP(nn.Module):
    def __init__(self, size_in, size_inter, size_out):
        super().__init__()
        #approx sin(x) : entree: x    sortie:sin(x)
        self.linear_in_to_hidden=nn.Linear(size_in,size_inter) #premiere couche du reseau de neurones (nbres de neurones en entrre*, nbre de neurones en sortie*) *du layer
        self.linear_hidden_to_out=nn.Linear(size_inter,size_out) #couche de sortie

    def forward (self, x): #donner la fonction a effectuer pour passer a la suite
        h=self.linear_in_to_hidden(x)
        h=nn.functional.relu(h) #fonction de non linearite (activation function)
        return self.linear_hidden_to_out(h)
    
    
if __name__ == "__main__":
    # Q learning
    g=GridWorld(4,4)
    g.add_start(1,1)
    n_s = g.state_space_size
    n_a = g.action_space_size
    s = g.reset()
    discount = 0.9
    n_it = 1000
    epsilon=0.01
    max_t = 200
    lr=0.1
    n_batch=32
    
    Q = MyMLP(1,32,4)
    optim=torch.optim.SGD(Q.parameters(), lr=0.01)
    a=torch.tensor(0,dtype=torch.float32)
    print(Q(a.unsqueeze(0)))
    for _ in tqdm.trange(n_it):
        q_s_a=[]
        targets=[]
        for i in range(n_batch):
            s=g.reset()
            done=False
            td_errors=[]
            print(s)
            a=Q(torch.tensor(s,dtype=torch.float32).unsqueeze(0)).argmax() if torch.rand((1,)) < epsilon else torch.randint(0,n_a,(1,))
            print(a)
            s_prime, r, done = g.step(a)
            target=0 if done else Q(torch.tensor(s_prime,dtype=torch.float32).unsqueeze(0)).argmax()
            target = r +discount*target
            """"
            td_errors.append((s,a,target-Q(s)[a]))
            """
            s=s_prime
            if done:
                break
            
            q_s_a.append(Q(torch.tensor(s,dtype=torch.float32).unsqueeze(0))[a])
            targets.append(target)
            """    
            for s,a,err in td_errors:
                Q(s)[a] += lr*err
            """
        targets_tensor = torch.tensor(targets,dtype=torch.float32)
        q_s_a_tensor = torch.tensor(q_s_a)
        loss=nn.functional.mse_loss(q_s_a_tensor, targets_tensor)
        print(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()
        