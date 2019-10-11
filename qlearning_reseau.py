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
import matplotlib.pyplot as plt

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
    n_it = 100
    epsilon=0.01
    max_t = 500
    lr=0.1
    n_size_dataset=100 #nbr de samples ajoute au dataset a chaque iteration
    
    Q = MyMLP(1,32,4)
    optim=torch.optim.SGD(Q.parameters(), lr=0.01)

    Q_prime=MyMLP(1,32,4)

    a=torch.tensor(0,dtype=torch.float32)
    start_state = 0
    end_state = g.state_space_size

    loss_list=[]

    s_list = []   #en fait on garde les samples produits lors des autres iterations
    a_list = []
    target_list = []
    for _ in tqdm.trange(n_it):


        for i in range(n_size_dataset):   #fait le dataset
            s = torch.tensor(1, dtype=torch.float32).random_(start_state, end_state)
            done=False
            td_errors=[]
            #print(s)
            a=Q(torch.tensor(s,dtype=torch.float32).unsqueeze(0)).argmax() if torch.rand((1,)) < epsilon else torch.randint(0,n_a,(1,))
            #print(a)
            s_prime, r, done = g.step(a)
            target=0 if done else Q_prime(torch.tensor(s_prime,dtype=torch.float32).unsqueeze(0)).max()
            target = r +discount*target
            """"
            td_errors.append((s,a,target-Q(s)[a]))
            """
            s_list.append(s)
            a_list.append(a)
            #s=s_prime   = pas besoin vu qu'on prend des states aleatoires
            #if done:
             #   break    : idem
            
            target_list.append(target)


        s_tensor=torch.tensor(s_list, dtype=torch.float32)
        a_tensor=torch.tensor(a_list, dtype=torch.long)
        target_tensor=torch.tensor(target_list, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(s_tensor, a_tensor, target_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)

        for i,(s, a, target) in  enumerate(loader):
            out=Q(s.unsqueeze(1))
            out_action=[]
            for i in range(len(a)):
                out_action.append(out[i, a[i]])
            out_action_tensor=torch.tensor(out_action, dtype=torch.float32, requires_grad=True)
            loss=nn.functional.mse_loss(out_action_tensor, target)
            print("Loss   ",loss)
            optim.zero_grad()
            loss.backward()
            optim.step()


        Q_prime.load_state_dict(Q.state_dict()) #copie les poids maj de Q dans Q_prime (fixe sur une iteration)

        prv_loss=loss/len(dataset)
        loss_list.append(prv_loss)

    plt.plot(range(0, n_it), loss_list)
    plt.savefig("loss.png")