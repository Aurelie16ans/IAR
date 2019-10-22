from gridworld_maj import GridWorld
import collections
import random
import torch
import torch.nn as nn
import tqdm
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from cnn_model import CNNModel

T_MAX = 200

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

def run_cnn(Q_value, g):
    g.reset()
    screen = g.full_observation()
    cumul = 0
    for t in range(T_MAX):
        a = Q_value(screen.unsqueeze(0).unsqueeze(0).float()).argmax().item()
        next_z, r, done = g.step(a)
        next_screen = g.full_observation()
        cumul += r
        #replay_memory.append((screen,a,r,next_screen,done))
        #z=next_z
        screen=next_screen
        if done:
            break
    return cumul

size_gridworld=random.randint(3,5)
print("Gridworld {} x {}".format(size_gridworld,size_gridworld))


h = create_random_griworld(size_gridworld,size_gridworld)
n_a = 4
s = h.reset()
print(h)

Q_value = CNNModel(n_a, size_gridworld+2,size_gridworld+2)
Q_value.load_state_dict(torch.load("cnn_{}x{}.pt".format(size_gridworld,size_gridworld)).state_dict())
reward = run_cnn(Q_value, h)
print(reward)