import torch
import torch.nn.functional as F
import math 
from random import random, randint, seed 
import numpy as np

torch.manual_seed(1337)
np.random.seed(1337)
seed(1337)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent: 
    def __init__(self, optimizer, eps_start=0.95, eps_end=0.05, eps_decay=10000, gamma=0.99):
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.optimizer = optimizer
    
    def select_action(self, m, sequence, action_dim, steps_taken): 
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * steps_taken / self.eps_decay)

        if random() < eps: 
            action = randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                q_values = m(torch.stack(list((sequence))).unsqueeze(0).to(device)) 
                action = torch.argmax(q_values[:,-1,:]).item()
 
        return action, eps

    def update(self, m, tgt, state_transitions, num_actions): 
        states = torch.stack(([torch.stack(([s.state for s in sequence])).to(device) for sequence in state_transitions])).to(device)
        rewards = torch.stack(([torch.stack(([torch.tensor(s.reward) for s in sequence])).to(device) for sequence in state_transitions])).to(device)
        mask = torch.stack(([torch.stack(([torch.tensor([0]) if (s.done) else torch.tensor([1]) for s in sequence])).to(device) for sequence in state_transitions])).to(device)
        next_states = torch.stack(([torch.stack(([torch.tensor(s.next_state) for s in sequence])).to(device) for sequence in state_transitions])).to(device)
        actions = torch.stack([torch.LongTensor([s.action for s in sequence]) for sequence in state_transitions]).to(device)
        
        with torch.no_grad():
            next_q_values = tgt(next_states).max(-1)[0]

        q_values = m(states)

        one_hot_actions = F.one_hot(actions, num_actions).to(device)
        loss = (((rewards + mask.squeeze(-1) * next_q_values * self.gamma - torch.sum(q_values * one_hot_actions, dim=-1))**2).sum(dim=1)).mean() 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    


