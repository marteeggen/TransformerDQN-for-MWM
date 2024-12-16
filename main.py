import numpy as np
from datetime import datetime
import torch
import torch.optim as optim
from random import randint, seed
import os
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from config import config
from agent import Agent
from env import CircularEnvironment
from dqn import TransformerDQN
from replay_buffer import ReplayBuffer, Sarsd

torch.manual_seed(1337)
np.random.seed(1337)
seed(1337)

hyperparameters = config['hyperparameters']
radius = hyperparameters['radius']
platform_radius = hyperparameters['platform_radius']
field_of_view = hyperparameters['field_of_view']
rotation_angle = hyperparameters['rotation_angle']
num_sight_lines = hyperparameters['num_sight_lines']
num_landmarks = hyperparameters['num_landmarks']
unique_colors = hyperparameters['unique_colors']
max_steps = hyperparameters['max_steps']
num_episodes = hyperparameters['num_episodes']
num_samples = hyperparameters['num_samples']
sequence_length = hyperparameters['sequence_length']
tgt_update = hyperparameters['tgt_update']
model_update = hyperparameters['model_update']
lr = hyperparameters['lr']
gamma = hyperparameters['gamma']
eps_start = hyperparameters['eps_start']
eps_end = hyperparameters['eps_end']
eps_decay = hyperparameters['eps_decay']
dropout = hyperparameters['dropout']
n_embd = hyperparameters['n_embd']
n_head = hyperparameters['n_head']
n_layer = hyperparameters['n_layer']
buffer_size = hyperparameters['buffer_size']
min_replay_buffer_size = hyperparameters['min_replay_buffer_size']


def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())

def initialize_replay_buffer(env, replay_buffer, action_dim):
    state = env.reset()
    while (len(replay_buffer.buffer) <= min_replay_buffer_size): 
        state = torch.FloatTensor(state).to(device)
        action = randint(0, action_dim-1)
        next_state, reward, done, info = env.step(action)
        replay_buffer.insert(Sarsd(state, action, reward, next_state, done))
        state = next_state
        if done:
            state = env.reset()  
    

def save_checkpoint(m, optimizer, episode, steps, filepath):
    save_path = f"{filepath}/episode_{episode}.pt"
    torch.save({'episode': episode, 'steps': steps, 'model_state_dict': m.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),}, save_path)


def training_loop(m, tgt, env, agent, action_dim, saved_models_filepath, tensorboard_filepath):

    replay_buffer = ReplayBuffer(buffer_size)

    steps_since_tgt = 0
    steps_since_train = 0
    steps_taken = 0
    steps_taken_episode = 0

    m.train()
    tgt.train()
    update_tgt_model(m, tgt)

    writer = SummaryWriter(tensorboard_filepath)
    sequence = deque(maxlen=sequence_length)
    initialize_replay_buffer(env, replay_buffer, action_dim)
        
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps_taken_episode = 0

        for _ in range(env.max_steps):
            state = torch.FloatTensor(state).to(device)
            sequence.append(state)
            action, eps = agent.select_action(m, sequence, action_dim, steps_taken)
            next_state, reward, done, info = env.step(action)

            replay_buffer.insert(Sarsd(state, action, reward, next_state, done))

            if (steps_since_train > model_update):
                loss = agent.update(m, tgt, replay_buffer.sample(num_samples, sequence_length), action_dim)
                steps_since_train = 0

            if steps_since_tgt > tgt_update:
                update_tgt_model(m, tgt)
                steps_since_tgt = 0

            state = next_state
            total_reward += reward
            steps_since_tgt += 1
            steps_taken += 1
            steps_since_train += 1
            steps_taken_episode += 1

            if done:
                break

        writer.add_scalar('Total reward', total_reward, episode)
        writer.add_scalar('Steps per episode', steps_taken_episode, episode)
        writer.add_scalar('Training loss', loss.detach().cpu().item(), steps_taken)
        writer.add_scalar('Epsilon', eps, steps_taken)

        if episode % 1000 == 0:
            save_checkpoint(m, agent.optimizer, episode, steps_taken, saved_models_filepath)

    save_checkpoint(m, agent.optimizer, episode, steps_taken, saved_models_filepath)

def main():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
    tensorboard_filepath = f"runs/{timestamp}" 
    saved_models_filepath = f"saved_models/{timestamp}"
    os.makedirs(tensorboard_filepath) 
    os.makedirs(saved_models_filepath)

    env = CircularEnvironment(radius, platform_radius, num_sight_lines, field_of_view, rotation_angle, max_steps, num_landmarks, unique_colors)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Goal position: {env.invisible_platform}") 

    m = TransformerDQN(state_dim, action_dim, sequence_length, n_embd, n_head, n_layer, dropout)
    tgt = TransformerDQN(state_dim, action_dim, sequence_length, n_embd, n_head, n_layer, dropout)
    optimizer = optim.Adam(m.parameters(), lr=lr)
    agent = Agent(optimizer, eps_start, eps_end, eps_decay, gamma)

    m = m.to(device)
    tgt = tgt.to(device)

    training_loop(m, tgt, env, agent, action_dim, saved_models_filepath, tensorboard_filepath)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    main()

