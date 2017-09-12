# AI for the Self Driving Car

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


# Implements  Neural Network specifications
class Network(nn.Module):

    def __init__(self, input_size, number_of_actions):
        super(Network, self).__init__()
        self.input_size = input_size
        self.number_of_actions = number_of_actions

        # Two connections needed when using only 1 hidden layer (input to hidden and hidden to output)
        # Each layer is fully connected to eachother due to nn.Linear
        # The hidden layer has 30 neurons
        self.full_connection1 = nn.Linear(input_size, 30)
        self.full_connection2 = nn.Linear(30, number_of_actions)

    # Does forward propagation
    def forward(self, state):
        # Applies the rectifier activation function on the hidden layer neurons
        hidden_neurons = F.relu(self.full_connection1(state))

        # Obtains Q values from hidden layer neurons
        q_values = self.full_connection2(hidden_neurons)

        return q_values


# Implements Experience Replay
class ExperienceReplay(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    # Pushes a new event to the memory and makes sure the memory is not overcapacity
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    # Samples events from the memory
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


# Implements Deep Q Learning
class DeepQNetwork():

    def __init__(self, input_size, number_of_actions, gamma):
        # Sets the discount factor
        self.gamma = gamma

        # Displays average reward of last 100 events
        self.reward_average = []

        # Creates the Neural Network
        self.model = Network(input_size, number_of_actions)

        # Creates Experience Replay with capacity of 100,000
        self.memory = ExperienceReplay(100000)

        # Chooses which optimization algorithm to use to reduce the Loss/Cost function, and the Learning Rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Creates input Tensor with a fake first dimension
        self.last_state = torch.Tensor(input_size).unsqueeze(0)

        self.last_action = 0
        self.last_reward = 0

    # Decides what the next action should be
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile=True)) * 100)
        action = probs.multinomial()
        return action.data[0,0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables=True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)

        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_average.append(reward)

        if len(self.reward_average) > 1000:
            del self.reward_average[0]

        return action

    def score(self):
        return sum(self.reward_average) / (len(self.reward_average) + 1.)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, 'saved_ai.pth')

    def load(self):
        if os.path.isfile('saved_ai.pth'):
            print("Loading checkpoint...")
            checkpoint = torch.load('saved_ai.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Done!")
        else:
            print("No checkpoint found...")
