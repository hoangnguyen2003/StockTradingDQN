import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from .dqn import DQN

class DQNAgent:
    def __init__(self, config):
        self.state_size = config['model']['state_size']
        self.action_size = len(config['actions'])
        self.memory = deque(maxlen=config['training']['max_memory_size'])
        self.gamma = config['training']['gamma']
        self.epsilon = config['training']['epsilon']
        self.epsilon_min = config['training']['epsilon_min']
        self.epsilon_decay = config['training']['epsilon_decay']
        self.learning_rate = config['training']['learning_rate']
        
        self.model = DQN(config['model'], len(config['actions']))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state, actions, is_backtesting=False):
        if not is_backtesting and random.uniform(0, 1) < self.epsilon:
            return random.choice(list(actions))
        
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state in minibatch:
            target = reward
            if next_state is not None:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target_tensor = self.model(state_tensor).clone().detach()
            target_tensor[0][action] = target

            self.optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = self.criterion(output, target_tensor)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay