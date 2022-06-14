#!/usr/bin/env python
# coding=utf-8
'''
@Author: Yifu
@Email: YifuDeng@usc.edu
'''
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # 经验回放的容量
        self.buffer = [] # 缓冲区
        self.position = 0 
    
    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # 随机采出小批量转移
        state, action, reward, next_state, done =  zip(*batch) # 解压成状态，动作等
        return state, action, reward, next_state, done
    
    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)
class Actor(nn.Module):
    def __init__(self, state_size=6, hidden_size=12, action_size=2):
        super(Actor, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = state_size
        self.output_size = action_size
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        #x: batch_size * seq * input_size
        out, (hidden, cell) = self.rnn(x)
        a, b, c = hidden.shape
        #hidder: 1 * batch_size * hidden_size
        out = self.linear(hidden.reshape(a * b, c))
        #out: batch_size * output_size

        return out


class Critic(nn.Module):
    def __init__(self, state_size=6, hidden_size=12, action_size=2):
        super(Critic, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = state_size
        self.output_size = 1
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.linear_1 = nn.Linear(self.hidden_size, action_size)
        self.linear_2 = nn.Linear(action_size*2, self.output_size)

    def forward(self, obs, action):
        #action: batch_size * action_size
        out, (hidden, cell) = self.rnn(obs)  # x.shape : batch,seq_len,hidden_size , hn.shape and cn.shape : num_layes * direction_numbers,batch,hidden_size
        a, b, c = hidden.shape
        out = F.relu(self.linear_1(hidden.reshape(a * b, c)))
        out = torch.cat((out, action), dim=1)
        out = F.relu(self.linear_2(out))
        return out

class DDPG:
    def __init__(self, state_dim, action_dim, cfg):
        self.device = cfg.device
        self.critic = Critic(state_size=state_dim, action_size=action_dim, hidden_size=12).to(cfg.device)
        self.actor = Actor(state_size=state_dim, action_size=action_dim, hidden_size=12).to(cfg.device)
        self.target_critic = Critic(state_size=state_dim, action_size=action_dim, hidden_size=12).to(cfg.device)
        self.target_actor = Actor(state_size=state_dim, action_size=action_dim, hidden_size=12).to(cfg.device)

        # 复制参数到目标网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),  lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.batch_size = cfg.batch_size
        self.soft_tau = cfg.soft_tau # 软更新参数
        self.gamma = cfg.gamma

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        action = action.detach().cpu().numpy()[0]
        return action

    def update(self):
        if len(self.memory) < self.batch_size: # 当 memory 中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # 转变为张量
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)

        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
       
        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())
        
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        # 软更新
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )

    def save(self,path):
        torch.save(self.actor.state_dict(), path+'checkpoint.pt')

    def load(self,path):
        self.actor.load_state_dict(torch.load(path+'checkpoint.pt')) 