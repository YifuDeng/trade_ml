#!/usr/bin/env python
# coding=utf-8
'''
@Author: Yifu
@Email: YifuDeng@usc.edu
'''
import gym
import numpy as np
from gym import spaces
import random
class NormalizedActions(gym.ActionWrapper):
    ''' 将action范围重定在[0.1]之间
    '''
    def action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        return action

    def reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        return action

class OUNoise(object):
    '''Ornstein–Uhlenbeck噪声
    '''
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu # OU噪声的参数
        self.theta        = theta # OU噪声的参数
        self.sigma        = max_sigma # OU噪声的参数
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
    def reset(self):
        self.obs = np.ones(self.action_dim) * self.mu
    def evolve_obs(self):
        x  = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.obs = x + dx
        return self.obs
    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period) # sigma会逐渐衰减
        return np.clip(action + ou_obs, self.low, self.high) # 动作加上噪声后进行剪切


class TradeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super().__init__()
        self.up_max = np.float(df["up"].max())
        self.down_max = np.float(df["down"].max())
        self.close_max = np.float(df["close_price"].max())
        self.open_max = np.float(df["open_price"].max())
        self.volume_max = np.float(df["volume"].max())

        self.up_min = np.float(df['up'].min())
        self.down_min = np.float(df['down'].min())
        self.close_min = np.float(df['close_price'].min())
        self.open_min = np.float(df['open_price'].min())
        self.volume_min = np.float(df['volume'].min())

        self.bar_data = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

        self.array = np.append(np.array(self.bar_data, dtype=np.float32), np.zeros([len(self.bar_data), 1]), axis=1)

        # self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.fee = -0.0001
        self.num = 0.01
        self.balance = 0.0
        self.penalty = 0.01
        self.seq = 20
        # 定义动作和观测空间, 两者必须为 gym.spaces 对象

        # 动作的格式为: buy %, sell %, hold, 等
        self.action_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

        self.total_profit = 0.0
        # 价格包含最后5个价格的OHCL值
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.seq, 6), dtype=np.float32)

    def reset(self):
        # 重置环境的状态到初始状态
        self.inventory = 0
        self.balance = 0
        self.array = np.append(np.array(self.bar_data, dtype=np.float32), np.zeros([len(self.bar_data), 1]), axis=1)

        # 设置当前 step 到一个随机点
        # self.current_step = random.randint(self.seq, int(len(self.array)/2))
        self.current_step = 25
        state = self.array[self.current_step-self.seq: self.current_step]

        return state

    def step(self, action):
        print(self.current_step)
        print(action)
        print(self.total_profit)

        # 在环境中执行一步
        last_inventory = self.inventory
        last_balance = self.balance
        open_price = self.array[self.current_step][0] * (self.open_max-self.open_min) + self.open_min
        close_price = self.array[self.current_step][1] * (self.close_max-self.close_min) + self.close_min
        up = self.array[self.current_step][2] * (self.up_max-self.up_min) + self.up_min
        down = self.array[self.current_step][3] * (self.down_max-self.down_min) + self.down_min

        action_up = action[0] * (self.up_max-self.up_min) + self.up_min
        action_down = action[1] * (self.down_max-self.down_min) + self.down_min
        if up > action_up:
            self.inventory -= self.num
            self.balance += self.num * (open_price + action_up) * (1-self.fee)
        if down > action_down:
            self.inventory += self.num
            self.balance -= self.num * (open_price - action_down) * (1+self.fee)
        reward = (self.balance + self.inventory * close_price) - \
                 (last_balance + last_inventory * close_price) - \
                 (abs(self.inventory) - abs(last_inventory)) * close_price * self.penalty

        # print(close_price)
        # print([action_up, action_down])
        # print([up, down])
        # print(self.inventory)
        # print(reward)
        self.array[self.current_step, 5] = self.inventory

        self.current_step += 1



        done = self.current_step >= len(self.array)

        state = self.array[self.current_step-self.seq: self.current_step]

        self.total_profit = self.balance + self.inventory * close_price
        # {} 是要打印的信息


        return state, reward, done, {}


    def render(self, mode='human', close=False):

        pass