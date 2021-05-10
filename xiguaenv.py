#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/5/10 3:39 下午
# @File  : xiguaenv.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 使用gym封装一下

import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env

class DaxiguaEnv(gym.Env):
    """
    大西瓜env
    """
    def __init__(self, grid_size=10):
        super(DaxiguaEnv, self).__init__()
        # 定义动作和观察空间
        # 他们必须是gym.spaces对象
        # 当使用离散动作时，这里我们14个动作：代表放置14个位置
        n_actions = 14
        self.action_space = spaces.Discrete(n_actions)
        # 这里我们观察到的是agent的坐标, 这可以用离散空间和box空间来描述。Box的值的下限是0，上限是self.grid_size的大小，shape中的值为1，代表每次观察空间返回一个值
        self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                            shape=(1,), dtype=np.float32)

    def reset(self):
        """
        重要提示：观察必须是numpy数组
        :return: (np.array)
        """
        # 初始化网格右侧的agent
        self.agent_pos = self.grid_size - 1
        # 这里我们将其转换为float32，以使其更加通用(以防我们想使用连续操作), eg: 返回[9.]
        return np.array([self.agent_pos]).astype(np.float32)

    def step(self, action):
        if action == self.LEFT:
            self.agent_pos -= 1
        elif action == self.RIGHT:
            self.agent_pos += 1
        else:
            raise ValueError(f"收到了不属于动作空间的动作 {action}")

        # 网格的边界裁剪， 小于0的输出0，大于grid_size输出grid_size, 如果在0到grid_size中间，直接返回这个值
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

        # 我们是网格左边的吗？用于判断是否结束了这个episode
        done = bool(self.agent_pos == 0)

        # 除了到达目标时，其他地方的奖励都是null的（grid的左边）。
        reward = 1 if self.agent_pos == 0 else 0

        # 我们可以选择传递额外的信息，但我们现在还没有使用。
        info = {}
        # 所以返回了np.array([self.agent_pos]).astype(np.float32)代表观察空间，reward代表奖励
        return np.array([self.agent_pos]).astype(np.float32), reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # 用x代表agent的位置
        print(f"目前机器人的位置是，用x代表机器人: ")
        print("." * self.agent_pos, end="")
        print("x", end="")
        print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass


def do_check_env():
    """
    检查我们的自定义的env是否符合gym的规则
    :return:
    :rtype:
    """
    print(f"开始检查env是否满足gym的设定")
    env = GoLeftEnv()
    print(f"当前环境的观测空间是 {env.observation_space}")
    print(f"当前环境的动作空间是 {env.action_space}")
    print(f"当前环境的一个动作抽样是 {env.action_space.sample()}")
    # 如果环境不符合接口，将抛出一个错误。 来检查你的环境是否遵循Gym接口。它还可以选择检查环境是否与Stable-Baselines兼容（必要时发出警告）
    check_env(env, warn=True)




if __name__ == '__main__':
    do_check_env()