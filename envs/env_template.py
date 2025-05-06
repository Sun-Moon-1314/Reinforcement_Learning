# -*- coding: utf-8 -*-
"""
@File    : env_template.py      # 文件名，env_template表示当前文件名
@Time    : 2024/12/31         # 创建时间，2024/12/31表示当前时间
@Author  : <your_name>     # 作者
@Email   : <your_email>    # 作者电子邮件
@Desc    : <brief_description> # 文件的简要描述
"""

import gym
import numpy as np
import logging
from gym.spaces import Box

logger = logging.getLogger(__name__)  # 使用当前模块名


class Env:
    def __init__(self, env):
        # 创建 FrozenLake 环境, 是否开启动画
        # if render:
        #     self.env = gym.make(name, render_mode=render_mode)
        # else:
        #     self.env = gym.make(name)
        self.env = env

        # 获取环境的 unwrapped 属性，这使得我们可以访问环境的具体细节
        self.envs = self.env.unwrapped

        # 获取环境的状态空间和动作空间
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # 获取动作空间的大小，即可选择的动作数量
        # if isinstance(self.observation_space, Box):
        #     # print("self.observation_space is a Box object!")
        #     self.State_Num = self.env.observation_space.shape
        # else:
        #     # print("self.observation_space is not a Box object!")
        #     # # 获取状态空间的大小（假设 FrozenLake 是一个网格地图，这里 nrow 和 ncol 可以直接得到）
        #     self.State_Num = self.observation_space.n
        #
        # if isinstance(self.action_space, Box):
        #     # print("self.action_space is a Box object!")
        #     # 获取动作空间的大小，即可选择的动作数量
        #     self.Action_Num = self.env.action_space.shape
        # else:
        #     # print("self.action_space is not a Box object!")
        #     # # 获取动作空间的大小，即可选择的动作数量
        #     self.Action_Num = self.envs.action_space.n

        # 初始化Q（s，a）表
        # if isinstance(self.observation_space, Box) and isinstance(self.action_space, Box):
        #     self.q_sa = np.zeros((self.State_Num, self.Action_Num))
        # 模型存储控制
        self.save_policy = False
        self.load_model = False
        self.train = False

        # 折扣因子，决定了未来奖励的影响
        self.gamma = 0.9
        # 学习率
        self.learning_rate = 0.1
        # 柯西收敛范围
        self.tolerant = 1e-6
        # 衰减率
        self.epsilon = 0.01

    def reset(self):
        # 重置环境，开始新的游戏，返回初始状态
        return self.env.reset()

    def step(self, action):
        # 执行一个动作，返回新的状态、奖励、是否终止、是否超时和其他信息,
        # observation, reward, terminated, truncated, info
        return self.env.step(action)

    def agent_decide(self, state, policy):
        """
        智能体决策
        :return:
        """
        if policy is not None:
            state = self.state_transition(state)
            action = np.random.choice(self.Action_Num, p=policy[state])
        elif np.random.uniform() > self.epsilon:
            action = self.q_sa[state].argmax()
        else:
            action = np.random.randint(self.Action_Num)

        return action

    def agent_learn(self, state, action, reward, next_state, next_action, done):
        """
        智能体学习训练
        :return:
        """
        pass

    def state_transition(self, observation):
        """
        将一维int转换为四维tuple(int,int,int,int)
        :param observation:
        :return:
        """
        return tuple(self.envs.decode(observation))
