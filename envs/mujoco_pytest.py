# -*- coding: utf-8 -*-
"""
@File    : mujoco_pytest.py
@Time    : 2025/4/20 13:28
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    : 
"""
import gym
import mujoco_py

# 创建一个MuJoCo环境
env = gym.make('Humanoid-v2')

# 重置环境
obs = env.reset()

for _ in range(1000):
    # 随机选择一个动作
    action = env.action_space.sample()

    # 执行动作
    obs, reward, done, info = env.step(action)

    # 渲染环境
    env.render()

    if done:
        obs = env.reset()

env.close()
