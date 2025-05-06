# -*- coding: utf-8 -*-
"""
@File    : run_pendulum.py
@Time    : 2025/4/28 19:10
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    : 
"""
from envs.pendulum import *
from run_envs.run_select_func import run_select_func


def run_pendulum():
    """
    平衡杆
    :return:
    """
    # 创建 Cartpole 环境
    env = Pendulum(name="Pendulum-v1")
    # 策略评估并绘制价值函数图
    policy_name = {
        0: "深度确定性策略梯度算法",
        1: "双重延迟深度确定性策略梯度算法"
    }
    get_function = {
        0: lambda: env.game_iteration(policy_name[0]),  # 策略评估0
        1: lambda: env.game_iteration(policy_name[1]),  # 策略评估1
    }
    # 选择get_function中序号
    choice_method = 1
    run_select_func(get_function, choice_method)
