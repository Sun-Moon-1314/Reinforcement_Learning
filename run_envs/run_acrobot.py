# -*- coding: utf-8 -*-
"""
@File    : run_acrobot.py
@Time    : 2025/3/7 19:09
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    : 
"""
from envs.acrobot import *
from run_envs.run_select_func import run_select_func


def run_acrobot():
    """
    平衡杆
    :return:
    """
    # 创建 Cartpole 环境
    env = Acrobot(name="Acrobot-v1")
    # 策略评估并绘制价值函数图
    policy_name = {
        0: "动作actor_critic算法",
        1: "优势actor_critic算法",
        2: "资格迹actor_critic算法",
        3: "同步actor_critic算法",
        4: "异步actor_critic算法",
        5: "邻近PPO算法",
        6: "soft_actor_critic算法",
    }
    get_function = {
        0: lambda: env.game_iteration(policy_name[0]),  # 策略评估0
        1: lambda: env.game_iteration(policy_name[1]),  # 策略评估1
        2: lambda: env.game_iteration(policy_name[2]),  # 策略评估2
        3: lambda: env.game_iteration(policy_name[3]),  # 策略评估3
        4: lambda: env.game_iteration(policy_name[4]),  # 策略评估4
        5: lambda: env.game_iteration(policy_name[5]),  # 策略评估5
        6: lambda: env.game_iteration(policy_name[6]),  # 策略评估5
    }
    # 选择get_function中序号
    choice_method = 6
    run_select_func(get_function, choice_method)
