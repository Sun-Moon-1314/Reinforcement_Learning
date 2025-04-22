# -*- coding: utf-8 -*-
"""
@File    : run_cartpole.py
@Time    : 2025/2/2 17:35
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    : 
"""
from envs.cartpole import *
from run_function_by_class.run_select_func import run_select_func
def run_cartpole():
    """
    平衡杆
    :return:
    """
    # 创建 Cartpole 环境
    env = CartPole()
    # 策略评估并绘制价值函数图
    policy_name = {
        0: "同策策略梯度算法",
        1: "异策策略梯度算法"
    }
    get_function = {
        0: lambda: env.game_iteration(policy_name[0]),  # 策略评估0
        1: lambda: env.game_iteration(policy_name[1]),  # 策略评估0
    }
    # 选择get_function中序号
    choice_method = 1
    run_select_func(get_function, choice_method)