# -*- coding: utf-8 -*-
"""
@File    : run_ice_skating.py      # 文件名，run_ice_skating表示当前文件名
@Time    : 2024/12/27         # 创建时间，2024/12/27表示当前时间
@Author  : <your_name>     # 作者
@Email   : <your_email>    # 作者电子邮件
@Desc    : <brief_description> # 文件的简要描述
"""
from envs.frozenlake import FrozenEnv
from run_envs.run_select_func import run_select_func


def run_ice_skating_game():
    # 创建环境对象
    env = FrozenEnv(name="FrozenLake-v1")
    policy_name = {
        0: "策略迭代求解最优策略",
        1: "价值迭代求解最优策略"
    }
    get_function = {
        0: env.policy_iteration,  # 策略迭代求解最优策略
        1: env.value_iteration,  # 价值迭代求解最优策略
        2: lambda: env.policy_play(policy_name[0]),  # 策略评估0
        3: lambda: env.policy_play(policy_name[1]),  # 策略评估1
    }
    # 选择get_function中序号
    choice_method = 3
    run_select_func(get_function, choice_method)
