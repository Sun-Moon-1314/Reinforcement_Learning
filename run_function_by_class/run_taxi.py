# -*- coding: utf-8 -*-
"""
@File    : run_taxi.py      # 文件名，run_taxi表示当前文件名
@Time    : 2024/12/27         # 创建时间，2024/12/27表示当前时间
@Author  : <your_name>     # 作者
@Email   : <your_email>    # 作者电子邮件
@Desc    : <brief_description> # 文件的简要描述
"""
from envs.taxi import *
from run_function_by_class.run_select_func import run_select_func


def run_taxi_dispatch():
    """
    出租车调度
    :return:
    """
    # 创建 Blackjack 环境
    env = Taxi()
    # 策略评估并绘制价值函数图
    policy_name = {
        0: "显示SARSA策略更新",
        1: "隐藏SARSA策略更新",
        2: "期望SARSA策略更新",
        3: "Q-Learning更新",
        4: "Double-Q-Learning更新",
        5: "资格迹学习更新",
    }
    get_function = {
        0: env.step_one_info,  # 执行一步游戏
        1: env.test_sarsa,
        2: lambda: env.game_iteration(policy_name[0]),  # 策略评估0
        3: lambda: env.game_iteration(policy_name[1]),  # 策略评估1
        4: lambda: env.game_iteration(policy_name[2]),  # 策略评估2
        5: lambda: env.game_iteration(policy_name[3]),  # 策略评估3
        6: lambda: env.game_iteration(policy_name[4]),  # 策略评估4
        7: lambda: env.game_iteration(policy_name[5]),  # 策略评估4
    }
    # 选择get_function中序号
    choice_method = 2
    run_select_func(get_function, choice_method)
