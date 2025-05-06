# -*- coding: utf-8 -*-
"""
@File    : run_mountain_car.py      # 文件名，run_mountain_car_game表示当前文件名
@Time    : 2025/1/10         # 创建时间，2025/1/10表示当前时间
@Author  : <your_name>     # 作者
@Email   : <your_email>    # 作者电子邮件
@Desc    : <brief_description> # 文件的简要描述
"""

from envs.mountaincar import *
from run_envs.run_select_func import run_select_func


def run_mountain_car():
    """
    小车上山
    :return:
    """
    # 创建 MountainCar 环境
    env = MountainCar()
    # 策略评估并绘制价值函数图
    policy_name = {
        0: "函数近似SARSA算法",
        1: "函数近似SARSA(𝜆)算法",
        2: "深度Q学习算法_pytorch",
        3: "Double深度Q学习算法_pytorch",
    }
    get_function = {
        0: env.play_game,  # 执行一步游戏
        1: lambda: env.game_iteration(show_policy=policy_name[0]),
        2: lambda: env.game_iteration(show_policy=policy_name[1]),
        3: lambda: env.game_iteration(show_policy=policy_name[2]),
        4: lambda: env.game_iteration(show_policy=policy_name[3]),
    }
    # 选择get_function中序号
    choice_method = 3
    run_select_func(get_function, choice_method)
