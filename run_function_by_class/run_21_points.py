# -*- coding: utf-8 -*-
"""
@File    : run_21_points.py      # 文件名，run_21_points表示当前文件名
@Time    : 2024/12/27         # 创建时间，2024/12/27表示当前时间
@Author  : <your_name>     # 作者
@Email   : <your_email>    # 作者电子邮件
@Desc    : <brief_description> # 文件的简要描述
"""
import gym
from envs.blackjack import BlackjackEnv
from run_function_by_class.run_select_func import run_select_func
def run_21_points_game():
    # 创建 Blackjack 环境
    env = BlackjackEnv(name="Blackjack-v1")
    # 策略评估并绘制价值函数图
    policy_name = {
        0: "蒙特卡洛-同策策略求解",
        1: "蒙特卡洛-同策柔性策略求解-ep",
        2: "蒙特卡洛-同策柔性策略求解-ep-k",
        3: "蒙特卡洛-异策策略求解-重要性采样"
    }
    get_function = {
        0: env.entire_policy,  # 同策策略评估
        1: lambda: env.exploring_start(policy_name[0]),  # 同策策略求解
        2: lambda: env.exploring_start(policy_name[1]),  # 同策柔性策略求解
        3: lambda: env.evaluate_policy(policy_name[1], policy_name[2]),  # 同策柔性策略ep和ep-k求解并比较
        4: env.importance_sample,  # 异策策略评估
        5: lambda: env.exploring_start(policy_name[3]),  # 异策重要性采样最优策略求解
    }
    # 选择get_function中序号
    choice_method = 5
    run_select_func(get_function, choice_method)