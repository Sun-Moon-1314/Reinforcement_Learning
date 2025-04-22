# -*- coding: utf-8 -*-
"""
@File    : evaluator.py
@Time    : 2024/12/19 16:52
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    :
"""
import gym
import numpy as np

class PrintTool:
    @staticmethod
    def print_tool(policy, q_sa):
        print("策略 (Policy):")
        for player_sum in range(12, 22):  # 玩家点数范围 12-21
            for dealer_showing in range(1, 11):  # 庄家点数范围 1-10
                for has_ace in range(2):  # 是否有 Ace
                    action = np.argmax(policy[player_sum, dealer_showing, has_ace])  # 最优动作
                    print(
                        f"玩家点数: {player_sum}, "
                        f"庄家明牌: {dealer_showing}, "
                        f"有 Ace: {has_ace} -> 动作: {'停牌' if action == 0 else '要牌'}")
        print("\nQ 值 (Q-values):")
        for player_sum in range(12, 22):  # 玩家点数范围 12-21
            for dealer_showing in range(1, 11):  # 庄家点数范围 1-10
                for has_ace in range(2):  # 是否有 Ace
                    q_values = q_sa[player_sum, dealer_showing, has_ace]
                    print(
                        f"玩家点数: {player_sum}, "
                        f"庄家明牌: {dealer_showing}, "
                        f"有 Ace: {has_ace} -> Q 值: 停牌: {q_values[0]:.2f}, 要牌: {q_values[1]:.2f}")
        print("\n")  # 分隔符

    @staticmethod
    def print_all_env():
        # 获取所有注册的环境
        env_list = list(gym.envs.registry.keys())

        # 打印环境列表
        for env in env_list:
            print(env)

        cartpole_envs = [env for env in env_list if "Taxi" in env]
        print(cartpole_envs)

    @staticmethod
    def get_related_env(name):
        if len(name) == 0:
            return
        env = gym.make(name)
        print(env.spec)  # 打印环境的配置详情