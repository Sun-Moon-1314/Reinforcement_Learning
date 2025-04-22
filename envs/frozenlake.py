# -*- coding: utf-8 -*-
"""
@File    : frozenlake.py      # 文件名，frozenlake表示当前文件名
@Time    : 2024/12/20         # 创建时间，2024/12/20表示当前时间
@Author  : <your_name>     # 作者
@Email   : <your_email>    # 作者电子邮件
@Desc    : <brief_description> # 文件的简要描述
"""
import gym
import numpy as np
import logging

logger = logging.getLogger(__name__)  # 使用当前模块名
from envs.global_set import *
from envs.env_template import Env


class FrozenEnv(Env):
    def __init__(self, name=None, render_mode=render_model[0], render=True):
        # 创建 FrozenLake 环境, 是否开启动画
        if render:
            env = gym.make(name, render_mode=render_mode)
        else:
            env = gym.make(name)
        super().__init__(env=env)
        self.render = render
        # 获取状态空间的大小（假设 FrozenLake 是一个网格地图，这里 nrow 和 ncol 可以直接得到）
        self.State_Num = self.observation_space.n
        # 获取动作空间的大小，即可选择的动作数量
        self.Action_Num = self.envs.action_space.n
        # 游戏轮数
        self.game_rounds = 20
        # 折扣因子，决定了未来奖励的影响
        self.gamma = 1.0
        # 柯西收敛范围
        self.tolerant = 1e-6

    def reset(self):
        # 重置环境，开始新的游戏，返回初始状态
        return self.env.reset()

    def step(self, action):
        # 执行一个动作，返回新的状态、奖励、是否终止、是否超时和其他信息
        return self.env.step(action)

    def print_info(self):
        """
        打印环境信息，显示状态空间和动作空间的相关信息
        :return: 状态空间和动作空间
        """
        logger.info(f"Observation space: {self.observation_space}")
        logger.info(f"Action space: {self.action_space}")
        return self.observation_space, self.action_space

    def play_policy(self, policy, i):
        """
        按照给定的策略进行一次游戏，返回总奖励
        :param policy: 策略，应该是一个字典，包含每个状态对应的动作选择概率
        :param render: 是否渲染环境（可视化）
        :return: 总奖励
        """
        logger.info(f"————————第{i}轮————————")
        total_reward = 0.0
        search_times = 1
        # 重置环境，获取初始状态
        observation, _ = self.reset()  # 获取状态，假设返回值是整数，不需要解包

        # 标记游戏是否结束
        done = False

        # 打印初始状态，进行调试
        # print(f"Initial observation: {observation}, type: {type(observation)}")

        # 在游戏未结束时，持续进行
        while not done:
            # 如果需要渲染环境，显示环境状态
            if self.render:
                self.env.render()

            # 检查当前状态是否在策略中存在
            if observation not in policy:
                raise KeyError(f"Observation {observation} not found in policy")

            # 根据策略选择一个动作，动作的选择概率通过政策字典获取，选择概率为1的动作的索引
            action = np.random.choice(self.Action_Num, p=policy[observation])

            # 执行动作并获取下一个状态、奖励、是否终止、是否超时和其他信息
            observation, reward, terminated, truncated, info = self.step(action)

            # 根据终止状态或超时状态判断游戏是否结束
            done = terminated or truncated

            # 打印当前状态，进行调试
            logger.debug(f"New observation: {observation}, type: {type(observation)}")
            search_times += 1
            logger.info(f"+++第{search_times}次探索完成++")
            # 累加奖励
            total_reward += reward
            logger.info(f"打印奖励:{total_reward}")

        # 返回游戏的总奖励
        logger.info("找到目标！")
        return total_reward

    def vs_2_qsa(self, value, state=None):
        """
        根据给定的状态价值函数计算每个动作的价值。
        如果指定了单个状态，则返回该状态下的每个动作的价值；如果没有指定状态，则返回所有状态的动作价值。

        :param value: 前一轮的状态价值函数数组（V(s)）dim:（1*dim(state) or dim(state)*1）
        :param state: 需要计算动作价值的状态，如果为 None，则计算所有状态的动作价值 dim:1
        :return: q_value，动作价值函数（Q(s, a)）
        """
        if state is not None:
            # 对单个状态，初始化一个空的 Q 值数组
            q_value = np.zeros(self.Action_Num)
            # 对每个动作，计算其对应的 Q 值
            for action in range(self.Action_Num):
                # 从环境中获取该状态下所有可能的转移（状态转移概率、下一个状态、奖励、终止标志）
                for prob, next_state, reward, done in self.envs.P[state][action]:
                    # 对每个转移，计算该动作的期望回报
                    """
                    强化学习书籍第21页使用状态价值函数表示动作价值函数，其中1-done为了判断之后是否有状态
                    """
                    q_value[action] += prob * (reward + self.gamma * value[next_state] * (1. - done))
        else:
            # 如果没有指定状态，计算所有状态的动作价值函数
            q_value = np.zeros((self.State_Num, self.Action_Num))
            for state in range(self.State_Num):
                q_value[state] = self.vs_2_qsa(value, state)  # 递归调用计算每个状态的动作价值
        return q_value

    def vs_evaluate(self, policy):
        """
        根据给定的策略评估状态价值函数。通过迭代更新状态价值函数，直到收敛。
        :param policy: 当前策略，定义每个状态下每个动作的选择概率
        :param tolerance: 收敛容忍度，当价值函数的变化小于此值时停止迭代
        :return: 状态价值函数
        """
        value = np.zeros(self.State_Num)  # 初始化所有状态的价值函数为零
        while True:
            delta = 0  # 记录本轮迭代中状态价值函数的最大变化
            # 遍历所有状态，更新每个状态的价值函数
            for state in range(self.State_Num):
                # 获取当前状态下的动作概率分布
                action_probs = policy[state]
                # 使用 Q 函数计算该状态下每个动作的价值
                q_values = self.vs_2_qsa(value, state)
                # 计算该状态的价值函数 V(s)，它是当前策略下的期望动作价值
                V_s = np.dot(action_probs, q_values)  # TODO 点积计算当前状态下动作的加权期望回报，强化学习书籍第41页的使用动作价值函数计算状态价值函数
                # 记录状态价值函数变化的最大值
                delta = max(delta, abs(value[state] - V_s))
                value[state] = V_s  # 更新该状态的价值
            # TODO 如果所有状态的价值函数变化都小于 tolerance，则认为已收敛，知道所有状态价值收敛才推出
            if delta < self.tolerant:
                break
        return value  # 返回更新后的状态价值函数

    def random_policy_step(self):
        """
        生成16*4维度值为0.25的随机策略
        :return:
        """
        random_policy = {state: np.ones(self.Action_Num) / self.Action_Num for state in range(self.State_Num)}
        # print(f"random_policy:{random_policy}")
        return random_policy

    def policy_improvement(self, V_s, policy):
        """
        策略提升，根据状态求最大的动作
        :param V_s: 状态价值
        :param policy: 给定策略
        :return:
        """
        optimal = True
        for state in range(self.State_Num):
            Q_sa = self.vs_2_qsa(V_s, state)  # 每次计算一个状态下的不同动作的价值:[0.4,0.5,0.6,0.3]
            action_max = np.argmax(Q_sa)  # 取最大动作索引 2
            # 如果当前状态下的最大动作不符合策略，进行更新
            if policy[state][action_max] != 1.:
                optimal = False
                policy[state][:] = 0.  # 将当前状态下所有动作的概率设为 0
                policy[state][action_max] = 1.  # 将最大动作的概率设为 1
        return optimal

    def iterate_policy(self):
        """
        策略迭代实现
        :return:
        """
        # policy_1 = np.ones((self.State_Num, self.Action_Num)) / self.Action_Num
        # 生成16*4维度概率为0.25的随机策略{0:[0.25,0.25,0.25,0.25]}
        policy_1 = self.random_policy_step()
        while True:  # 针对策略改进的重复
            # 计算该策略下的状态价值函数，直到所有状态价值收敛才推出
            V_s = self.vs_evaluate(policy_1)
            # 策略中的动作有一个不符合最优的就一直更新，直到每个状态的最大动作a不更新为止
            is_best_policy = self.policy_improvement(V_s, policy_1)
            if is_best_policy:
                break
        return policy_1, V_s

    def iterate_value(self):
        """
        价值跌倒
        :return:
        """
        V_s = np.zeros(self.State_Num)  # 初始化
        # 迭代生成最优状态价值
        while True:
            delta = 0
            # 每个状态利用最优状态价值函数，每个状态有4个动作，组q_sa最大的作为更新值
            for state in range(self.State_Num):
                V_max = max(self.vs_2_qsa(V_s, state))  # 更新价值函数
                delta = max(delta, abs(V_s[state] - V_max))
                V_s[state] = V_max
            if delta < self.tolerant:  # 满足迭代需求
                break

        policy = self.random_policy_step()  # 计算最优策略
        # 根据平衡的状态价值，根据最优动作价值函数，计算每个状态的4个动作价值并选择最大动作价值的索引，重新赋值策略pi
        for state in range(self.State_Num):
            action = np.argmax(self.vs_2_qsa(V_s, state))
            policy[state][:] = 0.
            policy[state][action] = 1.
        return policy, V_s

    def value_iteration(self):
        """
        利用价值迭代求解最优策略
        :return:
        """
        logger.info("vvvvvvvvv利用价值迭代求解最优策略vvvvvvvvv")
        policy_vi, v_vi = self.iterate_value()
        print('状态价值函数 =')
        print(v_vi.reshape(4, 4))
        print('最优策略 =')
        # 将字典转换为一个二维数组
        Policy_vi_array = np.array(list(policy_vi.values()))
        print(np.argmax(Policy_vi_array, axis=1).reshape(4, 4))
        # TODO 根据每个状态价值选择最好的动作
        # for state in range(self.State_Num):
        #     # 输出每个状态的最优动作
        #     print(f"状态 {state}: 最优动作 {np.argmax(policy_vi[state])}")
        return policy_vi, v_vi

    def policy_iteration(self):
        """
        利用策略迭代求解最优策略
        :return:
        """
        logger.info("ppppppppp利用策略迭代求解最优策略ppppppppp")
        Policy_pi, V_pi = self.iterate_policy()
        print("状态价值函数 =")
        print(V_pi.reshape(4, 4))
        print("最优策略 =")
        # 将字典转换为一个二维数组
        Policy_pi_array = np.array(list(Policy_pi.values()))
        print(np.argmax(Policy_pi_array, axis=1).reshape(4, 4))
        # 根据每个状态价值选择最好的动作
        # for state in range(self.State_Num):
        #     # 输出每个状态的最优动作
        #     print(f"状态 {state}: 最优动作 {np.argmax(Policy_pi[state])}")
        return Policy_pi, V_pi

    def random_policy_improvement(self):
        """
        对随机策略进行策略改进
        :return:
        """
        random_policy = self.random_policy_step()
        # 计算该策略下的状态价值函数，直到所有状态价值收敛才推出
        v_random = self.vs_evaluate(random_policy)
        policy = random_policy.copy()
        optimal = self.policy_improvement(v_random, policy)
        if optimal:
            logger.info("策略无更新，最优策略为：")
        else:
            logger.info("有更新，更新后的策略为")
        logger.info(policy)

    def policy_play(self, input_policy):
        """
        使用随机策略进行多次游戏，返回平均奖励
        :return: 平均奖励
        """
        # 随机策略：对于每个状态，均匀随机选择动作
        # policy[state] = [action probabilities]
        policy = self.random_policy_step()
        # 使用随机策略进行多次游戏，并收集每次游戏的奖励
        if input_policy == "策略迭代求解最优策略":
            policy, _ = self.policy_iteration()
        if input_policy == "价值迭代求解最优策略":
            policy, _ = self.value_iteration()

        episode_rewards = [self.play_policy(policy, i + 1) for i in range(self.game_rounds)]

        # 计算所有游戏的平均奖励
        average_reward = np.mean(episode_rewards)

        # 打印平均奖励
        logger.info(f"随机策略 平均奖励：{average_reward:.4f}")

        # 返回平均奖励
        return average_reward
