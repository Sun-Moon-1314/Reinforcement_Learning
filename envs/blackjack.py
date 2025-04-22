# -*- coding: utf-8 -*-
"""
@File    : blackjack.py
@Time    : 2024/12/19 17:28
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    :
"""
import gym
from tools.visualizer import Visualizer
from tools.evaluator import Evaluator
from tools.printer_tool import PrintTool
import numpy as np
import logging

logger = logging.getLogger('specific_module')  # 使用特定模块的记录器
from envs.global_set import *
from envs.env_template import Env


class BlackjackEnv(Env):
    def __init__(self, name, render_mode=render_model[0], render=False):
        """
        初始化 Blackjack 环境
        :param name: 环境名称
        :param rendor_mode: 渲染模式
        :param render: 是否启用渲染
        """

        if render:
            env = gym.make(name, render_mode=render_mode)  # 创建带渲染模式的环境
        else:
            env = gym.make(name)  # 创建不带渲染模式的环境
        super().__init__(env=env)
        self.render = render

        # 游戏轮数，用于评估策略
        self.game_rounds = 3000

        # 获取环境的底层属性
        self.envs = self.env.unwrapped

        self.save_policy = True

        # 状态空间和动作空间
        self.observation_space = self.env.observation_space  # 玩家总点数、庄家明牌、是否有可用 Ace
        self.action_space = self.env.action_space  # 动作空间：0 表示停牌，1 表示继续要牌

        # 动作数量
        self.Action_Num = self.envs.action_space.n


        # 折扣因子
        self.gamma = 1.0

        # 策略收敛的阈值
        self.tolerant = 1e-6

    def reset(self):
        """
        重置环境，返回初始状态
        """
        return self.env.reset()

    def step(self, action):
        """
        执行指定动作，返回新状态和奖励信息
        :param action: 执行的动作
        """
        return self.env.step(action)

    def evaluate_action_monte_carlo(self, policy):
        """
        使用蒙特卡罗方法评估策略的动作价值函数 Q(s, a)
        :param policy: 策略（每个状态的动作概率分布）
        :return: 动作价值函数 Q(s, a)
        """
        # 初始化 Q 值和计数器
        Q_sa = np.zeros_like(policy)  # 动作价值函数 Q(s, a)
        count = np.zeros_like(policy)  # 动作计数器

        for _ in range(self.game_rounds):  # 多轮游戏模拟
            state_action_map = []  # 存储一个回合的状态-动作对
            observation, _ = self.reset()  # 重置环境并获取初始状态

            while True:
                state = self.obs_to_state(observation)  # 将观察转换为状态
                p = policy[state]
                action = np.random.choice(self.Action_Num, p=policy[state])  # 按策略选择动作[0,1]的概率分布选择动作
                state_action_map.append((state, action))  # 记录状态-动作对
                observation, reward, done, truncated, _ = self.step(action)  # 执行动作
                if done:  # 回合结束
                    break

            G = reward  # 回合累计奖励
            for state, action in state_action_map:  # 更新 Q 值
                count[state][action] += 1.  # 动作计数器累加
                Q_sa[state][action] += (G - Q_sa[state][action]) / count[state][action]  # 递增式蒙特卡罗更新

        return Q_sa

    def obs_to_state(self, observation):
        """
        将 Blackjack 的观察值转换为状态
        :param observation: (玩家总点数, 庄家明牌, 是否有 Ace)
        :return: 转换后的状态
        """
        return tuple(map(lambda x: int(x), observation[: 3]))  # 转换 Ace 为整数表示

    def play_one_round_by_random_policy(self):
        """
        使用随机策略进行一局游戏，打印详细日志信息
        """
        observation = self.reset()
        logger.info(f"观测：{observation}")
        while True:
            logger.info(f"玩家:{self.env.player}, 庄家:{self.env.dealer}")  # 打印玩家和庄家状态
            action = np.random.choice(self.action_space.n)  # 随机选择动作
            logger.info(f"动作：{action}")
            observation, reward, done, truncated, info = self.step(action)  # 执行动作
            logger.info(f"观测:{observation}, 奖励:{reward}, 结束指示:{done}")
            if done:
                break

    def monte_carlo_with_exploring_start(self, policy):
        """
        带起始探索的同策回合更新
        :return:
        """
        # 设置初始策略为所有状态下都选择动作 0（停牌）
        # 即在每个状态下，动作 0 的概率为 1（确定性策略）
        policy[:, :, :, 0] = 1.
        Q_sa = np.zeros_like(policy)  # 依据策略维度，初始化动作价值记录数组
        Count = np.zeros_like(policy)  # 依据策略维度，初始化价值动作对计数器记录数组，（s->a）对每出现一次，+1
        for i in range(self.game_rounds):
            logger.info(f"第{i}轮")
            # 随机生成的状态和动作
            state = (
                np.random.randint(12, 22),  # 随机生成玩家点数[12, 22)
                np.random.randint(1, 11),  # 随机生成庄家明牌[1, 10）
                np.random.randint(2)  # 随机生成是否有可用 Ace[0, 2)
            )
            action = np.random.randint(2)  # 随机选择动作（0：停牌，1：要牌）

            self.reset()  # 重置环境，将游戏初始化到起始状态
            if state[2]:  # 如果有可用 Ace，则初始化玩家点数为 11 + 随机生成的点数
                self.envs.player = [1, state[0] - 11]  # 玩家手牌：有A 和剩余点数
            else:
                if state[0] >= 20:  # 如果玩家点数为 21
                    self.envs.player = [10, 10]  # 玩家手牌为 [10, 10]，无 Ace
                else:  # 否则根据玩家点数初始化手牌
                    self.envs.player = [10, state[0] - 10]  # 玩家手牌：[10, 剩余点数]
            self.envs.dealer[0] = state[1]  # 初始化庄家明牌

            state_actions = []  # 存储当前回合中的 (状态, 动作) 对
            while True:
                state_actions.append((state, action))  # 记录状态和动作
                observation, reward, terminated, truncated, _ = self.step(action)  # 执行动作，返回新的状态信息
                if terminated or truncated:  # 如果游戏结束（玩家胜负已定）
                    break  # 退出循环
                state = self.obs_to_state(observation)  # 更新状态
                action = np.random.choice(self.Action_Num, p=policy[state])  # 根据策略选择下一个动作

            Return = reward  # 游戏结束后得到的回合累计奖励,因为只有在最后才能知道输赢获取奖励，所以只取最后的奖励
            for state, action in state_actions:  # 遍历回合中的每个状态-动作对
                Count[state][action] += 1.  # 更新计数器
                # TODO：依据公式逐渐调整Q_sa
                Q_sa[state][action] += (Return - Q_sa[state][action]) / Count[state][action]  # 更新 Q 值
                # TODO：更新之后会计算当前状态下所有动作中的价值最大值
                a = Q_sa[state].argmax()  # 根据当前 Q 值选择最优动作
                # TODO：将所有动作置为0
                policy[state] = 0.  # 重置当前状态的策略
                # TODO：将当前最大价值的动作置为1，每轮更新都有可能变化
                policy[state][a] = 1.  # 将最优动作的概率设为 1（确定性策略）
                # logger.info(f"策略:{policy}, 价值：{q}")

        return policy, Q_sa

    def entire_policy(self):
        """
        同策固定策略评估
        :return: 策略的价值函数
        """
        # 策略：玩家点数大于等于 20 停牌，否则继续要牌
        policy = np.zeros((22, 11, 2, 2))  # 初始化策略 (玩家点数, 庄家点数, 是否有 Ace, 动作)
        policy[20:, :, :, 0] = 1  # 点数大于等于 20 时选择动作 0 (停牌)
        policy[:20, :, :, 1] = 1  # 点数小于 20 时选择动作 1 (继续要牌)
        Q_sa = self.evaluate_action_monte_carlo(policy)  # 评估 Q 值
        v = (policy * Q_sa).sum(axis=-1)  # 计算状态价值函数 V(s)
        Visualizer.plot(v)

    def monte_carlo_with_soft(self, policy_soft, name):
        """
        基于柔性策略的同策回合更新
        :param epsilon:
        :return:
        """
        Q_sa = np.zeros_like(policy_soft)  # 依据策略维度，初始化动作价值记录数组
        Count = np.zeros_like(policy_soft)  # 依据策略维度，初始化价值动作对计数器记录数组，（s->a）对每出现一次，+1
        for i in range(self.game_rounds):
            epsilon = 1.0 if name == "ep-k" else 0.1
            # logger.info(f"第{i}轮")
            observation, _ = self.reset()  # 重置环境，将游戏初始化到起始状态
            state_actions = []  # 存储当前回合中的 (状态, 动作) 对

            while True:
                state = self.obs_to_state(observation)  # 更新状态
                action = np.random.choice(self.Action_Num, p=policy_soft[state])  # 根据策略选择下一个动作
                state_actions.append((state, action))  # 记录状态和动作
                observation, reward, terminated, truncated, _ = self.step(action)  # 执行动作，返回新的状态信息
                if terminated or truncated:  # 如果游戏结束（玩家胜负已定）
                    break  # 退出循环

            Q_total = reward  # 游戏结束后得到的回合累计奖励,因为只有在最后才能知道输赢获取奖励，所以只取最后的奖励
            for state, action in state_actions:  # 遍历回合中的每个状态-动作对
                Count[state][action] += 1.  # 更新计数器
                # TODO：依据公式逐渐调整Q_sa
                Q_sa[state][action] += (Q_total - Q_sa[state][action]) / Count[state][action]  # 更新 Q 值
                # TODO：更新之后会计算当前状态下所有动作中的价值最大值
                max_index_action = Q_sa[state].argmax()  # 根据当前 Q 值选择最优动作
                # TODO：将所有动作置为0
                # 衰减 epsilon，且不会低于0.1
                epsilon = max(0.1, (1.0 / np.sqrt(i + 1))) if name == "ep-k" else epsilon
                policy_soft[state] = epsilon / self.Action_Num  # 重置当前状态的策略
                # TODO：将当前最大价值的动作置为1，每轮更新都有可能变化
                policy_soft[state][max_index_action] += (1. - epsilon)  # 将最优动作的概率设为 1（确定性策略）
                # logger.info(f"策略:{policy_soft}, 价值：{q}")

        return policy_soft, Q_sa

    def evaluate_monte_carlo_importance_sample(self, policy, behavior_policy):
        """
        重要性采样策略评估
        :param policy: 当前评估的目标策略
        :param behavior_policy: 用于生成行为的策略（行为策略）
        :return: 动作价值函数 Q(s, a) 的估计
        """
        Q_sa = np.zeros_like(policy)  # 初始化 Q 值（动作价值函数）
        Counter = np.zeros_like(policy)  # 初始化计数器（每个状态-动作对出现的次数）

        # 执行多轮模拟游戏
        for i in range(self.game_rounds):
            state_action = []  # 存储当前回合的状态-动作对
            observation, _ = self.reset()  # 重置环境并获取初始观察值

            # 执行一轮游戏
            while True:
                state = self.obs_to_state(observation)  # 将观察转换为状态
                action = np.random.choice(self.Action_Num, p=behavior_policy[state])  # 根据行为策略选择动作
                state_action.append((state, action))  # 记录当前状态和动作
                observation, reward, terminal, truncated, _ = self.step(action)  # 执行动作，获取下一步的观察、奖励等信息
                if terminal or truncated:  # 如果游戏结束，退出循环
                    break

            G_t = reward  # 最终回合的奖励
            route = 1.0  # 初始化路径权重为 1（重要性采样权重）

            # 对回合中的每个状态-动作对进行回溯更新
            for state, action in state_action:
                Counter[state][action] += route  # 更新状态-动作对的计数器
                Q_sa[state][action] += (route / Counter[state][action]) * (G_t - Q_sa[state][action])  # 更新 Q 值
                route *= (policy[state][action] / behavior_policy[state][action])  # 更新路径权重
                if route == 0:  # 如果路径权重为 0，则不再更新
                    break

        return Q_sa  # 返回估计的动作价值函数

    def monte_carlo_importance_sample(self, policy, behavior_policy):
        """
        异策最优策略求解
        :param policy: 当前评估的目标策略
        :param behavior_policy: 用于生成行为的策略（行为策略）
        :return: 更新后的策略和 Q 值
        """
        policy[:, :, :, 0] = 1.  # 将策略初始化为全局策略（停牌）
        Q_sa = np.zeros_like(policy)  # 初始化 Q 值（动作价值函数）
        Counter = np.zeros_like(policy)  # 初始化计数器（每个状态-动作对出现的次数）

        # 执行多轮模拟游戏
        for i in range(self.game_rounds):
            state_action = []  # 存储当前回合的状态-动作对
            observation, _ = self.reset()  # 重置环境并获取初始观察值

            # 执行一轮游戏
            while True:
                state = self.obs_to_state(observation)  # 将观察转换为状态
                action = np.random.choice(self.Action_Num, p=behavior_policy[state])  # 根据行为策略选择动作
                state_action.append((state, action))  # 记录当前状态和动作
                observation, reward, terminal, truncated, _ = self.step(action)  # 执行动作，获取下一步的观察、奖励等信息
                if terminal or truncated:  # 如果游戏结束，退出循环
                    break

            G_t = reward  # 最终回合的奖励
            route = 1.0  # 初始化路径权重为 1（重要性采样权重）

            # 对回合中的每个状态-动作对进行反向回溯更新
            for state, action in reversed(state_action):  # 反向遍历
                Counter[state][action] += route  # 更新状态-动作对的计数器
                Q_sa[state][action] += (route / Counter[state][action]) * (G_t - Q_sa[state][action])  # 更新 Q 值
                a = Q_sa[state].argmax()  # 获取当前状态下最优动作
                policy[state] = 0.  # 将当前状态的策略置为全零
                policy[state][a] = 1.  # 将最优动作的概率设为 1（确定性策略）
                if a != action:  # 如果最优动作与当前动作不同，终止更新
                    break
                route /= behavior_policy[state][action]  # 更新路径权重（除以行为策略的概率）

        return policy, Q_sa  # 返回更新后的策略和 Q 值

    def exploring_start(self, name):
        """
        起始探索策略
        :return:
        """
        # 初始化策略表 policy
        # 形状为 (22, 11, 2, 2)
        # 维度含义：
        # - 玩家点数（从 0 到 21，包含 22 个值）
        # - 庄家明牌（从 0 到 10，包含 11 个值）
        # - 是否有可用 Ace（0 或 1，2 个值）
        # - 动作选择（0：停牌，1：要牌，2 个值）
        policy = np.zeros((22, 11, 2, 2))
        # 初始化时，所有状态的动作概率均为 0
        Q_sa = np.zeros_like(policy)
        method_name = None
        if name == "蒙特卡洛-同策策略求解":
            # 使用带起始探索的蒙特卡罗方法更新策略和动作价值函数
            # 输入：初始策略 policy（全停牌策略）
            # 输出：
            # - 更新后的策略 policy
            # - 动作价值函数 q，表示每个状态下每个动作的 Q 值
            policy, Q_sa = self.monte_carlo_with_exploring_start(policy)
            method_name = self.monte_carlo_with_exploring_start.__name__
        elif name == "蒙特卡洛-同策柔性策略求解-ep":
            policy = np.ones_like(policy)
            policy_soft = policy / self.Action_Num
            policy, Q_sa = self.monte_carlo_with_soft(policy_soft=policy_soft, name="ep")
            method_name = self.monte_carlo_with_soft.__name__
        elif name == "蒙特卡洛-同策柔性策略求解-ep-k":
            policy = np.ones_like(policy)
            policy_soft = policy / self.Action_Num
            policy, Q_sa = self.monte_carlo_with_soft(policy_soft=policy_soft, name="ep-k")
            method_name = self.monte_carlo_with_soft.__name__
        elif name == "蒙特卡洛-异策策略求解-重要性采样":
            behavior_policy = np.ones_like(policy) / self.Action_Num
            policy, Q_sa = self.monte_carlo_importance_sample(policy=policy, behavior_policy=behavior_policy)
            method_name = self.monte_carlo_importance_sample.__name__

        # if self.save_policy:
        #     Policy_loader.save_policy(method_name=method_name, class_name=BlackjackEnv.__name__, policy=policy)
        PrintTool.print_tool(policy, Q_sa)
        Visualizer.plot_policy_and_value(policy, Q_sa)

        return policy, Q_sa.max(axis=-1)

    def importance_sample(self):
        """
        重要性采样
        :return:
        """
        policy = np.zeros((22, 11, 2, 2))
        policy[20:, :, :, 0] = 1
        policy[:20, :, :, 1] = 1
        behavior_policy = np.ones_like(policy) * 0.5
        Q_sa = self.evaluate_monte_carlo_importance_sample(policy=policy, behavior_policy=behavior_policy)
        v = (Q_sa * policy).sum(axis=-1)
        Visualizer.plot(v)

    def evaluate_policy(self, policy_name1, policy_name2):
        """
        比较策略优劣
        :param policy_name1: 策略1名称
        :param policy_name2: 策略2名称
        :return: None
        """
        _, q_value1 = self.exploring_start(policy_name1)
        _, q_value2 = self.exploring_start(policy_name2)
        Evaluator.evaluate_policy((policy_name1, q_value1), (policy_name2, q_value2))
