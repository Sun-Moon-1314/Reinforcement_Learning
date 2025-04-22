# -*- coding: utf-8 -*-
"""
@File    : taxi.py      # 文件名，taxi表示当前文件名
@Time    : 2024/12/27         # 创建时间，2024/12/27表示当前时间
@Author  : <your_name>     # 作者
@Email   : <your_email>    # 作者电子邮件
@Desc    : <brief_description> # 文件的简要描述
"""
import time
import gym
import numpy as np
import logging

from envs.env_template import Env
from tools.visualizer import Visualizer
from tools.save_policy import Policy_loader

logger = logging.getLogger(__name__)  # 使用当前模块名
from envs.global_set import *


class EnvInit(Env):
    """
    算法参数初始化
    """

    def __init__(self, name='Taxi-v3', render_mode=render_model[0], render=True):

        # 是否开启动画
        if render:
            env = gym.make(name, render_mode=render_mode)
        else:
            env = gym.make(name)

        super().__init__(env=env)
        self.render = render
        # # 游戏轮数
        self.game_rounds = 2001
        # 获取状态空间的大小（假设 FrozenLake 是一个网格地图，这里 nrow 和 ncol 可以直接得到）
        self.State_Num = self.observation_space.n
        # 获取动作空间的大小，即可选择的动作数量
        self.Action_Num = self.envs.action_space.n
        # 保存模型
        self.save_policy = False
        # 加载模型
        self.load_model = True
        self.q_sa = np.zeros((self.State_Num, self.Action_Num))
        self.train = False
        # 初始化Q（s，a）表
        if self.load_model:
            self.q_sa = Policy_loader.load_policy(class_name=self.__class__.__name__,
                                                  method_name="play_game_by_tracy.csv")
        else:
            self.q_sa = np.zeros((self.State_Num, self.Action_Num))

        # 折扣因子，决定了未来奖励的影响
        self.gamma = 0.9
        # 学习率
        self.learning_rate = 0.1
        # 柯西收敛范围
        self.tolerant = 1e-6
        # ε-柔性策略因子
        self.epsilon = 0.01
        # 动作映射
        self.translate_map = {
            0: "向下移动",
            1: "向上移动",
            2: "向右移动",
            3: "向左移动",
            4: "乘客上车",
            5: "乘客下车",
        }

    def translate(self, action):
        return self.translate_map[action]


class SARSA(EnvInit):
    """
    SARSA算法
    """

    def __init__(self):
        super().__init__()
        pass

    def step_one_info(self):
        """
        初始化环境并执行一步
        :return: 状态空间和动作空间
        """
        state, _ = self.reset()
        taxi_row, taxi_col, pass_loc, dest_idx = self.envs.decode(state)
        logger.info(f"出租车位置: {(taxi_row, taxi_col)}")
        logger.info(f"乘客位置: {self.envs.locs[pass_loc]}")
        logger.info(f"乘客目标: {self.envs.locs[dest_idx]}")
        # self.env.render()
        self.step(1)

    def agent_learn_by_sarsa(self, state, action, reward, next_state, next_action, done):
        """
        SARSA核心：
        U_t = R_t+1 + gamma * Q(S_t+1, A_t+1),
        TD_error = U_t - Q(S_t, A_t)
        Q(S_t, A_t) = Q(S_t, A_t) + alpha * TD_error
        :param state: 当前状态
        :param action: 当前动作
        :param reward: 奖励
        :param next_state: 下个状态
        :param done: 是否停止
        :param next_action: 下个动作
        :return:
        """
        u_t = reward + self.gamma * self.q_sa[next_state][next_action] * (1. - done)
        td_error = u_t - self.q_sa[state][action]
        self.q_sa[state][action] += self.learning_rate * td_error

    def agent_learn_by_ex_sarsa(self, state, action, reward, next_state, done):
        """
        期望SARSA
        V_t+1 = epsilon * sum(Q(S_t+1, .)) / action_n + (1 - epsilon) * max_a(Q(S_t+1, a))
        U_t = R_t+1 + gamma * V(S_t+1),
        TD_error = U_t - Q(S_t, A_t)
        Q(S_t, A_t) = Q(S_t, A_t) + alpha * TD_error
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :param next_action:
        :return:
        """
        v_t = (self.q_sa[next_state].mean() * self.epsilon + self.q_sa[next_state].max() * (1 - self.epsilon))
        u_t = reward + self.gamma * v_t
        td_error = u_t - self.q_sa[state][action]
        self.q_sa[state][action] = self.q_sa[state][action] + self.learning_rate * td_error


class Qlearning(SARSA):
    """
    Q-Learning和Double-QLearning
    """

    def __init__(self):
        super().__init__()
        self.q_0 = np.zeros((self.State_Num, self.Action_Num))
        self.q_1 = np.zeros((self.State_Num, self.Action_Num))

    def agent_learn_by_qlearning(self, state, action, reward, next_state, done):
        """
        Q-learning算法
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        u_t = reward + self.gamma * self.q_sa[next_state].max() * (1. - done)
        td_error = u_t - self.q_sa[state][action]
        self.q_sa[state][action] += self.learning_rate * td_error

    def agent_decide_dql(self, state):
        # 使用 epsilon-greedy 策略选择动作
        if np.random.uniform() > self.epsilon:
            # epsilon-greedy 中，选择最大的 Q 值动作
            action_0 = self.q_0[state].argmax()  # 在 q_0 中选择当前状态下的最大 Q 值对应的动作
            action_1 = self.q_1[state].argmax()  # 在 q_1 中选择当前状态下的最大 Q 值对应的动作

            # 根据 q_0 和 q_1 中的 Q 值选择更优的动作
            # 选择 q_0 和 q_1 中 Q 值较大的对应动作
            action = action_0 if self.q_0[state][action_0] > self.q_1[state][action_1] else action_1
        else:
            # 如果随机数小于 epsilon，选择随机动作
            action = np.random.randint(self.Action_Num)  # 在动作空间中随机选择一个动作
        return action

    def agent_learn_dql(self, state, action, reward, next_state, done):
        # 使用 50% 的概率选择更新 q_0 或 q_1
        if np.random.randint(2):  # 50% 的概率更新 q_0，50% 的概率更新 q_1
            q_update = self.q_0  # q_0 用于更新
            q_target = self.q_1  # q_1 用于计算目标值
        else:
            q_update = self.q_1  # q_1 用于更新
            q_target = self.q_0  # q_0 用于计算目标值

        # 找到 next_state 下的最大 Q 值对应的动作
        max_action = q_update[next_state].argmax()  # 在 q_update 中找到最大 Q 值对应的动作

        # 计算目标值 u_t (Temporal Difference Target)
        u_t = reward + self.gamma * q_target[next_state, max_action] * (1. - done)  # 计算时序差分目标

        # 计算时序差分误差 (TD Error)
        td_error = u_t - q_update[state][action]  # 计算目标值与当前 Q 值的误差

        # 使用 TD 误差更新 Q 值
        q_update[state][action] += self.learning_rate * td_error  # 按照学习率更新当前状态-动作对的 Q 值


class SARSALamda(Qlearning):
    def __init__(self, lamda=0.5, bata=1.0):
        super().__init__()
        self.lamda = lamda  # 设置资格迹的衰减系数 λ
        self.beta = bata  # 设置额外的迹衰减因子 β
        self.e_tracy = np.zeros_like(self.q_sa)  # 初始化资格迹矩阵，与 q_sa 的形状一致

    def agent_learn_by_tracy(self, state, action, reward, next_state, next_action, done):
        """
        使用 SARSA(λ) 算法更新 Q 值
        :param state: 当前状态
        :param action: 当前动作
        :param reward: 当前奖励
        :param next_state: 下一状态
        :param next_action: 下一动作
        :param done: 是否结束
        """
        # 对资格迹进行衰减，公式：E(s, a) *= λ * γ
        self.e_tracy *= (self.lamda * self.gamma)
        # 对当前动作的资格迹增加，公式：E(s, a) = 1 + β * E(s, a)
        self.e_tracy[state][action] = 1. + self.beta * self.e_tracy[state][action]

        # 计算目标值 u_t，根据 SARSA 算法公式：u_t = R + γ * Q(s', a') * (1 - done)
        u_t = reward + self.gamma * self.q_sa[next_state][next_action] * (1. - done)
        # 计算时序差分误差 TD-error：δ = u_t - Q(s, a)
        td_error = u_t - self.q_sa[state][action]

        # 调试信息，用于检查 Q 值矩阵、资格迹矩阵的形状以及 TD-error
        logger.debug(f"q_sa shape: {self.q_sa.shape}, e_tracy shape: {self.e_tracy.shape}, td_error: {td_error}")

        # 更新 Q 值，公式：Q(s, a) += α * E(s, a) * δ
        self.q_sa += self.learning_rate * self.e_tracy * td_error

        # 如果当前回合结束，则将所有资格迹清零
        if done:
            self.e_tracy *= 0  # 清空资格迹，准备下一回合


class Taxi(SARSALamda):
    """
    主要用于执行策略
    """

    def __init__(self):
        super().__init__()

        self.class_name = self.__class__.__name__

    def play_game_by_sarsa(self, train=False):
        """
        使用SARSA算法训练
        :param train:
        :return:
        """
        episode_reward = 0
        observation, _ = self.reset()
        action = self.agent_decide(observation, None)
        done = False
        while True:
            if self.render:
                self.env.render()

            next_observation, reward, terminated, truncated, _ = self.step(action)
            taxi_row, taxi_col, pass_loc, dest_idx = self.envs.decode(next_observation)

            if not train:
                logger.info(f"下一个状态：{(taxi_row, taxi_col)}")
            episode_reward += reward

            next_action = self.agent_decide(next_observation, None)

            if not train:
                logger.info(f"下一个动作：{self.translate(action)}")

            if terminated or truncated:
                done = True

            if train:
                self.agent_learn_by_sarsa(observation, action, reward, next_observation, next_action, done)
            else:
                time.sleep(2)

            if done:
                logger.info(f"结束一轮游戏")
                break
            observation, action = next_observation, next_action
        return episode_reward

    def play_game_by_ex_sarsa(self, train=False):
        """
        使用SARSA算法训练
        :param train:
        :return:
        """
        episode_reward = 0
        observation, _ = self.reset()
        done = False
        while True:
            if self.render:
                self.env.render()
            action = self.agent_decide(observation, None)
            if not train:
                logger.info(f"当前动作：{self.translate(action)}")
            next_observation, reward, terminated, truncated, _ = self.step(action)
            taxi_row, taxi_col, pass_loc, dest_idx = self.envs.decode(next_observation)
            if not train:
                logger.info(f"下一个状态：{(taxi_row, taxi_col)}")
            episode_reward += reward

            if terminated or truncated:
                done = True
            if train:
                self.agent_learn_by_ex_sarsa(observation, action, reward, next_observation, done)
            else:
                time.sleep(2)
            if done:
                logger.info(f"结束一轮游戏")
                break
            observation = next_observation
        return episode_reward

    def play_game_by_qlearning(self, train=False):
        """
        使用Q-Learning算法训练
        :param train:
        :return:某一轮累积奖励
        """
        episode_reward = 0
        observation, _ = self.reset()
        done = False
        while True:
            if self.render:
                self.env.render()
            action = self.agent_decide(observation, None)
            if not train:
                logger.info(f"当前动作：{self.translate(action)}")
            next_observation, reward, terminated, truncated, _ = self.step(action)
            taxi_row, taxi_col, pass_loc, dest_idx = self.envs.decode(next_observation)
            if not train:
                logger.info(f"下一个状态：{(taxi_row, taxi_col)}")
            episode_reward += reward

            if terminated or truncated:
                done = True
            if train:
                self.agent_learn_by_qlearning(observation, action, reward, next_observation, done)
            else:
                time.sleep(2)
            if done:
                logger.info(f"结束一轮游戏")
                break
            observation = next_observation
        return episode_reward

    def play_game_by_dq_learning(self, train=False):
        """
        使用Q-Learning算法训练
        :param train:
        :return:
        """
        episode_reward = 0
        observation, _ = self.reset()
        done = False
        while True:
            if self.render:
                self.env.render()
            action = self.agent_decide_dql(observation)
            if not train:
                logger.info(f"当前动作：{self.translate(action)}")
            next_observation, reward, terminated, truncated, _ = self.step(action)
            taxi_row, taxi_col, pass_loc, dest_idx = self.envs.decode(next_observation)
            if not train:
                logger.info(f"下一个状态：{(taxi_row, taxi_col)}")
            episode_reward += reward

            if terminated or truncated:
                done = True
            if train:
                self.agent_learn_dql(observation, action, reward, next_observation, done)
            else:
                time.sleep(2)
            if done:
                logger.info(f"结束一轮游戏")
                break
            observation = next_observation
        self.q_sa = self.q_0
        return episode_reward

    def play_game_by_sarsa_policy(self, policy_use, round, train=False):
        """
        使用SARSA算法训练
        :param round:
        :param policy_use:
        :param train:
        :return: 某一轮累积奖励
        """
        episode_reward = 0
        # 获取环境初始状态
        observation, _ = self.reset()
        # 依据当前状态决策
        action = self.agent_decide(observation, policy_use)
        done = False
        while True:
            if self.render:
                self.env.render()
            # 根据上一次的reset或step的状态step，或取下一时刻的状态，奖励
            next_observation, reward, terminated, truncated, _ = self.step(action)
            taxi_row, taxi_col, pass_loc, dest_idx = self.envs.decode(next_observation)
            if not train:
                logger.info(f"下一个状态：{(taxi_row, taxi_col)}")
            episode_reward += reward
            # 根据下一时刻的状态获取下一时刻的动作
            next_action = self.agent_decide(next_observation, policy_use)
            if not train:
                logger.info(f"下一个动作：{self.translate(next_action)}")
            if terminated or truncated:
                done = True
            if train:
                # 依据TD算法更新动作价值
                self.agent_learn_by_sarsa(observation, action, reward, next_observation, next_action, done)
                # 更新策略
                max_index_action = np.argmax(self.q_sa[observation])
                epsilon = max(0.1, (1.0 / np.sqrt(round)))
                policy_use[self.state_transition(observation)] = epsilon / self.Action_Num  # 重置当前状态的策略
                # TODO：将当前最大价值的动作置为1，每轮更新都有可能变化
                policy_use[self.state_transition(observation)][max_index_action] += (
                        1. - epsilon)  # 将最优动作的概率设为 1（确定性策略）
            else:
                time.sleep(2)
            if done:
                logger.info(f"结束一轮游戏")
                break
            observation, action = next_observation, next_action
        return episode_reward

    def play_game_by_tracy(self, train=False):
        """
        使用资格迹算法训练
        :param train:
        :return:
        """
        episode_reward = 0
        observation, _ = self.reset()
        action = self.agent_decide(observation, None)
        done = False
        while True:
            if self.render:
                self.env.render()
            next_observation, reward, terminated, truncated, _ = self.step(action)
            taxi_row, taxi_col, pass_loc, dest_idx = self.envs.decode(next_observation)
            if not train:
                logger.info(f"下一个状态：{(taxi_row, taxi_col)}")
            episode_reward += reward
            next_action = self.agent_decide(next_observation, None)
            if not train:
                logger.info(f"下一个动作：{self.translate(next_action)}")
            if terminated or truncated:
                done = True
            if train:
                self.agent_learn_by_tracy(observation, action, reward, next_observation, next_action, done)
            else:
                time.sleep(2)
            if done:
                logger.info(f"结束一轮游戏")
                break
            observation, action = next_observation, next_action
        return episode_reward

    def game_iteration(self, show_policy):
        """
        迭代
        :param show_policy: 使用的更新策略方式
        """
        episode_reward = 0.
        episode_rewards = []  # 总轮数的奖励(某轮总奖励)列表
        # 创建一个形状为 (5, 5, 5, 4, 6) 的数组，初始化为相等的概率
        policy = np.ones((5, 5, 5, 4, 6))
        # 将策略乘以 0.5，但每个状态下的动作概率仍然需要归一化
        policy_soft = policy / self.Action_Num

        # 归一化：确保每个状态下的动作概率和为 1
        policy_use = policy_soft / np.sum(policy_soft, axis=-1, keepdims=True)

        method_name = "default"

        for round in range(1, self.game_rounds):
            logger.info(f"---第{round}轮训练---")
            if show_policy == "显示SARSA策略更新":
                episode_reward = self.play_game_by_sarsa_policy(policy_use, round, train=False)  # 第round轮次的累积reward
                method_name = self.play_game_by_sarsa_policy.__name__
            elif show_policy == "隐藏SARSA策略更新":
                episode_reward = self.play_game_by_sarsa(train=False)
                method_name = self.play_game_by_sarsa.__name__
            elif show_policy == "期望SARSA策略更新":
                episode_reward = self.play_game_by_ex_sarsa(train=False)
                method_name = self.play_game_by_ex_sarsa.__name__
            elif show_policy == "Q-Learning更新":
                episode_reward = self.play_game_by_qlearning(train=False)
                method_name = self.play_game_by_qlearning.__name__
            elif show_policy == "Double-Q-Learning更新":
                episode_reward = self.play_game_by_dq_learning(train=False)
                method_name = self.play_game_by_dq_learning.__name__
            elif show_policy == "资格迹学习更新":
                episode_reward = self.play_game_by_tracy(train=False)
                method_name = self.play_game_by_tracy.__name__

            if self.save_policy:
                Policy_loader.save_policy(method_name, self.class_name, self.q_sa)

            if episode_reward is not None:
                episode_rewards.append(episode_reward)
                logger.info(f"第{round}轮奖励: {episode_reward}")
            else:
                logger.warning(f"第{round}轮奖励为 None，已跳过。")

            Visualizer.plot_cumulative_avg_rewards(episode_rewards, round, self.game_rounds, self.class_name,
                                                   method_name)

        print(
            f"平均奖励：{(np.round(np.mean(episode_rewards), 2))} = {np.sum(episode_rewards)} / {len(episode_rewards)}")
        print(
            f"最后100轮奖励：{(np.round(np.mean(episode_rewards[-500:]), 2))} = {np.sum(episode_rewards[-500:])} / {len(episode_rewards[-500:])}")

    def test_sarsa(self):
        """
        测试SARSA算法
        :return:
        """
        self.epsilon = 0.
        episode_rewards = [self.play_game_by_sarsa() for _ in range(100)]
