# -*- coding: utf-8 -*-
"""
@File    : replayer_store.py
@Time    : 2025/3/24 15:03
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    : 
"""
import pandas as pd
import numpy as np

__all__ = [
    "PPOReplayer",
    "SACReplayer",
    "PERReplayer"
]

from tools.visualizer import logger


class PPOReplayer:
    """
    经验回放类，用于存储和采样 PPO 中的经验。

    经验回放池（Replay Buffer）用于存储智能体在与环境交互过程中生成的经验（状态、动作、奖励、下一个状态、是否终止）。这些经验随后用于训练神经网络，以使模型学习到最佳策略。
    """

    def __init__(self, capacity):
        """
        初始化经验回放池。

        参数：
        - capacity: int，回放池的容量，决定了最多能存储多少条经验。
        """
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['observation',  # 当前状态
                                            'action',  # 执行动作
                                            'pi',  # 收到的奖励
                                            'advantage',  # 下一状态
                                            'return'])  # 是否终止标志
        self.index = 0  # 当前存储位置的索引
        self.count = 0  # 当前回放池中存储的经验条数
        self.capacity = capacity  # 回放池的最大容量

    def replay_store(self, *args, pbar=None):
        """
        将新的经验存储到回放池。

        参数：
        - args: 包含一条经验的五个元素（当前状态、动作、奖励、下一个状态、是否终止）
        """
        # 存储经验
        self.memory.loc[self.index] = args
        # 更新存储位置的索引
        self.index = (self.index + 1) % self.capacity
        # 增加回放池中存储的经验条数，并确保不超过最大容量
        self.count = min(self.count + 1, self.capacity)

        # 如果进度条存在，更新进度条
        if pbar is not None:
            pbar.update(1)

    def replay_sample(self, size):
        """
        从经验回放池中随机采样一批经验。

        参数：
        - size: int，要采样的经验数量

        """
        # 从存储的经验中随机选择索引
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)
        # 把observation中的64个indices对应的数据堆叠起来
        # obs = [[-1,0.3],[-3, 0.2],...]


class SACReplayer:
    """
    经验回放类，用于存储和采样 DQN 中的经验。

    经验回放池（Replay Buffer）用于存储智能体在与环境交互过程中生成的经验（状态、动作、奖励、下一个状态、是否终止）。这些经验随后用于训练神经网络，以使模型学习到最佳策略。
    """

    def __init__(self, capacity):
        """
        初始化经验回放池。

        参数：
        - capacity: int，回放池的容量，决定了最多能存储多少条经验。
        """
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['observation',  # 当前状态
                                            'action',  # 执行动作
                                            'reward',  # 收到的奖励
                                            'next_observation',  # 下一状态
                                            "entropy",  # 熵
                                            'done'])  # 是否终止标志
        self.index = 0  # 当前存储位置的索引
        self.count = 0  # 当前回放池中存储的经验条数
        self.capacity = capacity  # 回放池的最大容量

    def replay_store(self, *args, pbar=None):
        """
        将新的经验存储到回放池。

        参数：
        - args: 包含一条经验的五个元素（当前状态、动作、奖励、下一个状态、是否终止）
        """
        # 存储经验
        self.memory.loc[self.index] = args
        # 更新存储位置的索引
        self.index = (self.index + 1) % self.capacity
        # 增加回放池中存储的经验条数，并确保不超过最大容量
        self.count = min(self.count + 1, self.capacity)
        # 如果进度条存在，更新进度条
        if pbar is not None:
            pbar.update(1)

    def replay_sample(self, size):
        """
        从经验回放池中随机采样一批经验。

        参数：
        - size: int，要采样的经验数量

        """
        # 从存储的经验中随机选择索引
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)


class PERReplayer:
    """
    优先级经验回放类，用于存储和采样 SAC 或其他强化学习算法中的经验。
    """

    def __init__(self, capacity, alpha_initial=0.7, alpha_final=1.0, beta=0.4):
        """
        初始化优先级经验回放池。

        参数：
        - capacity: int，回放池的容量，决定了最多能存储多少条经验。
        - alpha: float，控制优先级对采样概率的影响，范围 [0, 1]。
        - beta: float，控制重要性加权的程度，范围 [0, 1]。
        """
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['observation',  # 当前状态
                                            'action',  # 执行动作
                                            'reward',  # 收到的奖励
                                            'next_observation',  # 下一状态
                                            'entropy',  # 熵
                                            'done'])  # 是否终止标志
        self.priorities = np.zeros(capacity, dtype=np.float32)  # 用于存储优先级，初始化为零
        self.index = 0  # 当前存储位置的索引
        self.count = 0  # 当前回放池中存储的经验条数
        self.max_steps = 100000
        self.learn_step_counter = 0  # 记录训练步数
        self.capacity = capacity  # 回放池的最大容量
        self.alpha_initial = alpha_initial
        self.alpha_final = alpha_final
        self.alpha = alpha_initial  # 初始化 alpha
        self.beta = beta  # 重要性加权的程度

    def replay_store(self, *args, priority=None, pbar=None):
        """
        将新的经验存储到回放池，跳过形状异常的数据。

        参数：
        - args: 包含一条经验的多个元素（状态、动作、奖励等）。
        - priority: float，经验的优先级。
        - pbar: 可选，进度条对象，用于更新进度条。
        """
        # 处理数据形状，确保所有经验数据是统一的格式
        processed_args = []
        for arg in args:
            try:
                # 检查数据类型是否为数值型或数组
                if isinstance(arg, (list, np.ndarray)):
                    arg = np.array(arg, dtype=np.float32)  # 转换为数组
                    if len(arg.shape) > 1:  # 如果是嵌套数组
                        arg = arg.flatten()  # 展平为一维数组
                elif isinstance(arg, (int, float)):  # 如果是单个数值
                    arg = np.array([arg], dtype=np.float32)
                else:
                    raise ValueError(f"不支持的类型: {type(arg)}")  # 抛出异常
                processed_args.append(arg)
            except ValueError as e:
                logger.error(f"数据形状异常，跳过存储: {arg}, 错误信息: {e}")
                return

        # 存储经验到 DataFrame
        self.memory.loc[self.index] = processed_args

        # 存储优先级，默认优先级为最大值（确保新数据被采样）
        self.priorities[self.index] = priority if priority is not None else self.priorities.max() + 1e-5

        # 更新存储位置索引
        self.index = (self.index + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

        if pbar is not None:
            pbar.update(1)

    def replay_sample(self, size):
        """
        从经验回放池中按优先级采样一批经验，并返回重要性加权的批次。

        参数：
        - size: int，要采样的经验数量。

        返回：
        - batch: dict，包含采样的经验批次（状态、动作、奖励、下一状态、是否终止）。
        - weights: np.ndarray，重要性加权系数。
        - indices: np.ndarray，采样的经验索引，用于更新优先级。
        """
        if self.count == 0:
            raise ValueError("回放池为空，无法采样。")
        # 更新 alpha
        self.update_alpha()
        # 计算采样概率
        priorities = self.priorities[:self.count]  # 仅考虑已存储的经验
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()  # 归一化

        # 按概率采样
        indices = np.random.choice(self.count, size=size, p=probabilities)

        # 生成批次数据
        batch = {field: np.stack(self.memory.loc[indices, field]) for field in self.memory.columns}

        # 计算重要性加权
        weights = (self.count * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化权重

        return batch, weights, indices

    def update_alpha(self):
        # 动态更新 alpha
        self.alpha = self.alpha_initial + (self.alpha_final - self.alpha_initial) * (
                    self.learn_step_counter / self.max_steps)
        self.alpha = min(self.alpha, self.alpha_final)  # 确保 alpha 不超过 alpha_final

    def update_priorities(self, indices, td_errors):
        """
        根据 TD-误差更新采样的经验优先级。

        参数：
        - indices: np.ndarray，采样的经验索引。
        - td_errors: np.ndarray，采样的经验对应的 TD-误差。
        """
        # 确保 errors 是 numpy 数组
        errors = np.array(td_errors, dtype=np.float32)

        # 检查 indices 和 errors 的形状是否一致
        if len(indices) != len(errors):
            raise ValueError(f"indices 和 errors 的长度不一致: {len(indices)} vs {len(errors)}")

        # 更新优先级
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-5  # 避免优先级为零
