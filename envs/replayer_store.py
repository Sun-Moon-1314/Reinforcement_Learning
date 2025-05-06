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
    "PERReplayer",
    "DDPGReplayer",
    "TD3Replayer"
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

    def reset(self):
        """
        清空经验池，用于在每个采样周期结束后准备新一轮采样。
        """
        self.index = 0
        self.count = 0
        # 可选：重置 DataFrame 以释放内存
        self.memory = pd.DataFrame(index=range(self.capacity),
                                   columns=['observation', 'action', 'pi', 'advantage', 'return'])
        logger.info("经验池已清空，准备新一轮采样")


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

    def __init__(self, capacity,
                 alpha_initial=0.7,
                 alpha_final=1.0,
                 beta=0.4,
                 max_usage_limit=50,
                 max_age_limit=20000
                 ):
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
        self.timestamps = np.zeros(capacity, dtype=np.int64)  # 新增：记录每条数据的时间步长
        self.global_timestamp = 0  # 新增：全局时间步长计数器
        self.usage_counts = np.zeros(capacity, dtype=np.int32)  # 新增：记录每条数据的采样次数
        self.index = 0  # 当前存储位置的索引
        self.count = 0  # 当前回放池中存储的经验条数
        self.max_steps = 100000
        self.learn_step_counter = 0  # 记录训练步数
        self.capacity = capacity  # 回放池的最大容量
        self.alpha_initial = alpha_initial
        self.alpha_final = alpha_final
        self.alpha = alpha_initial  # 初始化 alpha
        self.beta = beta  # 重要性加权的程度
        self.max_usage_limit = max_usage_limit  # 新增：最大使用次数限制
        self.max_age_limit = max_age_limit  # 新增：最大存活时间步长限制

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

    def clean_old_data(self):
        """
        可选：清理使用次数超限的数据，将其从内存中移除。
        """
        valid_mask = self.usage_counts[:self.count] < self.max_usage_limit
        if not np.all(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            self.memory.iloc[:len(valid_indices)] = self.memory.iloc[valid_indices]
            self.priorities[:len(valid_indices)] = self.priorities[valid_indices]
            self.usage_counts[:len(valid_indices)] = self.usage_counts[valid_indices]
            self.count = len(valid_indices)
            logger.info(f"清理了 {len(valid_mask) - self.count} 条旧数据")

    def clean_old_data_by_age(self):
        """
        清理超过最大存活时间步长的数据。
        """
        if self.count == 0:
            return
        current_time = self.global_timestamp
        age = current_time - self.timestamps[:self.count]
        valid_mask = age < self.max_age_limit
        if not np.all(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                logger.warning("警告：清理后回放池为空，将保留部分旧数据以避免采样错误")
                # 保留一部分数据（例如最新的 10% 数据）
                retain_count = max(1, int(self.count * 0.1))
                valid_indices = np.arange(self.count - retain_count, self.count)
            self.memory.iloc[:len(valid_indices)] = self.memory.iloc[valid_indices]
            self.priorities[:len(valid_indices)] = self.priorities[valid_indices]
            self.timestamps[:len(valid_indices)] = self.timestamps[valid_indices]
            old_count = self.count
            self.count = len(valid_indices)
            logger.info(f"清理了 {old_count - self.count} 条过旧数据，剩余 {self.count} 条数据")
        else:
            logger.info("没有需要清理的过旧数据")


class DDPGReplayer:
    """
    优先级经验回放类，用于存储和采样 SAC 或其他强化学习算法中的经验。
    """

    def __init__(self,
                 capacity,
                 alpha_initial=0.7,
                 alpha_final=1.0,
                 beta=0.4,
                 max_usage_limit=50,
                 max_age_limit=20000
                 ):
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
                                            'done'])  # 是否终止标志
        self.priorities = np.zeros(capacity, dtype=np.float32)  # 用于存储优先级，初始化为零
        self.timestamps = np.zeros(capacity, dtype=np.int64)  # 新增：记录每条数据的时间步长
        self.global_timestamp = 0  # 新增：全局时间步长计数器
        self.usage_counts = np.zeros(capacity, dtype=np.int32)  # 新增：记录每条数据的采样次数
        self.index = 0  # 当前存储位置的索引
        self.count = 0  # 当前回放池中存储的经验条数
        self.max_steps = 100000
        self.learn_step_counter = 0  # 记录训练步数
        self.capacity = capacity  # 回放池的最大容量
        self.alpha_initial = alpha_initial
        self.alpha_final = alpha_final
        self.alpha = alpha_initial  # 初始化 alpha
        self.beta = beta  # 重要性加权的程度
        self.max_usage_limit = max_usage_limit  # 新增：最大使用次数限制
        self.max_age_limit = max_age_limit  # 新增：最大存活时间步长限制

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
        self.timestamps[self.index] = self.global_timestamp  # 新增：记录存储时的时间步长
        self.global_timestamp += 1  # 新增：更新全局时间步长
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

    def clean_old_data(self):
        """
        可选：清理使用次数超限的数据，将其从内存中移除。
        """
        valid_mask = self.usage_counts[:self.count] < self.max_usage_limit
        if not np.all(valid_mask):
            logger.info(f"++++clean_old_data++++")
            logger.info(f"count:{self.count}")
            valid_indices = np.where(valid_mask)[0]
            self.memory.iloc[:len(valid_indices)] = self.memory.iloc[valid_indices]
            self.priorities[:len(valid_indices)] = self.priorities[valid_indices]
            self.usage_counts[:len(valid_indices)] = self.usage_counts[valid_indices]
            self.count = len(valid_indices)
            logger.info(f"清理了 {len(valid_mask) - self.count} 条旧数据")

    def clean_old_data_by_age(self):
        """
        清理超过最大存活时间步长的数据。
        """
        if self.count == 0:
            return
        current_time = self.global_timestamp
        age = current_time - self.timestamps[:self.count]
        valid_mask = age < self.max_age_limit
        if not np.all(valid_mask):
            logger.info(f"----clean_old_data_by_age----")
            logger.info(f"age:{age}")
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                logger.warning("警告：清理后回放池为空，将保留部分旧数据以避免采样错误")
                # 保留一部分数据（例如最新的 10% 数据）
                retain_count = max(1, int(self.count * 0.1))
                valid_indices = np.arange(self.count - retain_count, self.count)
            self.memory.iloc[:len(valid_indices)] = self.memory.iloc[valid_indices]
            self.priorities[:len(valid_indices)] = self.priorities[valid_indices]
            self.timestamps[:len(valid_indices)] = self.timestamps[valid_indices]
            old_count = self.count
            self.count = len(valid_indices)
            logger.info(f"清理了 {old_count - self.count} 条过旧数据，剩余 {self.count} 条数据")
        else:
            logger.info("没有需要清理的过旧数据")


class TD3Replayer:
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
