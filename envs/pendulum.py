# -*- coding: utf-8 -*-
"""
@File    : Pendulum.py
@Time    : 2025/4/26 18:00
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    : 
"""

import time
from collections import deque

from torch import optim, dtype
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
import torch.nn as nn
import torch

from envs.env_template import Env

from envs.replayer_store import *
from envs.replayer_store import TD3Replayer
from tools.visualizer import Visualizer
from tools.save_policy import Policy_loader
from networks.build_network import BuildNetwork, BuildA3CNetwork
from envs.global_set import *
import logging

# 避免重复配置
logger = logging.getLogger(__name__)  # 使用当前模块名
logger.propagate = True  # 禁用继承
# 设置日志级别为 INFO
# logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

# 禁用日志
# logging.disable(logging.CRITICAL)
# 恢复日志
# logging.disable(logging.NOTSET)

seed = 100
current_time = time.localtime()


def make_env(env_name):
    def _init():
        return gym.make(env_name)

    return _init


class EnvInit(Env):
    """
    算法参数初始化
    """

    def __init__(self, env):
        super().__init__(env)
        self.i = 0
        # 是否开启动画
        self.env = env
        # 在强化学习中，环境的随机性会影响训练效果。通过设置固定的随机种子，
        # 可以确保每次运行时环境的随机行为一致，从而使实验结果具有可重复性。
        # self.env.reset(seed=seed)
        self.render = render
        # 游戏轮数
        self.game_rounds = 30000
        # 获取动作空间的大小，即可选择的动作数量
        if isinstance(self.observation_space, Box):
            self.state_dim = self.env.observation_space.shape
            self.state_num = None  # 连续空间状态总数不可枚举
        elif hasattr(self.observation_space, 'n'):
            self.State_Num = self.observation_space.n
        else:
            raise NotImplementedError(f"Unsupported observation space: {self.observation_space}")

        if isinstance(self.action_space, Box):
            self.action_dim = self.env.action_space.shape
            self.action_num = None
        elif hasattr(self.action_space, 'n'):
            self.action_dim = self.envs.action_space.n
        else:
            raise NotImplementedError(f"Unsupported action space: {self.action_space}")
        # 位置
        self.positions = []
        # 用于跟踪最近游戏的完成率
        self.done_rate = deque(maxlen=300)
        self.done_rate.clear()
        # 速度
        self.velocities = []
        # 保存模型
        self.save_policy = True
        # 加载模型
        self.load_model = False
        # 是否开启tensorboard记录logs
        self.is_open_writer = True
        # 是否全局训练，用于设置某些记录
        self.global_is_train = True
        # 折扣因子，决定了未来奖励的影响
        self.gamma = 1.
        # 学习率
        self.learning_rate = 0.0001
        # 柯西收敛范围
        self.tolerant = 1e-6
        # ε-柔性策略因子
        self.epsilon = 0.001
        self.translate_action = {
            0: "左",
            1: "无",
            2: "右"
        }

    def env_init(self, name):
        # 是否开启动画
        self.env = gym.make(name)


class OUNoise:
    """
    OU动作噪声

    disturbance_next = disturbance - theta * (disturbance - mu) * dt + sigma * sqrt(dt) * N(0,1)
    :param action_dim: OU噪声维度，输出的维度需要与动作维度一致
    :param theta: 回归速度，控制噪声回归均值速度，越大，速度越快，平滑性低，越小越平滑
    :param mu: 噪声均值，为0，表示无偏
    :param sigma: 波动强度，控制噪声的大小
    :param dt: 时间步长
    """

    def __init__(self, action_dim, mu=0.0, sigma=1.0, theta=0.15, dt=0.01):
        self.disturbance = np.ones(action_dim) * mu
        self.action_dim = action_dim
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt

    def reset(self, disturbance=0.):
        """
        重制扰动
        :param disturbance: 原始噪声扰动，初始为0或者随机值
        :return:
        """
        self.disturbance = np.ones(self.action_dim) * disturbance

    def __call__(self, *args, **kwargs):
        """
        生成下一步的OU噪声
        :param args:
        :param kwargs:
        :return:
        """
        self.disturbance += -self.theta * (self.disturbance - self.mu) + self.sigma * np.sqrt(
            self.dt) * np.random.randn(self.action_dim if isinstance(self.action_dim, int) else self.action_dim[0])
        return self.disturbance


class DDPGAgent(EnvInit):
    """
    Deep Deterministic Policy Gradient

    :param batch_size: 采样批次
    :param sample_rounds: 采样次数(从经验池中)
    :param required_sample_size: 经验池开始训练最低批次
    :param capacity: 经验池容量
    :param alpha_initial: 控制经验池中优先级程度(越大越趋向于重要性高的, 反之越随机)
    :param gamma: 折扣因子
    :param soft_update_alpha: 软更新因子(控制主网络->目标网络的参数传递平滑性)
    :param actor_learning_rate: 策略网络学习率
    :param critic_learning_rate: 价值网络学习率
    :param explore: 开启OU噪声以及策略探索机制
    :param load_model: 加载已训练模型
    :param noise_scale: 波动强度，控制噪声的大小
    """

    def __init__(self, env,
                 batch_size=100,
                 sample_rounds=10,
                 required_sample_size=1000,
                 capacity=100000,
                 alpha_initial=0.1,
                 alpha_final=1.0,
                 gamma=0.99,
                 soft_update_alpha=0.005,
                 actor_learning_rate=0.00005,
                 critic_learning_rate=0.00005,
                 explore=True,
                 load_model=True,
                 noise_scale=0.01
                 ):
        super().__init__(env)
        self.env = env
        self.load_model = load_model
        self.action_low = self.action_space.low
        self.action_high = self.action_space.high
        self.explore = explore  # 噪声探索
        self.ou_noise = OUNoise(action_dim=self.action_dim, sigma=noise_scale)
        self.ou_noise.reset()
        self.ddpg_replayer = DDPGReplayer(
            capacity=capacity,
            alpha_initial=alpha_initial,
            alpha_final=alpha_final
        )
        self.sample_rounds = sample_rounds  # 采样轮数
        self.required_sample_size = required_sample_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.soft_update_alpha = soft_update_alpha
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.learn_step_counter = int(0)
        self.single_step = int(0)
        if bool(False):
            log_dir = time.strftime("runs/ddpg_policy/%Y_%m_%d_%H_%M", current_time)
            if self.is_open_writer:
                self.writer = SummaryWriter(log_dir=log_dir)

        self.observation_action_dim = self.action_dim[0] + self.state_dim[0]
        self.ddpg_actor = BuildNetwork(
            in_dim=self.state_dim,
            out_dim=self.action_dim,
            hidden_layers=[128, 256],
            hidden_activation=nn.ReLU,
            out_activation=nn.Tanh,
            optimizer_params={"lr": self.actor_learning_rate}
        )
        self.ddpg_actor_target = BuildNetwork(
            in_dim=self.state_dim,
            out_dim=self.action_dim,
            hidden_layers=[128, 256],
            hidden_activation=nn.ReLU,
            out_activation=nn.Tanh,
        )
        self.ddpg_critic = BuildNetwork(
            in_dim=self.observation_action_dim,
            out_dim=1,
            hidden_layers=[128, 256],
            hidden_activation=nn.ReLU,
            optimizer_params={"lr": self.critic_learning_rate}
        )
        self.ddpg_critic_target = BuildNetwork(
            in_dim=self.observation_action_dim,
            out_dim=1,
            hidden_layers=[128, 256],
            hidden_activation=nn.ReLU,
        )
        self.ddpg_actor_optimizer = self.ddpg_actor.get_optimizer()
        self.ddpg_critic_optimizer = self.ddpg_critic.get_optimizer()

        # 如果加载模型
        if self.load_model:
            checkpoint = torch.load("tools/policy_dir/Pendulum/ddpg_actor.pth", weights_only=True)
            self.ddpg_actor.load_state_dict(checkpoint["model_state_dict"])
            self.ddpg_actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.ddpg_actor_target.load_state_dict(self.ddpg_actor.state_dict())
            logger.info(f"成功加载--->ddpg_actor")
            checkpoint = torch.load("tools/policy_dir/Pendulum/ddpg_critic.pth", weights_only=True)
            self.ddpg_critic.load_state_dict(checkpoint["model_state_dict"])
            self.ddpg_critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.ddpg_critic_target.load_state_dict(self.ddpg_critic.state_dict())
            logger.info(f"成功加载--->ddpg_critic")

        # 学习率调度器
        self.ddpg_actor_scheduler = optim.lr_scheduler.StepLR(
            self.ddpg_actor_optimizer, step_size=200, gamma=0.9
        )

        self.ddpg_critic_scheduler = optim.lr_scheduler.StepLR(
            self.ddpg_critic_optimizer, step_size=200, gamma=0.9
        )

    def actor_decide(self, observation, train):
        if train and self.explore and self.ddpg_replayer.count < self.required_sample_size:
            return np.random.uniform(self.action_low, self.action_high)

        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        tensor_logits = self.ddpg_actor(observation)
        action = tensor_logits.squeeze(0).cpu().detach().numpy()
        if train and self.explore:
            action = np.clip(action + self.ou_noise(), self.action_low, self.action_high)
        return action

    def loader_pool(self, observation, action, reward, next_observation, done, pbar=None):
        remaining = max(0, self.required_sample_size - self.ddpg_replayer.count)  # 计算还需要填充的样本数量
        if remaining > 0:
            if self.ddpg_replayer.count <= self.required_sample_size:
                self.ddpg_replayer.replay_store(
                    observation, action, reward, next_observation, done, priority=0.8, pbar=pbar
                )

    def ddpg_actor_critic_learn(self, observation, action, reward, next_observation, done):
        """
        actor critic算法
        :param observation:当前状态S
        :param action:动作A
        :param reward:奖励R
        :param done:回合是否结束
        :param next_observation:下个状态S'
        :return:None
        """
        self.ddpg_replayer.replay_store(
            observation, action, reward, next_observation, done, priority=0.8,
        )
        if self.ddpg_replayer.count < self.required_sample_size:
            return
        else:
            critic_losses = []
            actor_losses = []

            for _ in range(self.sample_rounds):
                batch, weights, indices = self.ddpg_replayer.replay_sample(self.batch_size)
                # numpy转为tensor
                observations = torch.tensor(batch["observation"], dtype=torch.float32)
                actions = torch.tensor(batch["action"], dtype=torch.float32)
                rewards = torch.tensor(batch["reward"], dtype=torch.float32)
                next_observations = torch.tensor(batch["next_observation"], dtype=torch.float32)
                dones = torch.tensor(batch["done"], dtype=torch.float32)
                # 利用critic状态目标更新
                with torch.no_grad():
                    logger.debug(f"next_observations dim:{next_observations.shape}")
                    next_actions = self.ddpg_actor_target(next_observations)
                    logger.debug(f"next_actions dim:{next_actions.shape}")
                    next_combinations = torch.cat((next_observations, next_actions), dim=-1)
                    q_values = self.ddpg_critic_target(next_combinations)
                    uts = rewards + q_values * self.gamma * (1. - dones)

                # 价值梯度
                combinations = torch.cat((observations, actions), dim=-1)
                predict_qs = self.ddpg_critic(combinations)
                td_error = uts - predict_qs.detach()
                self.ddpg_replayer.update_priorities(indices, td_error.tolist())
                target_qs = uts.detach()
                critic_loss = nn.SmoothL1Loss()(predict_qs, target_qs)
                # Actor部分：计算策略损失
                actions_t = self.ddpg_actor(observations)
                combinations_t = torch.cat((observations, actions_t), dim=-1)
                predict_qs_max = self.ddpg_critic_target(combinations_t)
                logger.debug(f"predict_qs_max:{predict_qs_max}")
                actor_loss = -predict_qs_max.mean()  # 策略梯度损失
                # critic梯度更新
                self.ddpg_critic_optimizer.zero_grad()
                critic_loss.backward()  # 反向传播
                self.ddpg_critic_optimizer.step()  # 更新 Critic 网络
                critic_losses.append(critic_loss)
                # actor梯度更新
                self.ddpg_actor_optimizer.zero_grad()
                actor_loss.backward()  # 反向传播
                self.ddpg_actor_optimizer.step()  # 更新 Actor 网络
                actor_losses.append(actor_loss)

                # 更新学习率
                if self.learn_step_counter % 100 == 0:
                    self.ddpg_actor_scheduler.step()
                    self.ddpg_critic_scheduler.step()

                # 更新参数，main->target
                if self.learn_step_counter > 0 and self.learn_step_counter % 10 == 0:
                    self.soft_update(self.ddpg_actor_target, self.ddpg_actor)
                    self.soft_update(self.ddpg_critic_target, self.ddpg_critic)

                if self.learn_step_counter % 1000 == 0:
                    logger.info(f"self.learn_step_counter: {self.learn_step_counter}")
                    self.ddpg_replayer.clean_old_data_by_age()
                if self.learn_step_counter % 2000 == 0:
                    logger.info(f"self.learn_step_counter: {self.learn_step_counter}")
                    self.ddpg_replayer.clean_old_data()

            critic_loss_r = sum(critic_losses) / len(critic_losses)
            actor_loss_r = sum(actor_losses) / len(actor_losses)
            # 记录训练信息
            if self.is_open_writer:
                self.writer.add_scalar('Loss/Critic', critic_loss_r, self.single_step)
                self.writer.add_scalar('Loss/Actor', actor_loss_r, self.single_step)
                self.writer.add_scalar("Learning Rate/actor",
                                       self.ddpg_actor_scheduler.get_last_lr()[0], self.learn_step_counter)
                self.writer.add_scalar("Learning Rate/critic",
                                       self.ddpg_critic_scheduler.get_last_lr()[0], self.learn_step_counter)
                self.single_step += 1

    def soft_update(self, target, main):
        """
        参数软更新，平滑参数传递过程
        :return:
        """
        for target_params, main_params in zip(target.parameters(), main.parameters()):
            target_params.data.copy_(
                self.soft_update_alpha * main_params + (1 - self.soft_update_alpha) * target_params)

    def close_writer(self):
        self.writer.close()

    def play_actor_critic(self, train=False):
        """
        训练过程
        :param train: bool
        :return:
        """
        # 获取初始状态
        episode_reward = 0.0
        observation, _ = self.env.reset()
        done = False
        # 选择train/eval模式
        if train:
            logger.info(f"-----开启训练模式-----")
            self.ddpg_actor.train()
            self.ddpg_critic.train()
        else:
            logger.info(f"*****开启评估模式*****")
            self.ddpg_actor.eval()
            self.ddpg_critic.eval()

        while not done:
            # 开启动画
            if self.render:
                self.env.render()
            # 智能体决策
            if not train:
                with torch.no_grad():
                    action = self.actor_decide(observation, train)
            else:
                action = self.actor_decide(observation, train)
            # 环境更新
            next_observation, reward, terminated, truncated, _ = self.step(action)
            if train:
                self.loader_pool(observation, action, reward, next_observation, done)

            if terminated or truncated:
                done = True
            # 奖励更新
            episode_reward += reward
            # agent学习
            if train:
                self.ddpg_actor_critic_learn(observation, action, reward, next_observation, done)
            # 达到结束状态
            if done:
                self.learn_step_counter += 1
                logger.info(f"结束一轮游戏, 奖励为${episode_reward}")
                flag = True if episode_reward >= -100 else False
                self.done_rate.append(flag)
                break
            # 状态更新
            observation = next_observation

        return episode_reward


class TD3Agent(DDPGAgent):
    """
    Twin Delay Deep Deterministic Policy Gradient
    """

    def __init__(self, env,
                 actor_learning_rate=0.00001,
                 critic_learning_rate=0.00001,
                 capacity=100000,
                 load_model=True,
                 ):
        super().__init__(env)
        self.env = env
        self.load_model = load_model

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        if bool(True):
            log_dir = time.strftime("runs/td3_policy/%Y_%m_%d_%H_%M", current_time)
            if self.is_open_writer:
                self.writer = SummaryWriter(log_dir=log_dir)

        self.observation_action_dim = self.action_dim[0] + self.state_dim[0]
        self.td3_replayer = TD3Replayer(capacity=capacity)
        # 策略评估网络
        self.td3_actor = BuildNetwork(
            in_dim=self.state_dim,
            out_dim=self.action_dim,
            hidden_layers=[128, 256],
            hidden_activation=nn.ReLU,
            out_activation=nn.Tanh,
            optimizer_params={"lr": self.actor_learning_rate}
        )
        # 策略评估目标
        self.td3_actor_target = BuildNetwork(
            in_dim=self.state_dim,
            out_dim=self.action_dim,
            hidden_layers=[128, 256],
            hidden_activation=nn.ReLU,
            out_activation=nn.Tanh,
        )
        # 价值评估q0和q1
        self.td3_critic_0 = BuildNetwork(
            in_dim=self.observation_action_dim,
            out_dim=1,
            hidden_layers=[128, 256],
            hidden_activation=nn.ReLU,
            optimizer_params={"lr": self.critic_learning_rate}
        )
        self.td3_critic_0_target = BuildNetwork(
            in_dim=self.observation_action_dim,
            out_dim=1,
            hidden_layers=[128, 256],
            hidden_activation=nn.ReLU,
        )
        self.td3_critic_1 = BuildNetwork(
            in_dim=self.observation_action_dim,
            out_dim=1,
            hidden_layers=[128, 256],
            hidden_activation=nn.ReLU,
            optimizer_params={"lr": self.critic_learning_rate}
        )
        self.td3_critic_1_target = BuildNetwork(
            in_dim=self.observation_action_dim,
            out_dim=1,
            hidden_layers=[128, 256],
            hidden_activation=nn.ReLU,
        )
        self.td3_actor_optimizer = self.td3_actor.get_optimizer()
        self.td3_critic_0_optimizer = self.td3_critic_0.get_optimizer()
        self.td3_critic_1_optimizer = self.td3_critic_1.get_optimizer()

        # 如果加载模型
        if self.load_model:
            checkpoint = torch.load("tools/policy_dir/Pendulum/td3_actor.pth", weights_only=True)
            self.td3_actor.load_state_dict(checkpoint["model_state_dict"])
            self.td3_actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.td3_actor_target.load_state_dict(self.td3_actor.state_dict())
            logger.info(f"成功加载--->td3_actor")
            checkpoint = torch.load("tools/policy_dir/Pendulum/td3_critic_0.pth", weights_only=True)
            self.td3_critic_0.load_state_dict(checkpoint["model_state_dict"])
            self.td3_critic_0_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.td3_critic_0_target.load_state_dict(self.td3_critic_0.state_dict())
            logger.info(f"成功加载--->td3_critic_0")
            checkpoint = torch.load("tools/policy_dir/Pendulum/td3_critic_1.pth", weights_only=True)
            self.td3_critic_1.load_state_dict(checkpoint["model_state_dict"])
            self.td3_critic_1_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.td3_critic_1_target.load_state_dict(self.td3_critic_1.state_dict())
            logger.info(f"成功加载--->td3_critic_1")

        # 学习率调度器
        self.td3_actor_scheduler = optim.lr_scheduler.StepLR(
            self.td3_actor_optimizer, step_size=200, gamma=0.9
        )

        self.td3_critic_0_scheduler = optim.lr_scheduler.StepLR(
            self.td3_critic_0_optimizer, step_size=200, gamma=0.9
        )

        self.td3_critic_1_scheduler = optim.lr_scheduler.StepLR(
            self.td3_critic_1_optimizer, step_size=200, gamma=0.9
        )

    def actor_decide(self, observation, train):
        if train and self.explore and self.td3_replayer.count < self.required_sample_size:
            return np.random.uniform(self.action_low, self.action_high)

        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        tensor_logits = self.td3_actor(observation)
        action = tensor_logits.squeeze(0).cpu().detach().numpy()
        if train and self.explore:
            # noise = np.random.normal(0, 0.1, size=action.shape)
            # action = np.clip(action + noise, self.action_low, self.action_high)
            action = np.clip(action + self.ou_noise(), self.action_low, self.action_high)
        return action

    def td3_actor_critic_learn(self, observation, action, reward, next_observation, done):
        """
        actor critic算法
        :param observation:当前状态S
        :param action:动作A
        :param reward:奖励R
        :param done:回合是否结束
        :param next_observation:下个状态S'
        :return:None
        """
        self.td3_replayer.replay_store(
            observation, action, reward, next_observation, done
        )
        if self.td3_replayer.count < self.required_sample_size:
            return
        else:
            critic_losses_0 = []
            critic_losses_1 = []
            actor_losses = []

            for _ in range(self.sample_rounds):
                observations_np, actions_np, rewards_np, next_observations_np, dones_np = (
                    self.td3_replayer.replay_sample(self.batch_size))
                # numpy转为tensor
                observations = torch.tensor(observations_np, dtype=torch.float32)
                actions = torch.tensor(actions_np, dtype=torch.float32)
                rewards = torch.tensor(rewards_np, dtype=torch.float32)
                next_observations = torch.tensor(next_observations_np, dtype=torch.float32)
                dones = torch.tensor(dones_np, dtype=torch.float32)
                # 利用critic状态目标更新
                with torch.no_grad():
                    # 使用目标 Actor 网络
                    next_actions = self.td3_actor_target(next_observations)
                    # 添加目标策略平滑噪声（高斯噪声）
                    noise_scale = 0.1  # 超参数，可调整
                    noise = torch.clamp(torch.randn_like(next_actions) * noise_scale, min=-0.5, max=0.5)  # 裁剪噪声
                    next_actions = next_actions + noise
                    # 裁剪动作到合法范围
                    action_low = torch.tensor(self.action_low, dtype=torch.float32, device=next_observations.device)
                    action_high = torch.tensor(self.action_high, dtype=torch.float32, device=next_observations.device)
                    next_actions = torch.clamp(next_actions, min=action_low, max=action_high)
                    # 拼接状态和动作
                    next_combinations = torch.cat((next_observations, next_actions), dim=-1)
                    # 计算目标 Q 值
                    q0_values = self.td3_critic_0_target(next_combinations)
                    q1_values = self.td3_critic_1_target(next_combinations)
                    q_values = torch.min(q0_values, q1_values).squeeze()
                    # 计算最终目标值
                    uts = rewards + q_values * self.gamma * (1. - dones)

                # 价值梯度q0
                combinations = torch.cat((observations, actions), dim=-1)
                predict_q0s = self.td3_critic_0(combinations)
                target_qs = uts.unsqueeze(dim=-1)
                critic_loss_0 = nn.SmoothL1Loss()(predict_q0s, target_qs)
                # 价值梯度q1
                predict_q1s = self.td3_critic_1(combinations)
                critic_loss_1 = nn.SmoothL1Loss()(predict_q1s, target_qs)
                # Actor部分：计算策略损失
                actions_t = self.td3_actor(observations)  # 没想到这里居然还是ddpg的actor，难怪没有效果，吐血了
                combinations_t = torch.cat((observations, actions_t), dim=-1)
                predict_qs_min = (
                    torch.min(self.td3_critic_0_target(combinations_t), self.td3_critic_1_target(combinations_t)))
                logger.debug(f"predict_qs_max:{predict_qs_min}")
                actor_loss = -predict_qs_min.mean()  # 策略梯度损失
                # critic0梯度更新
                self.td3_critic_0_optimizer.zero_grad()
                critic_loss_0.backward()  # 反向传播
                torch.nn.utils.clip_grad_norm_(self.td3_critic_0.parameters(), max_norm=1)
                self.td3_critic_0_optimizer.step()  # 更新 Critic 网络
                critic_losses_0.append(critic_loss_0)
                # critic1梯度更新
                self.td3_critic_1_optimizer.zero_grad()
                critic_loss_1.backward()  # 反向传播
                torch.nn.utils.clip_grad_norm_(self.td3_critic_1.parameters(), max_norm=1)
                self.td3_critic_1_optimizer.step()  # 更新 Critic 网络
                critic_losses_1.append(critic_loss_1)
                actor_losses.append(actor_loss)
                # actor梯度更新
                if self.learn_step_counter > 0 and self.learn_step_counter % 5 == 0:
                    self.td3_actor_optimizer.zero_grad()
                    actor_loss.backward()  # 反向传播
                    torch.nn.utils.clip_grad_norm_(self.td3_actor.parameters(), max_norm=1)
                    self.td3_actor_optimizer.step()  # 更新 Actor 网络
                    # 更新参数，main->target
                    self.soft_update(self.td3_actor_target, self.td3_actor)
                    self.soft_update(self.td3_critic_0_target, self.td3_critic_0)
                    self.soft_update(self.td3_critic_1_target, self.td3_critic_1)

                # 更新学习率
                if self.learn_step_counter >0 and self.learn_step_counter % 200 == 0:
                    self.td3_actor_scheduler.step()
                    self.td3_critic_0_scheduler.step()
                    self.td3_critic_1_scheduler.step()

            critic_loss_r_0 = sum(critic_losses_0) / len(critic_losses_0)
            critic_loss_r_1 = sum(critic_losses_1) / len(critic_losses_1)

            actor_loss_r = sum(actor_losses) / len(actor_losses)
            # 记录训练信息
            if self.is_open_writer:
                self.writer.add_scalar('Loss/Critic_0', critic_loss_r_0, self.single_step)
                self.writer.add_scalar('Loss/Critic_1', critic_loss_r_1, self.single_step)
                self.writer.add_scalar('Loss/Actor', actor_loss_r, self.single_step)
                self.writer.add_scalar("Learning Rate/actor",
                                       self.td3_actor_scheduler.get_last_lr()[0], self.learn_step_counter)
                self.writer.add_scalar("Learning Rate/critic0",
                                       self.td3_critic_0_scheduler.get_last_lr()[0], self.learn_step_counter)
                self.writer.add_scalar("Learning Rate/critic1",
                                       self.td3_critic_1_scheduler.get_last_lr()[0], self.learn_step_counter)

                self.single_step += 1

    def soft_update(self, target, main):
        """
        参数软更新，平滑参数传递过程
        :return:
        """
        for target_params, main_params in zip(target.parameters(), main.parameters()):
            target_params.data.copy_(
                self.soft_update_alpha * main_params + (1 - self.soft_update_alpha) * target_params)

    def close_writer(self):
        self.writer.close()

    def loader_pool(self, observation, action, reward, next_observation, done, pbar=None):
        remaining = max(0, self.required_sample_size - self.td3_replayer.count)  # 计算还需要填充的样本数量
        if remaining > 0:
            if self.td3_replayer.count <= self.required_sample_size:
                self.td3_replayer.replay_store(
                    observation, action, reward, next_observation, done
                )

    def td3_play_actor_critic(self, train=False):
        """
        训练过程
        :param train: bool
        :return:
        """
        # 获取初始状态
        episode_reward = 0.0
        observation, _ = self.env.reset()
        done = False
        # 选择train/eval模式
        if train:
            logger.info(f"-----开启训练模式-----")
            self.td3_actor.train()
            self.td3_critic_0.train()
            self.td3_critic_1.train()
        else:
            logger.info(f"*****开启评估模式*****")
            self.td3_actor.eval()
            self.td3_critic_0.eval()
            self.td3_critic_1.eval()

        while not done:
            # 开启动画
            if self.render:
                self.env.render()
            # 智能体决策
            if not train:
                with torch.no_grad():
                    action = self.actor_decide(observation, train)
            else:
                action = self.actor_decide(observation, train)
            # 环境更新
            next_observation, reward, terminated, truncated, _ = self.step(action)
            if train:
                self.loader_pool(observation, action, reward, next_observation, done)

            if terminated or truncated:
                done = True
            # 奖励更新
            episode_reward += reward
            # agent学习
            if train:
                self.td3_actor_critic_learn(observation, action, reward, next_observation, done)
            # 达到结束状态
            if done:
                self.learn_step_counter += 1
                logger.info(f"结束一轮游戏, 奖励为${episode_reward}")
                flag = True if episode_reward >= -100 else False
                self.done_rate.append(flag)
                break
            # 状态更新
            observation = next_observation

        return episode_reward


class Pendulum(TD3Agent):
    def __init__(self, name=None, render_mode=render_model[0], render=True):
        # 在Acrobot中创建全局模型和优化器
        if render:
            env = gym.make(name, render_mode=render_mode)
        else:
            env = gym.make(name, render_mode="rgb_array")
        env.reset(seed=seed)
        # 调用父类的初始化
        super().__init__(env=env)

        logger.info(f"搜索顺序:{Pendulum.mro()}")
        self.class_name = self.__class__.__name__

    def game_iteration(self, show_policy, *args, **kwargs):
        """
        迭代
        :param show_policy: 使用的更新策略方式
        """
        episode_rewards = []  # 总轮数的奖励(某轮总奖励)列表
        logger.info(f"*****启动: {show_policy}*****")
        for game_round in range(1, self.game_rounds):
            start_time = time.time()
            logger.info(f"---第{game_round}轮训练---")
            episode_reward = 0
            method_name = ""
            if show_policy == "深度确定性策略梯度算法":
                logger.info(f"深度确定性策略梯度算法")
                episode_reward = self.play_actor_critic(train=False)  # 第round轮次的累积reward
                method_name = self.play_actor_critic.__name__
            if show_policy == "双重延迟深度确定性策略梯度算法":
                logger.info(f"双重延迟深度确定性策略梯度算法")
                episode_reward = self.td3_play_actor_critic(train=False)  # 第round轮次的累积reward
                method_name = self.td3_play_actor_critic.__name__

            if self.global_is_train and self.save_policy and (
                    game_round % 100 == 0 or game_round == self.game_rounds - 1):
                if show_policy == "深度确定性策略梯度算法":
                    save_data = {"ddpg_actor": self.ddpg_actor,
                                 "ddpg_critic": self.ddpg_critic,
                                 "ddpg_actor_optimizer": self.ddpg_actor_optimizer,
                                 "ddpg_critic_optimizer": self.ddpg_critic_optimizer}
                    Policy_loader.save_policy(method_name, self.class_name, save_data, step=game_round)
            if self.global_is_train and self.save_policy and (
                    game_round % 100 == 0 or game_round == self.game_rounds - 1):
                if show_policy == "双重延迟深度确定性策略梯度算法":
                    save_data = {"td3_actor": self.td3_actor,
                                 "td3_critic_0": self.td3_critic_0,
                                 "td3_critic_1": self.td3_critic_1,
                                 "td3_actor_optimizer": self.td3_actor_optimizer,
                                 "td3_critic_0_optimizer": self.td3_critic_0_optimizer,
                                 "td3_critic_1_optimizer": self.td3_critic_1_optimizer
                                 }
                    Policy_loader.save_policy(method_name, self.class_name, save_data, step=game_round)

            if episode_reward is not None:
                episode_rewards.append(episode_reward)
                if self.is_open_writer:
                    if self.learn_step_counter % 10 == 0:  # 每 10 轮记录一次奖励
                        self.writer.add_scalar("Episode Reward", episode_reward,
                                               global_step=self.learn_step_counter)
                if self.global_is_train:
                    if (len(self.done_rate) == 300
                            and np.round(np.mean(episode_rewards[-300:]),
                                         2) >= -100):
                        logger.info(f"!!!平均值大于-100，自动停止训练!!!")
                        break
            else:
                logger.warning(f"第{game_round}轮奖励为 None，已跳过。")

            Visualizer.plot_cumulative_avg_rewards(episode_rewards, game_round, self.game_rounds, self.class_name,
                                                   method_name)
            # 记录结束时间
            end_time = time.time()
            # 计算时间差
            elapsed_time = end_time - start_time
            logger.info(f"第{game_round}轮耗时: {elapsed_time:.2f}秒！")

        self.close_writer()
        print(
            f"平均奖励：{(np.round(np.mean(episode_rewards), 2))} "
            f"= {np.sum(episode_rewards)} / {len(episode_rewards)}")
        print(
            f"最后300轮奖励：{(np.round(np.mean(episode_rewards[-300:]), 2))} = "
            f"{np.sum(episode_rewards[-300:])} / {len(episode_rewards[-300:])}")
        logger.info(f"*****结束: {show_policy}*****")
