# -*- coding: utf-8 -*-
"""
@File    : acrobot.py
@Time    : 2025/3/5 14:47
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    : 
"""
import time
from collections import deque

from torch import optim, dtype
from torch.utils.tensorboard import SummaryWriter
import gym
import numpy as np
from gym.spaces import Box
import torch.nn as nn
import torch
import tensorboard

from envs.env_template import Env
import torch.multiprocessing as mp

from envs.replayer_store import *
from tools.visualizer import Visualizer
from tools.save_policy import Policy_loader
from networks.build_network import BuildNetwork, BuildA3CNetwork
from networks.data_processor import *
from torch.utils.data import DataLoader, TensorDataset
from envs.global_set import *
from gym.vector import SyncVectorEnv
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

seed = 42
current_time = time.localtime()
print(f"\nPyTorch version: {torch.__version__}")  # 需要 >=1.8.0
print(f"TensorBoard version:{tensorboard.__version__}")  # 需要 >=2.4.0


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
            # print("self.observation_space is a Box object!")
            self.State_Num = self.env.observation_space.shape
        else:
            # print("self.observation_space is not a Box object!")
            # # 获取状态空间的大小（假设 FrozenLake 是一个网格地图，这里 nrow 和 ncol 可以直接得到）
            self.State_Num = self.observation_space.n
        if isinstance(self.action_space, Box):
            # print("self.action_space is a Box object!")
            # 获取动作空间的大小，即可选择的动作数量
            self.Action_Num = self.env.action_space.shape
        else:
            # print("self.action_space is not a Box object!")
            # # 获取动作空间的大小，即可选择的动作数量
            self.Action_Num = self.envs.action_space.n

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
        self.load_model = True
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


class QActorCriticAgent(EnvInit):
    """
    动作actor critic算法
    """

    def __init__(self, env, gamma=0.99, learning_rate=0.0005):
        super().__init__(env)
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learn_step_counter = int(0)
        self.temperature = 1.0  # 添加温度参数
        if bool(False):
            log_dir = time.strftime("runs/actor_critic/%Y_%m_%d_%H_%M", current_time)
            if self.is_open_writer:
                self.q_ac_writer = SummaryWriter(log_dir=log_dir)
        self.discount = 1.
        self.actor = BuildNetwork(
            in_dim=self.State_Num,
            out_dim=self.Action_Num,
            hidden_layers=[128, 128, 64],
            hidden_activation=nn.ReLU,
            out_activation=nn.Softmax,
            optimizer_params={"lr": 0.001}
        )
        self.critic = BuildNetwork(
            in_dim=self.State_Num,
            out_dim=self.Action_Num,
            hidden_layers=[128, 128, 64],
            hidden_activation=nn.ReLU,
            optimizer_params={"lr": 0.001}
        )
        self.actor_optimizer = self.actor.get_optimizer()
        self.critic_optimizer = self.critic.get_optimizer()

        # 如果加载模型
        if self.load_model and False:
            checkpoint = torch.load("tools/policy_dir/Acrobot/q_best_actor.pth", weights_only=True)
            self.actor.load_state_dict(checkpoint["model_state_dict"])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"成功加载--->actor")
            checkpoint = torch.load("tools/policy_dir/Acrobot/q_best_critic.pth", weights_only=True)
            self.critic.load_state_dict(checkpoint["model_state_dict"])
            self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"成功加载--->critic")

        # 学习率调度器
        self.policy_scheduler = optim.lr_scheduler.StepLR(
            self.actor_optimizer, step_size=200, gamma=0.9
        )
        self.baseline_scheduler = optim.lr_scheduler.StepLR(
            self.critic_optimizer, step_size=200, gamma=0.9
        )

    def actor_decide(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        probs = self.actor(observation)
        probs = probs.squeeze(0).cpu().detach().numpy()
        # 引入温度参数调节分布
        logits = np.log(probs + 1e-8) / self.temperature  # 防止 log(0)
        probs = np.exp(logits) / np.sum(np.exp(logits))  # 重新归一化

        assert np.isclose(probs.sum(), 1.0), "Probs必须和为1!"
        action = np.random.choice(self.Action_Num, p=probs)

        return action

    def actor_critic_learn(self, observation, action, reward, next_observation, next_action, done):
        """
        actor critic算法
        :param observation:当前状态S
        :param action:动作A
        :param reward:奖励R
        :param done:回合是否结束
        :param next_observation:下个状态S'
        :param next_action: 下个动作A‘
        :return:None
        """
        # numpy转为tensor
        observation = torch.tensor(observation, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_observation = torch.tensor(next_observation, dtype=torch.float32)
        next_action = torch.tensor(next_action, dtype=torch.long)
        # 利用critic状态目标更新
        with torch.no_grad():
            vs = self.critic(next_observation)
            q_value = vs[next_action]
            ut = reward + q_value * self.gamma * (1. - done)

        # ut的tensor转换必须在torch.no_grad()外面
        ut = torch.tensor(ut, dtype=torch.float32)
        # 仅仅是使用当前的去掉梯度的动作价值不可行
        # advantage = self.critic(observation)[action].detach()
        # Critic部分：计算 TD误差
        predict_vs = self.critic(observation)
        advantage = ut - predict_vs.detach()
        predict_qs = predict_vs[action]
        # TODO: predict.detach()
        target_q = ut
        # logger.info(f"cur_q_value: {cur_q_value}, ut: {ut}")

        # Actor部分：计算策略损失
        action_probs = self.actor(observation)  # 动作概率分布
        action_prob = action_probs[action.item()]  # 选择当前动作的概率
        actor_loss = -(torch.log(action_prob) * advantage).mean()  # 策略梯度损失
        # critic梯度更新
        self.critic_optimizer.zero_grad()
        critic_loss = nn.SmoothL1Loss()(predict_qs, target_q)
        critic_loss.backward()  # 反向传播
        self.critic_optimizer.step()  # 更新 Critic 网络
        # actor梯度更新
        self.actor_optimizer.zero_grad()
        actor_loss.backward()  # 反向传播
        self.actor_optimizer.step()  # 更新 Actor 网络

        # 记录训练信息
        if self.is_open_writer:
            self.q_ac_writer.add_scalar('Loss/Critic', critic_loss.item(), self.learn_step_counter)
            self.q_ac_writer.add_scalar('Loss/Actor', actor_loss.item(), self.learn_step_counter)

        # 更新学习率
        self.policy_scheduler.step()
        self.baseline_scheduler.step()

        self.learn_step_counter += 1

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
            self.actor.train()
            self.critic.train()
        else:
            logger.info(f"*****开启评估模式*****")
            self.actor.eval()
            self.critic.eval()
        # agent循环
        while True:
            # 开启动画
            if self.render:
                self.env.render()
            # 智能体决策
            if not train:
                with torch.no_grad():
                    action = self.actor_decide(observation)
            else:
                action = self.actor_decide(observation)
            # 环境更新
            next_observation, reward, terminated, truncated, _ = self.step(action)

            if terminated or truncated:
                done = True
            # 奖励更新
            episode_reward += reward
            # agent学习
            if train:
                next_action = self.actor_decide(next_observation)
                self.actor_critic_learn(observation, action, reward, next_observation, next_action, done)
            # 达到结束状态
            if done:
                logger.info(f"结束一轮游戏, 奖励为${episode_reward}")
                flag = True if episode_reward >= -150 else False
                self.done_rate.append(flag)
                break
            # 状态更新
            observation = next_observation

        return episode_reward


class AActorCriticAgent(QActorCriticAgent):
    """
    优势函数actor critic算法
    """

    def __init__(self, env, gamma=0.99, learning_rate=0.0005):
        super().__init__(env)
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learn_step_counter = int(0)
        self.temperature = 1.0  # 添加温度参数
        if bool(False):
            log_dir = time.strftime("runs/ad_actor_critic/%Y_%m_%d_%H_%M", current_time)
            if self.is_open_writer:
                self.writer = SummaryWriter(log_dir=log_dir)
        self.discount = 1.
        self.ad_actor = BuildNetwork(
            in_dim=self.State_Num,
            out_dim=self.Action_Num,
            hidden_layers=[64, 128],
            hidden_activation=nn.ReLU,
            out_activation=nn.Softmax,
            optimizer_params={"lr": 0.0001}
        )
        self.ad_critic = BuildNetwork(
            in_dim=self.State_Num,
            out_dim=1,
            hidden_layers=[128],
            hidden_activation=nn.ReLU,
            optimizer_params={"lr": 0.001}
        )
        self.ad_actor_optimizer = self.ad_actor.get_optimizer()
        self.ad_critic_optimizer = self.ad_critic.get_optimizer()

        # 如果加载模型
        if self.load_model and False:
            checkpoint = torch.load("tools/policy_dir/Acrobot/ad_actor_best.pth", weights_only=True)
            self.ad_actor.load_state_dict(checkpoint["model_state_dict"])
            self.ad_actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"成功加载--->ad_actor")
            checkpoint = torch.load("tools/policy_dir/Acrobot/ad_critic_best.pth", weights_only=True)
            self.ad_critic.load_state_dict(checkpoint["model_state_dict"])
            self.ad_critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"成功加载--->ad_critic")

        # 学习率调度器
        self.policy_scheduler = optim.lr_scheduler.StepLR(
            self.ad_actor_optimizer, step_size=500, gamma=0.9
        )
        self.baseline_scheduler = optim.lr_scheduler.StepLR(
            self.ad_critic_optimizer, step_size=500, gamma=0.9
        )

    def ad_actor_decide(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        probs = self.ad_actor(observation)
        normalized_probs = action_probs_normalized(probs)
        normalized_probs = normalized_probs.squeeze(0).cpu().detach().numpy()

        # 引入温度参数调节分布
        # logits = np.log(probs + 1e-8) / self.temperature  # 防止 log(0)
        # probs = np.exp(logits) / np.sum(np.exp(logits))  # 重新归一化

        assert np.isclose(normalized_probs.sum(), 1.0), "Probs必须和为1!"
        action = np.random.choice(self.Action_Num, p=normalized_probs)

        return action

    def ad_actor_critic_learn(self, observation, action, reward, next_observation, next_action, done):
        """
        actor critic算法
        :param observation:当前状态S
        :param action:动作A
        :param reward:奖励R
        :param done:回合是否结束
        :param next_observation:下个状态S'
        :param next_action: 下个动作A‘
        :return:None
        """
        # numpy转为tensor
        observation = torch.tensor(observation, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_observation = torch.tensor(next_observation, dtype=torch.float32)
        next_action = torch.tensor(next_action, dtype=torch.long)
        # 利用critic状态目标更新
        with torch.no_grad():
            vs = self.ad_critic(next_observation)
            ut = reward + vs * self.gamma * (1. - done)

        # ut的tensor转换必须在torch.no_grad()外面
        ut = torch.tensor(ut, dtype=torch.float32)
        # 仅仅是使用当前的去掉梯度的动作价值不可行
        # advantage = self.critic(observation)[action].detach()
        # Critic部分：计算 TD误差
        predict_v = self.ad_critic(observation)
        advantage = ut - predict_v.detach()
        # TODO: predict.detach()
        target_v = ut
        # logger.info(f"cur_q_value: {cur_q_value}, ut: {ut}")

        # Actor部分：计算策略损失
        action_probs = self.ad_actor(observation)  # 动作概率分布
        action_prob = action_probs[action.item()]  # 选择当前动作的概率
        actor_loss = -(torch.log(action_prob) * advantage).mean()  # 策略梯度损失
        # critic梯度更新
        self.ad_critic_optimizer.zero_grad()
        critic_loss = nn.SmoothL1Loss()(predict_v, target_v)
        critic_loss.backward()  # 反向传播
        self.ad_critic_optimizer.step()  # 更新 Critic 网络
        # actor梯度更新
        self.ad_actor_optimizer.zero_grad()
        actor_loss.backward()  # 反向传播
        self.ad_actor_optimizer.step()  # 更新 Actor 网络

        # 记录训练信息
        if self.is_open_writer:
            self.writer.add_scalar('Loss/Critic', critic_loss.item(), self.learn_step_counter)
            self.writer.add_scalar('Loss/Actor', actor_loss.item(), self.learn_step_counter)

        # 更新学习率
        self.policy_scheduler.step()
        self.baseline_scheduler.step()

        self.learn_step_counter += 1

    def ad_play_actor_critic(self, train=False):
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
            self.ad_actor.train()
            self.ad_critic.train()
        else:
            logger.info(f"*****开启评估模式*****")
            self.ad_actor.eval()
            self.ad_critic.eval()
        # agent循环
        while True:
            # 开启动画
            if self.render:
                self.env.render()
            # 智能体决策
            if not train:
                with torch.no_grad():
                    action = self.ad_actor_decide(observation)
            else:
                action = self.ad_actor_decide(observation)
            # 环境更新
            next_observation, reward, terminated, truncated, _ = self.step(action)

            if terminated or truncated:
                done = True
            # 奖励更新
            episode_reward += reward
            # agent学习
            if train:
                next_action = self.ad_actor_decide(next_observation)
                self.ad_actor_critic_learn(observation, action, reward, next_observation, next_action, done)
            # 达到结束状态
            if done:
                logger.info(f"结束一轮游戏, 奖励为${episode_reward}")
                flag = True if episode_reward >= -100 else False
                self.done_rate.append(flag)
                break
            # 状态更新
            observation = next_observation

        return episode_reward


class LambdaActorCriticAgent(AActorCriticAgent):
    """
    资格迹actor critic算法
    """

    def __init__(self, env, gamma=0.99, actor_lambda=0.9, critic_lambda=0.9):
        super().__init__(env)
        self.env = env
        self.gamma = gamma
        self.learn_step_counter = int(0)
        self.entropy_coefficient = 0.01
        self.temperature = 1.0  # 添加温度参数
        if bool(False):
            log_dir = time.strftime("runs/lambda_actor_critic/%Y_%m_%d_%H_%M", current_time)
            if self.is_open_writer:
                self.writer = SummaryWriter(log_dir=log_dir)
        self.discount = 1.
        self.lambda_actor = BuildNetwork(
            in_dim=self.State_Num,
            out_dim=self.Action_Num,
            hidden_layers=[64, 128, 64],
            hidden_activation=nn.ReLU,
            out_activation=nn.Softmax,
            optimizer_params={"lr": 0.0001}
        )
        # actor资格迹
        self.actor_e_traces = {name: torch.zeros_like(params) for name, params in self.lambda_actor.named_parameters()}
        self.actor_lambda = actor_lambda
        self.actor_learning_rate = 0.0001

        self.lambda_critic = BuildNetwork(
            in_dim=self.State_Num,
            out_dim=1,
            hidden_layers=[256],
            hidden_activation=nn.ReLU,
        )
        # critic资格迹
        self.critic_e_traces = {name: torch.zeros_like(params) for name, params in
                                self.lambda_critic.named_parameters()}
        self.critic_lambda = critic_lambda
        self.critic_learning_rate = 0.0001

        self.lambda_actor_optimizer = self.lambda_actor.get_optimizer()

        # 学习率调度器
        self.lambda_actor_scheduler = optim.lr_scheduler.StepLR(
            self.lambda_actor_optimizer, step_size=500, gamma=0.9
        )

        # 如果加载模型
        if self.load_model and False:
            checkpoint = torch.load("tools/policy_dir/Acrobot/lambda_actor.pth", weights_only=True)
            self.lambda_actor.load_state_dict(checkpoint["model_state_dict"])
            self.lambda_actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.actor_e_traces = checkpoint["actor_e_traces"]
            logger.info(f"成功加载--->lambda_actor")
            checkpoint = torch.load("tools/policy_dir/Acrobot/lambda_critic.pth", weights_only=True)
            self.lambda_critic.load_state_dict(checkpoint["model_state_dict"])
            self.critic_e_traces = checkpoint["critic_e_traces"]
            logger.info(f"成功加载--->lambda_critic")

    def lambda_actor_decide(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        probs = self.lambda_actor(observation)
        probs = probs.squeeze(0).cpu().detach().numpy()

        # 引入温度参数调节分布
        logits = np.log(probs + 1e-8) / self.temperature  # 防止 log(0)
        probs = np.exp(logits) / np.sum(np.exp(logits))  # 重新归一化

        assert np.isclose(probs.sum(), 1.0), "Probs必须和为1!"
        action = np.random.choice(self.Action_Num, p=probs)

        return action

    def lambda_actor_critic_learn(self, observation, action, reward, next_observation, done):
        """
        actor critic算法
        :param observation:当前状态S
        :param action:动作A
        :param reward:奖励R
        :param done:回合是否结束
        :param next_observation:下个状态S'
        :return:None
        """
        # numpy转为tensor
        observation = torch.tensor(observation, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = np.clip(reward, -1.0, 1.0)
        reward = torch.tensor(reward, dtype=torch.float32)

        next_observation = torch.tensor(next_observation, dtype=torch.float32)
        # 利用critic状态目标更新
        with torch.no_grad():
            vs = self.lambda_critic(next_observation)
            ut = reward + vs * self.gamma * (1. - done)
            target_v = ut
        # ut的tensor转换必须在torch.no_grad()外面
        # 仅仅是使用当前的去掉梯度的动作价值不可行
        # Critic部分：计算 TD误差
        predict_v = self.lambda_critic(observation)
        advantage = ut - predict_v.detach()
        # 确保TD_error是标量
        # Actor部分：计算策略损失
        action_probs = self.lambda_actor(observation)
        action_prob = torch.softmax(action_probs, dim=-1)[action.item()]
        actor_loss = -(torch.log(action_prob + 1e-8) * advantage).mean()

        # critic梯度更新
        self.lambda_critic.zero_grad()  # 清零梯度
        critic_loss = nn.SmoothL1Loss()(predict_v, target_v)
        critic_loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(self.lambda_critic.parameters(), max_norm=1)  # 防止梯度爆炸
        # self.lambda_critic_optimizer.step()
        # 更新资格迹
        with torch.no_grad():
            for name, param in self.lambda_critic.named_parameters():
                grad_norm = torch.norm(param.grad)
                if grad_norm > 0:
                    self.critic_e_traces[name] = \
                        self.gamma * self.critic_lambda * self.critic_e_traces[name] + param.grad

                self.critic_e_traces[name] = torch.clamp(self.critic_e_traces[name], -1, 1)
        # 使用资格迹更新 Actor 网络参数
        with torch.no_grad():
            for name, param in self.lambda_critic.named_parameters():
                param += self.critic_learning_rate * self.critic_e_traces[name]

        # actor梯度更新
        self.lambda_actor.zero_grad()  # 清零梯度
        actor_loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(self.lambda_actor.parameters(), max_norm=1)  # 防止梯度爆炸

        # 更新资格迹
        with torch.no_grad():
            for name, param in self.lambda_actor.named_parameters():
                grad_norm = torch.norm(param.grad)
                if grad_norm > 0:
                    self.actor_e_traces[name] = \
                        self.gamma * self.actor_lambda * self.actor_e_traces[name] + param.grad
                self.actor_e_traces[name] = torch.clamp(self.actor_e_traces[name], -1, 1)
        # # 使用资格迹更新 Actor 网络参数
        with torch.no_grad():
            for name, param in self.lambda_actor.named_parameters():
                param.grad = self.actor_e_traces[name] * self.actor_learning_rate
            self.lambda_actor_optimizer.step()

        # 记录训练信息
        if self.is_open_writer:
            self.writer.add_scalar('Loss/U_t', ut.item(), self.learn_step_counter)
            self.writer.add_scalar('Loss/Predict V_s', predict_v.item(), self.learn_step_counter)
            self.writer.add_scalar('Loss/Lambda Critic', critic_loss.item(), self.learn_step_counter)
            self.writer.add_scalar('Loss/Lambda Actor', actor_loss.item(), self.learn_step_counter)
            self.writer.add_scalar('Hyperparameters/Temperature', self.temperature, self.learn_step_counter)
            self.writer.add_scalar('Hyperparameters/Actor LR', self.actor_learning_rate, self.learn_step_counter)
            self.writer.add_scalar('Hyperparameters/Critic LR', self.critic_learning_rate, self.learn_step_counter)
            # self.writer.add_scalar('Advantage', advantage.item(), self.learn_step_counter)
            self.writer.add_scalar('TD_Error', advantage.item(), self.learn_step_counter)
            self.writer.add_scalar('Action_Prob', action_prob.item(), self.learn_step_counter)
        self.learn_step_counter += 1
        self.lambda_actor_scheduler.step()

    def lambda_play_actor_critic(self, train=False):
        """
        训练过程
        :param train: bool
        :return:
        """
        # 获取初始状态
        episode_reward = 0.0
        observation, _ = self.env.reset()
        # 选择train/eval模式
        if train:
            logger.info(f"-----开启训练模式-----")
            self.lambda_actor.train()
            self.lambda_critic.train()
        else:
            logger.info(f"*****开启评估模式*****")
            self.lambda_actor.eval()
            self.lambda_critic.eval()
        # agent循环
        while True:
            # 开启动画
            if self.render:
                self.env.render()
            # 智能体决策
            if not train:
                with torch.no_grad():
                    action = self.lambda_actor_decide(observation)
            else:
                action = self.lambda_actor_decide(observation)

            # 环境更新
            next_observation, reward, terminated, truncated, _ = self.step(action)
            done = terminated or truncated
            # 奖励更新
            episode_reward += reward
            # agent学习
            if train:
                self.lambda_actor_critic_learn(observation, action, reward, next_observation, done)
            # 达到结束状态
            if done:
                # 初始化资格迹
                if self.learn_step_counter > 1 and self.learn_step_counter % (500 * 3) == 0:
                    self.actor_learning_rate *= 0.95
                    self.critic_learning_rate *= 0.95
                self.temperature = max(0.1, self.temperature * 0.995)  # 每次训练降低温度
                self.critic_e_traces = {name: torch.zeros_like(param)
                                        for name, param in self.lambda_critic.named_parameters()}
                self.actor_e_traces = {name: torch.zeros_like(param)
                                       for name, param in self.lambda_actor.named_parameters()}
                self.discount = 1
                logger.info(f"结束一轮游戏, 奖励为${episode_reward}")
                flag = True if episode_reward >= -100 else False
                self.done_rate.append(flag)
                break
            # 状态更新
            observation = next_observation

        return episode_reward


def wrap(angle, min_val, max_val):
    return ((angle - min_val) % (max_val - min_val)) + min_val


class A2CActorCriticAgent(LambdaActorCriticAgent):
    """
    多并行采样actor critic算法
    """

    def __init__(self, env, gamma=0.99, num_envs=1):
        super().__init__(env)
        self.env = env
        self.steps = None
        self.gamma = gamma
        self.num_envs = num_envs
        # 创建多个并行环境
        self.a2c_envs = SyncVectorEnv([
            lambda: gym.make('Acrobot-v1', disable_env_checker=True) for _ in range(self.num_envs)
        ])
        # print(f"Max episode steps: {self.a2c_envs.envs[0].spec.max_episode_steps}")
        self.learn_step_counter = int(0)
        self.temperature = 1.0  # 添加温度参数
        self.load_model = True

        if bool(False):
            log_dir = time.strftime("runs/a2c_actor_critic/%Y_%m_%d_%H_%M", current_time)
            if self.is_open_writer:
                self.writer = SummaryWriter(log_dir=log_dir)

        self.a2c_actor = BuildNetwork(
            in_dim=self.State_Num,
            out_dim=self.Action_Num,
            hidden_layers=[64, 128, 64],
            hidden_activation=nn.ReLU,
            out_activation=nn.Softmax,
            optimizer_params={"lr": 0.0001}
        )
        self.a2c_critic = BuildNetwork(
            in_dim=self.State_Num,
            out_dim=1,
            hidden_layers=[128],
            hidden_activation=nn.ReLU,
            optimizer_params={"lr": 0.001}
        )
        self.a2c_actor_optimizer = self.a2c_actor.get_optimizer()
        self.a2c_critic_optimizer = self.a2c_critic.get_optimizer()

        # 如果加载模型
        self.load_model = False
        if self.load_model:
            checkpoint = torch.load("tools/policy_dir/Acrobot/a2c_actor_best.pth", weights_only=True)
            self.a2c_actor.load_state_dict(checkpoint["model_state_dict"])
            self.a2c_actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"成功加载--->a2c_actor")
            checkpoint = torch.load("tools/policy_dir/Acrobot/a2c_critic_best.pth", weights_only=True)
            self.a2c_critic.load_state_dict(checkpoint["model_state_dict"])
            self.a2c_critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"成功加载--->a2c_critic")

        # 学习率调度器
        self.a2c_actor_scheduler = optim.lr_scheduler.StepLR(
            self.a2c_actor_optimizer, step_size=500, gamma=0.9
        )
        self.a2c_critic_scheduler = optim.lr_scheduler.StepLR(
            self.a2c_critic_optimizer, step_size=500, gamma=0.9
        )

    def a2c_actor_decide(self, observations):
        observations = torch.tensor(observations, dtype=torch.float32)
        # 确保 observations 是二维的 [num_envs, state_dim]
        if len(observations.shape) == 1:  # 如果传入的是单状态
            observations = observations.unsqueeze(0)
        observations = torch.tensor(observations, dtype=torch.float32)
        # 因为a2c_actor需要tensor，所以前面需要转换
        probs = self.a2c_actor(observations)
        # probs 是一个二维张量，形状为 (batch_size, Action_Num)
        logits = torch.log(probs + 1e-8) / self.temperature  # 温度缩放
        probs_normalized = torch.softmax(logits, dim=-1)  # 使用 softmax 归一化

        # 检查所有样本的概率归一化是否正确
        assert torch.allclose(probs_normalized.sum(dim=-1), torch.ones(probs_normalized.size(0)),
                              atol=1e-6), "Probs 必须和为 1!"

        # 根据概率分布采样动作，直接对整个批量操作
        actions = torch.multinomial(probs_normalized, num_samples=1).squeeze(-1)  # 形状为 (batch_size,)
        actions = actions.cpu().detach().numpy()

        return np.array(actions)  # [num_envs]

    def a2c_actor_critic_learn(self, observations, actions, rewards, next_observations, dones):
        """
        actor critic算法
        :param observations:当前状态S
        :param actions:动作A
        :param rewards:奖励R
        :param dones:回合是否结束
        :param next_observations:下个状态S'
        :return:None
        """
        # numpy转为tensor
        observations = torch.tensor(observations, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_observations = torch.tensor(next_observations, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        # 利用critic状态目标更新
        with torch.no_grad():
            next_values = self.a2c_critic(next_observations)  # [num_envs, 1]
            td_targets = rewards + self.gamma * next_values.squeeze() * (1 - dones)  # [num_envs]
        current_values = self.a2c_critic(observations).squeeze()  # [num_envs]
        advantages = td_targets - current_values.detach()  # [num_envs]

        # Critic部分：计算值函数损失
        critic_loss = nn.SmoothL1Loss()(current_values, td_targets)
        # Actor部分：计算策略损失
        action_probs = self.a2c_actor(observations)  # (batch_size, action_dim)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)  # [num_envs]
        actor_loss = -(log_probs * advantages).mean()  # 策略梯度损失

        # critic梯度更新
        self.a2c_critic_optimizer.zero_grad()
        critic_loss.backward()  # 反向传播
        self.a2c_critic_optimizer.step()  # 更新 Critic 网络
        # actor梯度更新
        self.a2c_actor_optimizer.zero_grad()
        actor_loss.backward()  # 反向传播
        self.a2c_actor_optimizer.step()  # 更新 Actor 网络

        # 记录训练信息
        if self.is_open_writer:
            self.writer.add_scalar('Loss/Critic', critic_loss.item(), self.learn_step_counter)
            self.writer.add_scalar('Loss/Actor', actor_loss.item(), self.learn_step_counter)
            self.writer.add_scalar('Loss/advantages', advantages.mean().item(), self.learn_step_counter)

        # 更新学习率
        self.a2c_actor_scheduler.step()
        self.a2c_critic_scheduler.step()

        self.learn_step_counter += 1

    def a2c_play_actor_critic(self, train=False):
        avg_reward = 0.0
        episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        observations, _ = self.a2c_envs.reset(seed=[seed + i for i in range(self.num_envs)])
        done = False
        self.steps = np.zeros(self.num_envs, dtype=int)

        if train:
            logger.info(f"-----开启训练模式-----")
            self.a2c_actor.train()
            self.a2c_critic.train()
        else:
            logger.info(f"*****开启评估模式*****")
            self.a2c_actor.eval()
            self.a2c_critic.eval()
            if self.render:
                self.env.reset()

        step = 0
        while not done:
            step += 1
            actions = self.a2c_actor_decide(observations)
            next_observations, rewards, terminates, truncates, _ = self.a2c_envs.step(actions)
            self.steps += 1

            computed_terminates = [(-obs[0] - obs[2]) > 1.0 for obs in observations]
            if list(terminates) != computed_terminates:
                logger.info(f"Termination mismatch! Expected: {computed_terminates}, Got: {terminates}")
                terminates = np.array(computed_terminates)
            truncates = self.steps >= 500

            for i in range(self.num_envs):
                height = -next_observations[i][0] - next_observations[i][2]
                logger.info(f"Step: {step}, Env {i}, Action: {actions[i]}, Height: {height}")

            dones = np.logical_or(terminates, truncates)
            episode_rewards += rewards

            def render_success(time_inter):
                for i in range(self.num_envs):
                    obs = observations[i]  # 使用当前状态
                    theta1 = np.arctan2(obs[1], obs[0])
                    theta2_abs = np.arctan2(obs[3], obs[2])
                    theta2 = theta2_abs - theta1
                    theta1 = wrap(theta1, -np.pi, np.pi)
                    theta2 = wrap(theta2, -np.pi, np.pi)
                    state = np.array([theta1, theta2, obs[4], obs[5]])
                    # logger.info(f"Rendering Step {step}, Env {i}: theta1={np.degrees(theta1):.2f}°,"
                    #             f"theta2={np.degrees(theta2):.2f}°, height={-obs[0] - obs[2]}")
                    self.env.unwrapped.state = state
                    self.env.render()
                    time.sleep(time_inter)  # 动态渲染

            # 持续渲染当前状态
            if self.render and not train:
                render_success(0.1)

            # 记录触发时的状态
            terminated_states = [None] * self.num_envs
            if any(terminates):
                for i in range(self.num_envs):
                    if terminates[i]:
                        terminated_states[i] = observations[i]
                        height = -observations[i][0] - observations[i][2]
                        logger.info(f"Terminated at Step {step}, Env {i}: height={height}, obs={observations[i]}")
                logger.info(f"Terminates: {terminates}, Truncates: {truncates}")

                # 渲染终止状态并暂停
                if self.render and not train:
                    render_success(1)
                    break

            if train:
                self.a2c_actor_critic_learn(observations, actions, rewards, next_observations, dones)

            observations = next_observations
            if any(dones) or step >= 500:
                done = True

            if done:
                avg_reward = np.mean(episode_rewards)
                final_heights = [-obs[0] - obs[2] for obs in next_observations]
                avg_height = np.mean(final_heights)
                logger.info(f"结束一轮游戏, 奖励为${np.round(avg_reward, 3)}, Avg height: {avg_height}")
                flag = True if avg_height > 1.0 else False
                self.done_rate.append(flag)

        return avg_reward


# 启用PyTorch梯度异常检测
# torch.autograd.set_detect_anomaly(True)  # 训练完成后关闭以提升性能


class A3CActorCriticAgent(A2CActorCriticAgent, torch.multiprocessing.Process):
    def __init__(self,
                 env=None,
                 global_a3c_model=None,
                 global_optimizer=None,
                 local_a3c_model=None,
                 gamma=0.99,
                 worker_id=0
                 ):
        super().__init__(env)
        # thread door
        self.env = env
        self.env.reset(seed=(seed + worker_id))
        self.open_multi_thread = True
        self.temperature = 1.0
        self.learn_step_counter = int(0)
        self.n_step = 20
        # define a A3C model
        self.global_a3c_model = global_a3c_model
        # optimizer
        self.global_optimizer = global_optimizer

        self.worker_id = worker_id
        self.gamma = gamma

        # local model
        self.local_a3c_model = local_a3c_model

        # 学习率调度器
        self.a3c_global_scheduler = optim.lr_scheduler.StepLR(
            self.global_optimizer, step_size=200, gamma=0.9
        )
        if bool(False):
            log_dir = time.strftime("runs/a3c_actor_critic/%Y_%m_%d_%H_%M", current_time)
            if self.is_open_writer:
                self.writer = SummaryWriter(log_dir=log_dir)

    def run(self, train=False, metrics_queue=None):
        self.local_a3c_model_init()  # 局部模型延迟初始化，防止过早传入
        return self.a3c_play(train=train, metrics_queue=metrics_queue)

    def local_a3c_model_init(self):
        # 直接使用本地模型优化器（单线程无需全局模型）
        self.local_a3c_model.load_state_dict(state_dict=self.global_a3c_model.state_dict())

    def a3c_agent_decide(self, state):
        """
        decision
        :param state:
        :return:
        """
        observation = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        policy, _ = self.local_a3c_model(observation)
        normalized_probs = action_probs_normalized(policy)
        normalized_probs = normalized_probs.squeeze(0).cpu().detach().numpy()
        # 引入温度参数调节分布
        # logits = np.log(normalized_action + 1e-8) / self.temperature  # 防止 log(0)
        # probs = np.exp(logits) / np.sum(np.exp(logits))  # 重新归一化
        #
        # assert np.isclose(probs.sum(), 1.0), "Probs必须和为1!"
        choose_action = np.random.choice(self.Action_Num, p=normalized_probs)

        return choose_action

    def a3c_agent_learn(self, states, actions, rewards, done, metrics_queue=None):
        """
        A3C algorithm
        :param states:
        :param actions:
        :param rewards:
        :param done:
        :return:
        """
        # 归一化时转换为 numpy 数组
        if len(rewards) >= self.n_step or done:
            if self.n_step == 1:
                logger.info(f"-----a3c学习策略-----")
            # normalized_rewards = (np.array(rewards) - np.mean(rewards)) / (np.std(rewards) + 1e-8)
            # 计算折扣回报
            # 奖励归一化
            # rewards = np.array(rewards)
            # rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

            discounted_rewards = []
            R = 0
            with torch.no_grad():
                for r in reversed(rewards):
                    R = r + R * self.gamma
                    discounted_rewards.insert(0, R)
                discounted_rewards = torch.tensor(np.array(discounted_rewards), dtype=torch.float32)

            # calculate advantage
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.int64)

            # 修正方案：使用global_model进行前向计算
            policies, values = self.local_a3c_model(states_tensor)

            # 分步计算保持计算图
            current_values = values.squeeze(1)

            advantage = discounted_rewards.detach() - values.detach()
            action_probs = policies.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

            # get loss
            # 添加熵正则化
            entropy = -torch.sum(policies * torch.log(policies + 1e-8), dim=-1).mean()
            actor_loss = -(torch.log(action_probs) * advantage).mean() - 0.05 * entropy  # 熵系数 0.01
            # actor_loss = -(torch.log(action_probs) * advantage).mean()
            critic_loss = nn.SmoothL1Loss()(current_values, discounted_rewards)
            total_loss = actor_loss + critic_loss

            # 在反向传播前插入检查点
            from torchviz import make_dot
            # make_dot(total_loss, params=dict(self.global_a3c_model.named_parameters())).render("a3c_graph")

            # update global loss
            self.global_optimizer.zero_grad()
            # 将损失和优势值放入队列
            if self.open_multi_thread:
                if metrics_queue is not None:
                    metrics = {
                        "actor_loss": actor_loss.item(),
                        "critic_loss": critic_loss.item(),
                        "total_loss": total_loss.item(),
                        "advantage_mean": advantage.mean().item(),
                        # "entropy": entropy.item()
                    }
                    metrics_queue.put(metrics)
            else:
                self.writer.add_scalar("Loss/Actor", actor_loss.item(), self.learn_step_counter)
                self.writer.add_scalar("Loss/Critic", critic_loss.item(), self.learn_step_counter)
                self.writer.add_scalar("Loss/Total", total_loss.item(), self.learn_step_counter)
                # self.writer.add_scalar("Entropy", entropy.item(), self.learn_step_counter)

            total_loss.backward()
            # 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.global_a3c_model.parameters(), max_norm=1)

            # 更新全局模型
            self.global_optimizer.step()
            self.a3c_global_scheduler.step()
            # update synchronously
            if self.learn_step_counter > 1 and self.learn_step_counter % 50 == 0:  # 每 5 次学习同步一次
                self.local_a3c_model.load_state_dict(state_dict=self.global_a3c_model.state_dict())
            # 清空缓存
            states.clear()
            actions.clear()
            rewards.clear()

    def a3c_play(self, train=False, metrics_queue=None):
        """
        训练过程
        :param train: bool
        :return:
        """
        # 获取初始状态
        episode_reward = np.float64(0.0)
        observation, _ = self.env.reset()
        done = False
        # 选择train/eval模式
        if train:
            logger.info(f"-----开启训练模式-----")
            self.global_a3c_model.train()
            self.local_a3c_model.train()
        else:
            logger.info(f"*****开启评估模式*****")
            self.global_a3c_model.eval()
            self.local_a3c_model.eval()
        observations = []
        actions = []
        rewards = []
        # agent循环
        while True:
            # 开启动画
            if self.render:
                self.env.render()
            # 智能体决策
            if not train:
                with torch.no_grad():
                    action = self.a3c_agent_decide(observation)
            else:
                action = self.a3c_agent_decide(observation)
            # 环境更新
            next_observation, reward, terminated, truncated, _ = self.env.step(action)

            if terminated or truncated:
                done = True
            # 奖励更新
            episode_reward += reward
            # agent学习
            if train:
                self.a3c_agent_learn(observations, actions, rewards, done, metrics_queue)
            # 达到结束状态
            if done:
                logger.info(f"结束一轮游戏, 奖励为${episode_reward}")
                flag = True if episode_reward >= -100 else False

                if self.learn_step_counter > 0 and self.learn_step_counter % 300 == 0:
                    self.temperature = max(0.1, self.temperature * 0.995)  # 改为按episode衰减
                if bool(False):
                    self.writer.add_scalar("Temperature", self.temperature,
                                           self.learn_step_counter)
                self.done_rate.append(flag)
                observations.clear()
                actions.clear()
                rewards.clear()
                self.learn_step_counter += 1
                break

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            # 状态更新
            observation = next_observation

        return episode_reward


def worker_process(worker_id, reward_queue, metrics_queue, train, global_a3c_model, global_optimizer):
    """
    child thread run
    :param worker_id:
    :param reward_queue:
    :param metrics_queue:
    :param train:
    :param global_a3c_model:
    :param global_optimizer:
    :return:
    """
    try:
        # 子线程任务
        if bool(True):
            env = gym.make("Acrobot-v1", render_mode=render_model[0])
        else:
            env = gym.make("Acrobot-v1")
        local_a3c_model = BuildA3CNetwork(
            in_state_dim=global_a3c_model.in_state_dim,
            out_value_dim=global_a3c_model.out_value_dim,
            out_action_dim=global_a3c_model.out_action_dim,
            hidden_layer=global_a3c_model.hidden_layer
        )
        local_a3c_model.load_state_dict(global_a3c_model.state_dict())
        agent = A3CActorCriticAgent(env=env,
                                    global_a3c_model=global_a3c_model,
                                    global_optimizer=global_optimizer,
                                    local_a3c_model=local_a3c_model,
                                    worker_id=worker_id
                                    )
        episode_reward = agent.run(train=train, metrics_queue=metrics_queue)  # 传递metrics_queue
        reward_queue.put_nowait(episode_reward)
        pass
    except Exception as e:
        print(f"Worker-{worker_id} encountered an error: {e}")


from multiprocessing import Value


class Worker(A3CActorCriticAgent):
    def __init__(self, env, global_a3c_model, global_optimizer, num_workers=4):
        # 初始化全局模型和优化器
        super().__init__(env, global_a3c_model, global_optimizer)
        self.global_a3c_model = global_a3c_model
        self.num_workers = num_workers if num_workers is not None else mp.cpu_count()
        self.global_step = Value('i', 0)  # 共享整数，初始值为 0
        self.lock = mp.Lock()  # 用于同步的锁
        self.metrics_queue = mp.Queue()  # 用于收集损失和优势值
        if bool(False):
            log_dir = time.strftime("runs/a3c_actor_critic/%Y_%m_%d_%H_%M", current_time)
            if self.is_open_writer:
                self.writer = SummaryWriter(log_dir=log_dir)

        self.lock = mp.Lock()  # 保护writer和global_step

    def run_workers(self, train):
        """
        run mutil-threading
        :return:
        """
        workers = []
        reward_queue = mp.Queue()
        for worker_id in range(self.num_workers):
            p = mp.Process(target=worker_process,
                           name=f"Worker-{worker_id + 1}",  # 手动命名
                           args=(
                               worker_id,
                               reward_queue,
                               self.metrics_queue,
                               train,
                               self.global_a3c_model,  # 传递全局模型
                               self.global_optimizer  # 传递全局优化器
                           )
                           )
            workers.append(p)
            p.start()
            # logger.info(f"开启第{worker_id + 1}个子线程")

        # 主线程定期汇总并记录metrics
        while any(w.is_alive() for w in workers):
            metrics_list = []
            while not self.metrics_queue.empty():
                metrics_list.append(self.metrics_queue.get_nowait())  # 非阻塞获取
            if metrics_list:
                avg_metrics = {
                    key: sum(m[key] for m in metrics_list) / len(metrics_list)
                    for key in metrics_list[0].keys()
                }
                with self.lock:
                    self.writer.add_scalar("Loss/Actor", avg_metrics["actor_loss"], int(self.global_step.value))
                    self.writer.add_scalar("Loss/Critic", avg_metrics["critic_loss"], int(self.global_step.value))
                    self.writer.add_scalar("Loss/Total", avg_metrics["total_loss"], int(self.global_step.value))
                    self.writer.add_scalar("Advantage/Mean", avg_metrics["advantage_mean"],
                                           int(self.global_step.value))
                    # self.a3c_writer.add_scalar("Entropy", avg_metrics["entropy"], int(self.global_step.value))
                    self.global_step.value += 1

        for worker in workers:
            if worker.is_alive():
                print(f"{worker.name} is alive")

        for worker in workers:
            # worker.terminate()  # 强制终止子线程
            worker.join()
            # logger.info(f"关闭第{worker.name}子线程")

        results = []
        while not reward_queue.empty():
            results.append(reward_queue.get_nowait())
        # 计算平均值
        if len(results) > 0:
            average_result = sum(results) / len(results) if results else 0.0
            # 在主线程记录平均奖励
            self.writer.add_scalar("Reward/Average", average_result, int(self.global_step.value))
            # self.global_step.value += 1
            # print(f"Average result: {average_result}")
        else:
            average_result = 0.0
        # self.close_writer()  # 关闭writer，确保数据写入
        reward_queue.close()
        reward_queue.join_thread()

        return average_result

    def close_writer(self):
        self.writer.close()


class PPOActorCriticAgent(Worker):
    """
    PPO算法
    """

    def __init__(self, env, global_a3c_model, global_optimizer,
                 clip_ratio=0.2,
                 batch_size=64,
                 ):
        super().__init__(env, global_a3c_model, global_optimizer)
        self.env = env
        self.env.reset(seed=seed)
        self.temperature = 1
        self.gamma = 0.99
        self.lamda = 0.95

        self.learn_step_counter = int(0)
        if bool(False):
            log_dir = time.strftime("runs/ppo_actor_critic/%Y_%m_%d_%H_%M", current_time)
            if self.is_open_writer:
                self.writer = SummaryWriter(log_dir=log_dir)
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        self.trajectory = []
        # PPO 策略网络
        self.ppo_actor = BuildNetwork(
            in_dim=self.State_Num,
            out_dim=self.Action_Num,
            hidden_layers=[64, 128],
            out_activation=None,
            optimizer_params={"lr": 0.0001}
        )
        self.ppo_actor_optim = self.ppo_actor.get_optimizer()

        # PPO 价值网络
        self.ppo_critic = BuildNetwork(
            in_dim=self.State_Num,
            out_dim=1,
            hidden_layers=[256],
            out_activation=None,
            optimizer_params={"lr": 0.0001}
        )
        self.ppo_critic_optim = self.ppo_critic.get_optimizer()
        # 如果加载模型
        self.load_model = False
        if self.load_model:
            checkpoint = torch.load("tools/policy_dir/Acrobot/ppo_actor.pth", weights_only=True)
            self.ppo_actor.load_state_dict(checkpoint["model_state_dict"])
            self.ppo_actor_optim.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"成功加载--->ppo_actor")
            checkpoint = torch.load("tools/policy_dir/Acrobot/ppo_critic.pth", weights_only=True)
            self.ppo_critic.load_state_dict(checkpoint["model_state_dict"])
            self.ppo_critic_optim.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"成功加载--->ppo_critic")
        # 学习率调度器
        self.ppo_actor_scheduler = optim.lr_scheduler.StepLR(
            self.ppo_actor_optim, step_size=500, gamma=0.9
        )
        self.ppo_critic_scheduler = optim.lr_scheduler.StepLR(
            self.ppo_critic_optim, step_size=500, gamma=0.9
        )

    def compute_advantage(self, rewards, values, next_values):
        """
        计算优势函数：GAE
        :param rewards:
        :param values:
        :return:
        """
        # logger.info(f"values before flip: {values}")
        rewards = rewards.flip(dims=[0])
        next_values = next_values.flip(dims=[0])
        if next_values.dim() == 0:
            next_values = next_values.unsqueeze(0)
        values = values.flip(dims=[0])
        if values.dim() == 0:
            values = values.unsqueeze(0)
        # logger.info(f"values after flip: {values}")
        # values由于cat多了一位
        advantages = torch.zeros_like(rewards)
        td_targets = torch.zeros_like(rewards)
        generalized_advantage_estimation = 0
        with torch.no_grad():
            for i in range(len(rewards)):
                ut = rewards[i] + self.gamma * next_values[i]  # 跟values有关系，很大关系
                td_targets[i] = ut
                td_error = ut - values[i]
                generalized_advantage_estimation = td_error + self.gamma * self.lamda * generalized_advantage_estimation
                advantages[i] = generalized_advantage_estimation
        advantages = advantages.flip(dims=[0])
        td_targets = td_targets.flip(dims=[0])
        return advantages, td_targets

    def ppo_decide(self, observations):
        """
        根据当前策略网络选择动作
        :param observations: 当前的状态 (可以是单个样本或批量样本)
        :return: 动作 (单个整数或批量动作) 和 log_probs (PyTorch 张量)
        """
        # 将 observations 转换为 PyTorch 张量
        observations = torch.tensor(observations, dtype=torch.float32)

        # 如果是单个样本，添加批量维度
        if len(observations.shape) == 1:  # 如果是单个样本
            observations = observations.unsqueeze(0)  # [1, State_Num]

        # 获取动作概率分布
        logits = self.ppo_actor(observations)  # [batch_size, Action_Num]

        # 通过温度系数调节分布
        adjust_logits = logits / self.temperature

        # 归一化动作概率分布
        action_dist = torch.distributions.Categorical(logits=adjust_logits)

        # 采样动作
        actions = action_dist.sample()  # 动作索引 (batch_size,)
        log_probs = action_dist.log_prob(actions)  # 动作的 log 概率 (batch_size,)

        # 如果输入是批量样本，返回批量动作和 log_probs
        return actions, log_probs, action_dist.probs

    @staticmethod
    def check_nan_inf(tensor, name):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logger.error(f"{name} contains Nan or Inf.")
        else:
            logger.debug(f"{name} is clean,")

    def poo_learn(self, observations, actions, rewards, next_observations, old_log_probs, done):
        """
        决策学习
        :param observations:
        :param actions:
        :param rewards:
        :param old_log_probs:
        :return:
        """

        tensor_observations = torch.tensor(observations, dtype=torch.float32)
        tensor_actions = torch.tensor(actions, dtype=torch.int64)
        tensor_rewards = torch.tensor(rewards, dtype=torch.float32)
        tensor_next_observations = torch.tensor(next_observations, dtype=torch.float32)
        tensor_old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        # 创建数据集
        dataset = TensorDataset(tensor_observations,
                                tensor_actions,
                                tensor_rewards,
                                tensor_next_observations,
                                tensor_old_log_probs)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        # 遍历数据批次
        for _observations, _actions, _rewards, _next_observations, _old_log_probs in dataloader:
            # 检查数据类型
            logger.debug(f"tensor_observations type: {type(_observations)}, dtype: {_observations.dtype}")
            logger.debug(f"_actions type: {type(_actions)}, dtype: {_actions.dtype}")
            logger.debug(f"_rewards type: {type(_rewards)}, dtype: {_rewards.dtype}")
            logger.debug(f"_old_log_probs type: {type(_old_log_probs)}, dtype: {_old_log_probs.dtype}")
            # 检查数据形状，第一个维度必须一致，batch sizes
            logger.debug(f"_observations shape: {_observations.shape}")
            logger.debug(f"_actions shape: {_actions.shape}")
            logger.debug(f"_rewards shape: {_rewards.shape}")
            logger.debug(f"_old_log_probs shape: {_old_log_probs.shape}")
            logger.debug(
                f"_actions dim equal old_log_probs? Answer: {_actions.shape == _old_log_probs.shape}")
            # 检查数据范围
            logger.debug(
                f"env observation range:{self.env.observation_space.low[0]} to {self.env.observation_space.high[1]}")
            logger.debug(
                f"_observations range:{_observations.min().item()} to {_observations.max().item()}")
            logger.debug(f"_rewards range:{_rewards.min().item()} to {_rewards.max().item()}")
            logger.debug(f"_old_log_probs range:{_old_log_probs.min().item()} to {_old_log_probs.max().item()}")
            # 验证是否匹配
            logger.debug(
                f"Batch sizes: {_observations.shape[0]}, {_rewards.shape[0]}, {_old_log_probs.shape[0]}")
            assert _observations.shape[0] == _rewards.shape[0] == _old_log_probs.shape[0], "Not match."
            # 检查数据是否存在空值或者无穷大值
            PPOActorCriticAgent.check_nan_inf(_observations, "_observations")
            PPOActorCriticAgent.check_nan_inf(_rewards, "_rewards")
            PPOActorCriticAgent.check_nan_inf(_old_log_probs, "_old_log_probs")
            # 检查数据所在设备: cpu or gpu
            logger.debug(f"_observations device: {_observations.device}")
            logger.debug(f"_rewards device: {_rewards.device}")
            logger.debug(f"_old_log_probs device: {_old_log_probs.device}")
            # if get gpu, 将数据放入gpu else cpu
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.debug(f"device: {device}")
            _observations = _observations.to(device)
            _rewards = _rewards.to(device)
            _old_log_probs = _old_log_probs.to(device)

            next_values = self.ppo_critic(_next_observations).squeeze()
            values = self.ppo_critic(_observations).squeeze()
            advantages, td_targets = self.compute_advantage(_rewards, values, next_values)
            advantages = torch.tensor(advantages, dtype=torch.float32).detach()

            # TODO 检查critic_losses是否存在问题？
            # Critic 损失
            critic_losses = nn.SmoothL1Loss()(values, td_targets.detach()).mean()
            assert not torch.isnan(critic_losses), "critic_losses contains NaN values!"
            logger.debug(f"critic_losses: {critic_losses}")
            # Actor 损失
            _, _, policy = self.ppo_decide(_observations)
            new_log_probs = torch.log(policy[range(len(_actions)), _actions])

            importance_ratio = torch.exp(new_log_probs - _old_log_probs)
            # logger.info(f"importance_ratio: {importance_ratio}")

            # 重要性采样裁剪
            ratio = importance_ratio
            clipped_ratio = torch.clamp(importance_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            # print(f"Ratio: {ratio}, \nClipped Ratio: {clipped_ratio}")
            actor_losses = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            # actor_losses = -(new_log_probs * advantages.detach()).mean()
            assert not torch.isnan(actor_losses), "actor_losses contains NaN values!"
            # logger.info(f"actor_losses: {actor_losses}")

            # update critic net
            self.ppo_critic_optim.zero_grad()
            critic_losses.backward()
            # 梯度不能裁剪过大，否则捕捉不到变化的部分
            torch.nn.utils.clip_grad_norm_(self.ppo_critic.parameters(), max_norm=2)
            # 检查梯度
            # for name, params in self.ppo_critic.named_parameters():
            #     if params.grad is not None:
            #         logger.debug(f"Critic Gradient of {name} is {params.grad.norm()}")
            #     else:
            #         logger.error(f"Critic Gradient of {name} is None!")
            self.ppo_critic_optim.step()
            # old_params_c = {}
            # for name, params in self.ppo_critic.named_parameters():
            #     old_params_c[name] = params.clone()
            # self.ppo_critic_optim.step()
            # for name, params in self.ppo_critic.named_parameters():
            #     logger.debug(f"Critic Change of {name} is {(old_params_c[name] - params).norm()}")

            # update actor net
            self.ppo_actor_optim.zero_grad()
            actor_losses.backward()
            torch.nn.utils.clip_grad_norm_(self.ppo_actor.parameters(), max_norm=2)
            # for name, params in self.ppo_actor.named_parameters():
            #     if params is not None:
            #         logger.debug(f"Actor Gradient of {name} is {params.grad.norm()}")
            #     else:
            #         logger.error(f"Actor Gradient of {name} is not change!")

            # old_params_a = {}
            # for name, params in self.ppo_actor.named_parameters():
            #     old_params_a[name] = params.clone()
            self.ppo_actor_optim.step()
            # for name, params in self.ppo_actor.named_parameters():
            #     logger.debug(f"Actor Change of {name} is {(old_params_a[name] - params).norm()}")

            if done:
                self.writer.add_scalar("Advantages/advantages", advantages.mean().item(), self.learn_step_counter)
                self.writer.add_scalar("Ratio/importance_ratio", importance_ratio.mean().item(),
                                       self.learn_step_counter)
                self.writer.add_scalar("Loss/actor_losses", actor_losses.item(), self.learn_step_counter)
                self.writer.add_scalar("Loss/critic_losses", critic_losses.item(), self.learn_step_counter)
                self.writer.add_scalar("Learning Rate/actor",
                                       self.ppo_actor_scheduler.get_last_lr()[0], self.learn_step_counter)
                self.writer.add_scalar("Learning Rate/critic",
                                       self.ppo_critic_scheduler.get_last_lr()[0], self.learn_step_counter)
                self.learn_step_counter += 1

            self.ppo_actor_scheduler.step(epoch=self.learn_step_counter)
            self.ppo_critic_scheduler.step(epoch=self.learn_step_counter)
        if done:
            observations.clear()
            actions.clear()
            rewards.clear()
            old_log_probs.clear()

    def ppo_play(self, train):
        # 获取初始状态
        episode_reward = 0.0
        observation, _ = self.env.reset()
        logger.debug(f"Initial Observation: {observation}")
        observations, actions, next_observations, rewards, log_probs = [], [], [], [], []
        done = False
        # 选择train/eval模式
        if train:
            logger.info(f"-----开启训练模式-----")
            self.ppo_actor.train()
            self.ppo_critic.train()
        else:
            logger.info(f"*****开启评估模式*****")
            self.ppo_actor.eval()
            self.ppo_critic.eval()
        # agent循环
        while True:
            # 开启动画
            if self.render:
                self.env.render()

            # 智能体决策
            if not train:
                with torch.no_grad():
                    _actions, probs, _ = self.ppo_decide(observation)
                    action = _actions.item()
                    prob = probs[0]
            else:
                _actions, probs, action_probs = self.ppo_decide(observation)
                action = _actions.item()
                prob = probs[0]
                logger.debug(f"_actions: {_actions}")
                logger.debug(f"Action: {action}")
                logger.debug(f"Prob: {prob}")
                logger.debug(f"Action_Probs: {action_probs}")

            logger.debug(f"Action range is {self.env.action_space.contains(action)}")
            # 环境更新
            next_observation, reward, terminated, truncated, _ = self.step(action)

            logger.debug(f"Next_Observation: {next_observation} range is "
                         f"{self.env.observation_space.contains(next_observation)}")
            logger.debug(f"Reward: {reward} range is {self.env.reward_range[0] < reward < self.env.reward_range[1]}")

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_observation)
            log_probs.append(prob)

            if terminated or truncated:
                done = True
            # 奖励更新
            episode_reward += reward
            # 达到结束状态
            if done:
                self.poo_learn(observations, actions, rewards, next_observations, log_probs, done)
                observations.clear()
                actions.clear()
                rewards.clear()
                log_probs.clear()
                self.temperature = max(0.1, self.temperature * 0.995)  # 每次训练降低温度
                self.clip_ratio = max(0.1, self.clip_ratio * 0.995)  # 每次训练降低温度
                logger.info(f"结束一轮游戏, 奖励为${episode_reward}")
                flag = True if episode_reward >= -100 else False
                self.done_rate.append(flag)
                # self.learn_step_counter += 1
                break
            # 状态更新
            observation = next_observation

        return episode_reward


class SACActorCriticAgent(PPOActorCriticAgent):
    def __init__(self, env, global_a3c_model, global_optimizer, replayer_capacity=100000):
        self.env = env
        # self.env.reset(seed=seed)
        super().__init__(env=env, global_optimizer=global_optimizer, global_a3c_model=global_a3c_model)
        self.gamma = 0.99
        self.entropy_alpha = 0.5
        self.target_alpha = 0.0005
        self.batch_size = 64
        self.batches = 5
        self.temperature = 1.0
        self.learn_step_counter = int(0)
        self.load_model = True
        # tensorboard
        if bool(True):
            log_dir = time.strftime("runs/sac_ac/%Y_%m_%d_%H_%M", current_time)
            if self.is_open_writer:
                self.writer = SummaryWriter(log_dir=log_dir)

        # replayer experience pool
        # self.sac_replayer = SACReplayer(replayer_capacity)
        self.sac_replayer = PERReplayer(capacity=replayer_capacity, beta=0.6)

        # evaluate net
        self.sac_actor = BuildNetwork(
            in_dim=self.State_Num,
            out_dim=self.Action_Num,
            hidden_layers=[64, 128],
            hidden_activation=nn.ReLU,
            out_activation=nn.Softmax,
            optimizer_params={"lr": 0.00001}
        )
        self.sac_actor_optimizer = self.sac_actor.get_optimizer()
        self.sac_actor_scheduler = optim.lr_scheduler.StepLR(self.sac_actor_optimizer,
                                                             step_size=200,
                                                             gamma=0.9)
        # dual q net
        self.q0_net = BuildNetwork(
            in_dim=self.State_Num,
            out_dim=self.Action_Num,
            hidden_layers=[64, 128],
            hidden_activation=nn.ReLU,
            out_activation=None,
            optimizer_params={"lr": 0.00001}
        )
        self.q0_net_optimizer = self.q0_net.get_optimizer()
        self.q0_net_scheduler = optim.lr_scheduler.StepLR(self.q0_net_optimizer,
                                                          step_size=200,
                                                          gamma=0.9)
        self.q1_net = BuildNetwork(
            in_dim=self.State_Num,
            out_dim=self.Action_Num,
            hidden_layers=[64, 128],
            hidden_activation=nn.ReLU,
            out_activation=None,
            optimizer_params={"lr": 0.00001}
        )
        self.q1_net_optimizer = self.q1_net.get_optimizer()
        self.q1_net_scheduler = optim.lr_scheduler.StepLR(self.q1_net_optimizer,
                                                          step_size=200,
                                                          gamma=0.9)
        # value net
        self.sac_critic_main = BuildNetwork(
            in_dim=self.State_Num,
            out_dim=1,
            hidden_layers=[256],
            hidden_activation=nn.ReLU,
            out_activation=None,
            optimizer_params={"lr": 0.0001}
        )
        self.sac_critic_main_optimizer = self.sac_critic_main.get_optimizer()
        self.sac_critic_main_scheduler = optim.lr_scheduler.StepLR(self.sac_critic_main_optimizer,
                                                                   step_size=200,
                                                                   gamma=0.9)
        # baseline net
        self.sac_critic_target = BuildNetwork(
            in_dim=self.State_Num,
            out_dim=1,
            hidden_layers=[256],
            hidden_activation=nn.ReLU,
            out_activation=None,
            optimizer_params={"lr": 0.0001}
        )
        self.sac_critic_target_optimizer = self.sac_critic_target.get_optimizer()
        self.sac_critic_target_scheduler = optim.lr_scheduler.StepLR(self.sac_critic_target_optimizer,
                                                                     step_size=200,
                                                                     gamma=0.9)

        if self.load_model:
            checkpoint = torch.load("tools/policy_dir/Acrobot/sac_actor.pth", weights_only=True)
            self.sac_actor.load_state_dict(checkpoint["model_state_dict"])
            self.sac_actor_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            checkpoint = torch.load("tools/policy_dir/Acrobot/q0_net.pth", weights_only=True)
            self.q0_net.load_state_dict(checkpoint["model_state_dict"])
            self.q0_net_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            checkpoint = torch.load("tools/policy_dir/Acrobot/q1_net.pth", weights_only=True)
            self.q1_net.load_state_dict(checkpoint["model_state_dict"])
            self.q1_net_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            checkpoint = torch.load("tools/policy_dir/Acrobot/sac_critic_main.pth", weights_only=True)
            self.sac_critic_main.load_state_dict(checkpoint["model_state_dict"])
            self.sac_critic_main_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            checkpoint = torch.load("tools/policy_dir/Acrobot/sac_critic_target.pth", weights_only=True)
            self.sac_critic_target.load_state_dict(checkpoint["model_state_dict"])
            self.sac_critic_target_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info(f"成功加载SAC所有模型及参数!")

    def sac_decide(self, observation):
        """
        sac policy
        :param observation:
        :return:
        """
        observation = torch.tensor(observation, dtype=torch.float32)
        probs = self.sac_actor(observation)
        action_distribution = torch.distributions.Categorical(probs)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        entropy = -log_prob
        _action = action.detach().cpu().numpy()
        _entropy = entropy.detach().cpu().numpy()
        logger.debug(f"observation: {observation}")
        return _action, _entropy

    def sac_learn(self, observation, action, reward, next_observation, entropy, done):
        """
        sac core algorithm
        :param observation:
        :param action:
        :param reward:
        :param next_observation:
        :param done:
        :return:
        """

        # self.sac_replayer.replay_store(observation, action, reward, next_observation, entropy, done)
        self.sac_replayer.replay_store(
            observation, action, reward, next_observation, entropy, done, priority=1.0
        )
        # 初始化累积变量
        original_entropies = []
        total_entropies = []
        total_critic_losses_q0 = []
        total_critic_losses_q1 = []
        total_critic_losses = []
        total_actor_losses = []
        for batch in range(self.batches):
            # observations, actions, rewards, next_observations, entropies, dones = (
            #     self.sac_replayer.replay_sample(self.batch_size)
            # )
            # 优先采样
            batch, weights, indices = self.sac_replayer.replay_sample(self.batch_size)

            observations = batch["observation"]
            actions = batch["action"]
            rewards = batch["reward"]
            next_observations = batch["next_observation"]
            entropies = batch["entropy"]
            dones = batch["done"]
            # 将权重转为张量并归一化
            weights_t = torch.tensor(weights, dtype=torch.float32)

            observations_t = torch.tensor(observations, dtype=torch.float32)
            actions_t = torch.tensor(actions, dtype=torch.int64)
            rewards_t = torch.tensor(rewards, dtype=torch.float32).squeeze()
            next_observations_t = torch.tensor(next_observations, dtype=torch.float32)
            entropies_t = torch.tensor(entropies, dtype=torch.float32).squeeze()
            dones_t = torch.tensor(dones, dtype=torch.float32).squeeze()

            # recalculate entropy
            probs = self.sac_actor(observations_t)
            probs_dis = torch.distributions.Categorical(probs)
            actions = probs_dis.sample()
            entropies = -probs_dis.log_prob(actions)
            # calculate ut for each trace
            next_values = self.sac_critic_target(next_observations_t).squeeze().detach()
            q_us = rewards_t + self.gamma * next_values * (1 - dones_t)
            logger.debug(f"动作价值目标: {q_us}")

            # q0 = torch.gather(self.q0_net(observations_t), dim=1, index=actions_t.unsqueeze(-1)).squeeze()
            q0 = self.q0_net(observations_t)[torch.arange(self.batch_size), actions_t]
            # q1 = torch.gather(self.q1_net(observations_t), dim=1, index=actions_t.unsqueeze(-1)).squeeze()
            q1 = self.q1_net(observations_t)[torch.arange(self.batch_size), actions_t]

            less_q = torch.min(q0, q1).squeeze()
            v_us = less_q + self.entropy_alpha * entropies
            logger.debug(f"状态价值目标: {v_us}")

            # critic_losses_q0 = nn.SmoothL1Loss()(q0, q_us)
            # critic_losses_q1 = nn.SmoothL1Loss()(q1, q_us)

            critic_losses_q0 = (weights_t * nn.SmoothL1Loss(reduction='none')(q0, q_us)).mean()
            critic_losses_q1 = (weights_t * nn.SmoothL1Loss(reduction='none')(q1, q_us)).mean()

            value = self.sac_critic_main(observations_t).squeeze()
            # critic_losses = nn.SmoothL1Loss()(value, v_us.detach())
            critic_losses = (weights_t * nn.SmoothL1Loss(reduction='none')(value, v_us.detach())).mean()

            td_error = (q_us - value).detach()
            self.sac_replayer.update_priorities(indices, td_error.tolist())

            # actor_losses = -(less_q.detach() + self.entropy_alpha * entropies).mean()

            actor_losses = -(weights_t * (less_q.detach() + self.entropy_alpha * entropies)).mean()
            logger.debug(f"actor_losses: {actor_losses}")

            self.sac_critic_main_optimizer.zero_grad()
            critic_losses.backward()
            torch.nn.utils.clip_grad_norm_(self.sac_critic_main.parameters(), max_norm=1)
            self.sac_critic_main_optimizer.step()

            self.q0_net_optimizer.zero_grad()
            critic_losses_q0.backward()
            torch.nn.utils.clip_grad_norm_(self.q0_net.parameters(), max_norm=1)
            self.q0_net_optimizer.step()

            self.q1_net_optimizer.zero_grad()
            critic_losses_q1.backward()
            torch.nn.utils.clip_grad_norm_(self.q1_net.parameters(), max_norm=1)
            self.q1_net_optimizer.step()

            self.sac_actor_optimizer.zero_grad()
            actor_losses.backward()
            torch.nn.utils.clip_grad_norm_(self.sac_actor.parameters(), max_norm=1)
            self.sac_actor_optimizer.step()
            if self.learn_step_counter > 0 and self.learn_step_counter % 10 == 0:
                self.soft_update()

            # 假设每个批次的结果是计算得到的
            original_entropies.append(entropies_t.detach().mean().item())
            total_entropies.append(entropies.mean().item())
            total_critic_losses_q0.append(critic_losses_q0.mean().item())
            total_critic_losses_q1.append(critic_losses_q1.mean().item())
            total_critic_losses.append(critic_losses.mean().item())
            total_actor_losses.append(actor_losses.mean().item())

        if done:
            # 在所有批次结束后，计算平均值
            ori_entropy = sum(original_entropies) / len(original_entropies)
            avg_entropy = sum(total_entropies) / len(total_entropies)
            avg_critic_loss_q0 = sum(total_critic_losses_q0) / len(total_critic_losses_q0)
            avg_critic_loss_q1 = sum(total_critic_losses_q1) / len(total_critic_losses_q1)
            avg_critic_loss = sum(total_critic_losses) / len(total_critic_losses)
            avg_actor_loss = sum(total_actor_losses) / len(total_actor_losses)
            # 记录到 TensorBoard
            self.writer.add_scalar("Entropy/total entropies", ori_entropy, self.learn_step_counter)
            self.writer.add_scalar("Entropy/update info", avg_entropy, self.learn_step_counter)
            self.writer.add_scalar("Loss/critic_losses_q0", avg_critic_loss_q0, self.learn_step_counter)
            self.writer.add_scalar("Loss/critic_losses_q1", avg_critic_loss_q1, self.learn_step_counter)
            self.writer.add_scalar("Loss/critic_losses", avg_critic_loss, self.learn_step_counter)
            self.writer.add_scalar("Loss/actor_losses", avg_actor_loss, self.learn_step_counter)
            # self.learn_step_counter += 1
            # self.sac_replayer.learn_step_counter += 1

            total_entropies.clear()
            total_critic_losses_q0.clear()
            total_critic_losses_q1.clear()
            total_critic_losses.clear()
            total_actor_losses.clear()

        self.sac_actor_scheduler.step()
        self.q0_net_scheduler.step()
        self.q1_net_scheduler.step()
        self.sac_critic_main_scheduler.step()
        self.sac_critic_target_scheduler.step()

    def soft_update(self):
        for target_params, main_params in zip(self.sac_critic_target.parameters(), self.sac_critic_main.parameters()):
            target_params.data.copy_(self.target_alpha * main_params + (1 - self.target_alpha) * target_params)

    def loader_pool(self, observation, action, reward, next_observation, entropy, done):
        from tqdm import tqdm
        remaining = max(0, 1000 - self.sac_replayer.count)  # 计算还需要填充的样本数量
        if remaining > 0:
            if self.sac_replayer.count < 1000:
                with tqdm(total=1000, initial=self.sac_replayer.count, dynamic_ncols=True,
                          desc="Experience Pool Loader") as pbar:
                    for _ in range(1000):
                        self.sac_replayer.replay_store(
                            observation, action, reward, next_observation, entropy, done, pbar=pbar
                        )
                        # self.sac_replayer.replay_store(
                        #     observation, action, reward, next_observation, entropy, done, priority=1.0, pbar=pbar
                        # )
                        # time.sleep(0.01)

    def sac_play(self, train):
        # 获取初始状态
        episode_reward = 0.0
        observation, _ = self.env.reset()
        logger.debug(f"Initial Observation: {observation}")
        done = False
        # 选择train/eval模式
        if train:
            logger.info(f"-----开启训练模式-----")
            self.ppo_actor.train()
            self.ppo_critic.train()
        else:
            logger.info(f"*****开启评估模式*****")
            self.ppo_actor.eval()
            self.ppo_critic.eval()
        # agent循环
        while True:
            # 开启动画
            if self.render:
                self.env.render()

            # 智能体决策
            if not train:
                with torch.no_grad():
                    action, entropy = self.sac_decide(observation)
            else:
                action, entropy = self.sac_decide(observation)
                logger.debug(f"_actions: {action}")

            logger.debug(f"Action range is {self.env.action_space.contains(action)}")
            # 环境更新
            next_observation, reward, terminated, truncated, _ = self.step(action)
            # 填充经验池
            self.loader_pool(observation, action, reward, next_observation, entropy, done)

            logger.debug(f"Next_Observation: {next_observation} range is "
                         f"{self.env.observation_space.contains(next_observation)}")
            logger.debug(f"Reward: {reward} range is {self.env.reward_range[0] < reward < self.env.reward_range[1]}")
            if terminated or truncated:
                done = True

            if train:
                # if done:
                #     next_observation = self.env.reset()
                self.sac_learn(observation, action, reward, next_observation, entropy, done)

            # 奖励更新
            episode_reward += reward
            # 达到结束状态
            if done:
                if self.learn_step_counter > 0 and self.learn_step_counter % 50 == 0:
                    self.temperature = max(0.1, self.temperature * 0.995)  # 每次训练降低温度
                    self.writer.add_scalar("Params/temperature", self.temperature, self.learn_step_counter)
                    self.entropy_alpha = max(0.1, self.entropy_alpha * 0.995)
                    self.writer.add_scalar("Params/entropy_alpha", self.entropy_alpha, self.learn_step_counter)
                self.learn_step_counter += 1
                logger.info(f"结束一轮游戏, 奖励为${episode_reward}")
                flag = True if episode_reward >= -100 else False
                self.done_rate.append(flag)
                # self.learn_step_counter += 1
                break
            # 状态更新
            observation = next_observation

        return episode_reward


class Acrobot(SACActorCriticAgent):
    def __init__(self, name=None, render_mode=render_model[0], render=True):
        # 在Acrobot中创建全局模型和优化器
        if render:
            env = gym.make(name, render_mode=render_mode)
        else:
            env = gym.make(name)
        global_a3c_model = BuildA3CNetwork(
            in_state_dim=6,  # 假设状态空间维度为6
            out_value_dim=1,
            out_action_dim=3,  # 假设动作空间维度为3
            hidden_layer=[256]
        )
        global_a3c_model.share_memory()

        global_optimizer = global_a3c_model.get_optimizer(
            model=global_a3c_model,
            learning_rate=0.00001
        )

        # 如果加载模型
        if bool(False):
            checkpoint = torch.load("tools/policy_dir/Acrobot/global_a3c_model.pth", weights_only=True)
            global_a3c_model.load_state_dict(checkpoint["model_state_dict"])
            global_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"成功加载--->global_a3c_model")

        # 调用父类的初始化
        super().__init__(env=env, global_a3c_model=global_a3c_model, global_optimizer=global_optimizer)

        logger.info(f"搜索顺序:{Acrobot.mro()}")
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

            if show_policy == "动作actor_critic算法":
                logger.info(f"动作actor_critic算法")
                episode_reward = self.play_actor_critic(train=False)  # 第round轮次的累积reward
                method_name = self.play_actor_critic.__name__
            elif show_policy == "优势actor_critic算法":
                logger.info(f"优势actor_critic算法")
                episode_reward = self.ad_play_actor_critic(train=False)  # 第round轮次的累积reward
                method_name = self.ad_play_actor_critic.__name__
            elif show_policy == "资格迹actor_critic算法":
                logger.info(f"资格迹actor_critic算法")
                episode_reward = self.lambda_play_actor_critic(train=False)  # 第round轮次的累积reward
                method_name = self.lambda_play_actor_critic.__name__
            elif show_policy == "同步actor_critic算法":
                logger.info(f"同步actor_critic算法")
                episode_reward = self.a2c_play_actor_critic(train=False)  # 第round轮次的累积reward
                method_name = self.a2c_play_actor_critic.__name__
            elif show_policy == "异步actor_critic算法":
                logger.info(f"异步actor_critic算法")
                episode_reward = self.run_workers(train=False)  # 第round轮次的累积reward
                method_name = self.a3c_play.__name__
            elif show_policy == "邻近PPO算法":
                logger.info(f"邻近PPO算法")
                episode_reward = self.ppo_play(train=False)  # 第round轮次的累积reward
                method_name = self.ppo_play.__name__
            elif show_policy == "soft_actor_critic算法":
                logger.info(f"soft_actor_critic算法")
                episode_reward = self.sac_play(train=False)  # 第round轮次的累积reward
                method_name = self.sac_play.__name__
            else:
                logger.error(f" show_policy = {show_policy} ")
                raise ValueError("输入的策略名称错误，请仔细检查!")

            if self.global_is_train and self.save_policy and (
                    game_round % 100 == 0 or game_round == self.game_rounds - 1):
                if show_policy == "动作actor_critic算法":
                    save_data = {"actor": self.actor,
                                 "critic": self.critic,
                                 "actor_optimizer": self.actor_optimizer,
                                 "critic_optimizer": self.critic_optimizer}
                    Policy_loader.save_policy(method_name, self.class_name, save_data, step=game_round)
                if show_policy == "优势actor_critic算法":
                    save_data = {"ad_actor": self.ad_actor,
                                 "ad_critic": self.ad_critic,
                                 "ad_actor_optimizer": self.ad_actor_optimizer,
                                 "ad_critic_optimizer": self.ad_critic_optimizer}
                    Policy_loader.save_policy(method_name, self.class_name, save_data, step=game_round)
                if show_policy == "资格迹actor_critic算法":
                    save_data = {"lambda_actor": self.lambda_actor,
                                 "lambda_critic": self.lambda_critic,
                                 "lambda_actor_optimizer": self.lambda_actor_optimizer,
                                 "actor_e_traces": self.actor_e_traces,
                                 "critic_e_traces": self.critic_e_traces}
                    Policy_loader.save_policy(method_name, self.class_name, save_data, step=game_round)
                if show_policy == "同步actor_critic算法":
                    save_data = {"a2c_actor": self.a2c_actor,
                                 "a2c_critic": self.a2c_critic,
                                 "a2c_actor_optimizer": self.a2c_actor_optimizer,
                                 "a2c_critic_optimizer": self.a2c_critic_optimizer}
                    Policy_loader.save_policy(method_name, self.class_name, save_data, step=game_round)
                if show_policy == "异步actor_critic算法":
                    save_data = {"local_a3c_model": self.local_a3c_model,
                                 "global_a3c_model": self.global_a3c_model,
                                 "global_optimizer": self.global_optimizer
                                 }
                    Policy_loader.save_policy(method_name, self.class_name, save_data, step=game_round)
                if show_policy == "邻近PPO算法":
                    save_data = {"ppo_actor": self.ppo_actor,
                                 "ppo_critic": self.ppo_critic,
                                 "ppo_actor_optim": self.ppo_actor_optim,
                                 "ppo_critic_optim": self.ppo_critic_optim
                                 }
                    Policy_loader.save_policy(method_name, self.class_name, save_data, step=game_round)
                if show_policy == "soft_actor_critic算法":
                    save_data = {"sac_actor": self.sac_actor,
                                 "sac_actor_optimizer": self.sac_actor_optimizer,
                                 "q0_net": self.q0_net,
                                 "q0_net_optimizer": self.q0_net_optimizer,
                                 "q1_net": self.q1_net,
                                 "q1_net_optimizer": self.q1_net_optimizer,
                                 "sac_critic_main": self.sac_critic_main,
                                 "sac_critic_main_optimizer": self.sac_critic_main_optimizer,
                                 "sac_critic_target": self.sac_critic_target,
                                 "sac_critic_target_optimizer": self.sac_critic_target_optimizer,
                                 }
                    Policy_loader.save_policy(method_name, self.class_name, save_data, step=game_round)

            if episode_reward is not None:
                episode_rewards.append(episode_reward)
                if self.is_open_writer:
                    if self.learn_step_counter % 10 == 0:  # 每 10 轮记录一次奖励
                        self.writer.add_scalar("Episode Reward", episode_reward, global_step=self.learn_step_counter)
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
