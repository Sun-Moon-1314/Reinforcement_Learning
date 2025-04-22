# -*- coding: utf-8 -*-
"""
@File    : build_network.py
@Time    : 2025/3/5 17:10
@Author  : ZhangJian
@Email   : your_email@example.com
@Desc    : 
"""
__all__ = [
    "BuildNetwork",
    "BuildA3CNetwork"
]

import torch.nn as nn
from torch import Tensor
from typing import List, Callable, Union, Type
import torch.optim as optim
from envs.cartpole import logger


class BuildNetwork(nn.Module):
    """
    Build a neural network
    :param in_dim: 输入维度，通常是状态的维度
    :param out_dim: 输出维度，价值函数值或者是动作概率，取决于输出激活函数类型
    :param hidden_layers: 隐藏层数量，单层int或者多层[int,...]
    :param hidden_activation: 隐藏层的激活函数, 默认为 nn.ReLU()
    :param out_activation: 输出层的激活函数，默认为 nn.Softmax(dim=-1)
    :param optimizer_cls: 优化器类，默认为 optim.Adam
    :param optimizer_params: 优化器参数，默认为 {"lr": 0.01}
    :param verbose: 是否显示网络结构，默认为 False
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            hidden_layers: Union[int, List[int], None] = None,
            hidden_activation: Union[Type[nn.Module], Callable[[], nn.Module]] = nn.ReLU,
            out_activation: Union[Type[nn.Module], Callable[[], nn.Module]] = None,
            optimizer_cls: Type[optim.Optimizer] = optim.Adam,
            optimizer_params: dict = None,
            verbose: bool = False,
    ):
        super().__init__()

        if isinstance(in_dim, tuple):
            self.in_dim = int(in_dim[0])
        else:
            self.in_dim = in_dim
        if isinstance(out_dim, tuple):
            self.out_dim = int(out_dim[0])
        else:
            self.out_dim = out_dim

        logger.debug(f"in_dim:{self.in_dim}, type:{type(self.in_dim)}")

        if hidden_layers is None:
            self.hidden_layers = []
        elif isinstance(hidden_layers, tuple):
            self.hidden_layers = int(hidden_layers[0])
        elif isinstance(hidden_layers, int):
            self.hidden_layers = [hidden_layers]
        elif isinstance(hidden_layers, list):
            self.hidden_layers = hidden_layers
        else:
            raise ValueError("hidden_sizes must be an int, a list of ints, or None.")

        logger.debug(f"hidden_layers:{self.hidden_layers}, type:{type(self.hidden_layers)}")

        if not callable(hidden_activation):
            raise ValueError("hidden_activation must be callable.")
        if out_activation is not None and not callable(out_activation):
            raise ValueError("out_activation must be callable.")

        self.hidden_activation = hidden_activation

        self.out_activation = out_activation if out_activation is not None else nn.Identity
        if self.out_activation == nn.Softmax:
            self.out_activation = lambda: nn.Softmax(dim=-1)

        self.model = self._build_dynamic_network()

        self.optimizer_cls = optimizer_cls
        self.optimizer_params = optimizer_params if optimizer_params else {"lr": 0.01}
        self.optimizer = self.get_optimizer()
        self._init_weights()
        if verbose:
            print("-------网络信息-------")
            print(f"Input Dimension: {self.in_dim}")
            for i, hidden_layer in enumerate(self.hidden_layers):
                print(f"Hidden Layer {i + 1}: {hidden_layer} units, "
                      f"Activation: {self.hidden_activation.__name__}")
            print(f"Output Dimension: {self.out_dim}")
            print(
                f"Output Activation: {self.out_activation.__name__ if self.out_activation != nn.Identity else 'Identity'}")

    def _init_weights(self):
        """
        权重初始化，缓解梯度消失或者爆炸，防止在最开始时输入输出方差过大难以训练
        :return:
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _build_dynamic_network(self):
        """
        构建动态可扩展网络结构
        :return: Sequential
        """
        layers = []
        in_features = self.in_dim

        if len(self.hidden_layers) > 0:
            for hidden_layer in self.hidden_layers:
                layers.append(nn.Linear(in_features=in_features, out_features=hidden_layer))
                layers.append(self.hidden_activation())
                in_features = hidden_layer
            layers.append(nn.Linear(in_features=self.hidden_layers[-1], out_features=self.out_dim))
        else:
            layers.append(nn.Linear(in_features=in_features, out_features=self.out_dim))

        if self.out_activation != nn.Identity:
            layers.append(self.out_activation())

        model = nn.Sequential(*layers)

        return model

    def forward(self, input_data: Tensor):
        """
        前馈神经网络
        :return: model
        """
        return self.model(input_data)

    # TODO: 优化器
    def get_optimizer(self):
        """
        创建并返回优化器实例
        :return: 优化器
        """
        return self.optimizer_cls(self.model.parameters(), **self.optimizer_params)


class BuildA3CNetwork(nn.Module):
    """
    Build a shared network for A3C specially.

    :param in_state_dim: 输入状态空间维度
    :param out_action_dim: 输出动作空间维度
    :param out_value_dim: 输出价值维度，=1为状态价值，>1为动作价值
    :param hidden_layer: 隐藏层神经元数
    """

    def __init__(self,
                 in_state_dim: int = None,
                 out_action_dim: int = None,
                 out_value_dim: int = None,
                 hidden_layer: Union[int, list[int]] = None,
                 ):
        super(BuildA3CNetwork, self).__init__()

        if hidden_layer is None or len(hidden_layer) == 0:
            raise ValueError("hidden_layer should not be None or [].")
        if isinstance(hidden_layer, int):
            hidden_layer = [hidden_layer]
        self.hidden_layer = hidden_layer
        if isinstance(in_state_dim, tuple):
            self.in_state_dim = int(in_state_dim[0])
        elif isinstance(in_state_dim, int):
            self.in_state_dim = in_state_dim
        else:
            raise ValueError("in_state_dim must be input.")
        if isinstance(out_action_dim, tuple):
            self.out_action_dim = int(out_action_dim[0])
        elif isinstance(out_action_dim, int):
            self.out_action_dim = out_action_dim
        else:
            raise ValueError("out_action_dim must be input.")

        self.out_value_dim = out_value_dim

        # 多层隐藏层
        if len(self.hidden_layer) >= 1:
            input_dim = self.in_state_dim
            layers = []
            for hidden in self.hidden_layer:
                layers.append(nn.Linear(input_dim, hidden))
                layers.append(nn.ReLU())
                input_dim = hidden
            self.shared_layer = nn.Sequential(*layers)
        # 策略输出层
        self.actor_layer = nn.Sequential(
            nn.Linear(self.hidden_layer[-1], self.out_action_dim),
            nn.Softmax(dim=-1)
        )
        # 状态价值输出层
        self.critic_layer = nn.Linear(self.hidden_layer[-1], self.out_value_dim)

        # 初始化权重
        self._init_weights()

    def get_modules(self):
        """
        获取完整模型结构
        :return:
        """
        for i, m in enumerate(self.modules()):
            logger.info(f"模型结构: {i} -> {m}")
        return self.modules()

    def _init_weights(self):
        """
        权重初始化，缓解梯度消失或者爆炸，防止在最开始时输入输出方差过大难以训练
        :return:
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, observation: Tensor):
        """
        前向传播网络
        :param observation: (Tensor) 输入状态，形状为 [batch_size, in_state_dim]
        :return:
            - policy: (Tensor) 策略概率分布，形状为 [batch_size, out_action_dim]
            - value: (Tensor) 状态价值，形状为 [batch_size, out_value_dim]
        """
        shared_feature = self.shared_layer(observation)
        policy = self.actor_layer(shared_feature)
        value = self.critic_layer(shared_feature)
        return policy, value

    @staticmethod
    def get_optimizer(
            model: nn.Module,
            learning_rate: float = 0.01
    ):
        """
        获取优化器
        :param model: 神经网络模型
        :param learning_rate: 学习率
        :return: Adam优化器
        """
        return optim.Adam(params=model.parameters(), lr=learning_rate)
