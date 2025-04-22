# -*- coding: utf-8 -*-
"""
@File    : data_processor.py
@Time    : 2025/3/13 20:03
@Author  : ZhangJian
@Email   : your_email@example.com
@Desc    : Build my own function
"""
__all__ = [
    "data_normalized",
    "action_probs_normalized"
]

from typing import Union, Optional

import numpy
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def data_normalized(
        data: Union[torch.Tensor, numpy.ndarray] = None,
        feature_range: tuple[int, int] = (0, 1)
):
    """
    对数据进行归一化处理
    :param data: 输入数据，可以是 torch.Tensor 或 numpy.ndarray
    :param feature_range: 归一化范围, 默认(0, 1)
    :return: 归一化后的数据, 可以是 torch.Tensor 或 numpy.ndarray

    Examples
    --------
    >>> import numpy as np
    >>> data_numpy = np.array([1, 2, 3, 4, 5])
    >>> normalized_numpy = data_normalized(data_numpy)
    >>> print("Normalized NumPy Data:", normalized_numpy)
    >>> import torch
    >>> data_tensor = torch.Tensor([1, 2, 3, 4, 5])
    >>> normalized_torch = data_normalized(data_tensor)
    >>> print("Normalized PyTorch Data:", normalized_torch)
    --------
    """

    if data is None:
        raise ValueError("Input data should not be None.")
    # 初始化归一化结果
    normalized_data = None
    if isinstance(data, torch.Tensor):
        # Min-Max 归一化到 [0, 1]
        normalized_data = F.normalize(data, p=2, dim=0)
    elif isinstance(data, numpy.ndarray):
        # Min-Max 归一化到 [0, 1]，需要确保数据是二维的
        data = data.reshape(-1, 1) if data.ndim == 1 else data
        scaler = MinMaxScaler(feature_range=feature_range)
        normalized_data = scaler.fit_transform(data)
        normalized_data = normalized_data.flatten()
    else:
        raise TypeError("Input data type must be torch.Tensor or numpy.ndarray!")

    return normalized_data


def action_probs_normalized(
        probs: torch.Tensor = None,
        temperature: float = 1.0
):
    """
    对神经网络的输出动作概率进行处理，归一化
    :param probs: 动作概率，torch.Tensor
    :param temperature: 温度控制, float: ==1，分布不变；>1且越大，概率分布更均匀；<1，分布更确定
    :return: 归一化的动作概率

    Examples
    --------
    >>> probs = torch.Tensor([[0.4, 0.5, 0.1], [0.2, 0.3, 0.5]])
    >>> normalized_probs = action_probs_normalized(probs, 0.5)
    >>> print(f"normalized_probs: {normalized_probs}")
    >>> probs_np = numpy.array([0.4, 0.5, 0.1])
    >>> normalized_probs_np = action_probs_normalized(probs_np, 0.9)
    >>> print(f"normalized_probs_np: {normalized_probs_np}")
    --------
    """

    if probs is None:
        raise ValueError("Probs should not be None!")
    normalized_probs = None
    if isinstance(probs, torch.Tensor):
        logits = torch.log(probs + 1e-8) / temperature
        normalized_probs = torch.softmax(logits, dim=-1)
        if probs.dim() == 1:  # 如果是 1D 张量
            assert torch.isclose(normalized_probs.sum(), torch.tensor(1.0), atol=1e-6), "Probs 必须和为 1!"
        else:  # 如果是 2D 张量
            assert torch.allclose(normalized_probs.sum(dim=-1), torch.ones(normalized_probs.size(0)),
                                  atol=1e-6), "Probs 必须和为 1!"
    elif isinstance(probs, numpy.ndarray):
        logits = np.log(probs + 1e-8) / temperature
        normalized_probs = np.exp(logits) / np.sum(np.exp(logits))
        assert np.isclose(normalized_probs.sum(), 1.0), "Probs必须和为1!"
    else:
        raise TypeError("Probs 必须是 torch.Tensor 或 numpy.ndarray!")

    return normalized_probs


