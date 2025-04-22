# -*- coding: utf-8 -*-
"""
@File    : __init__.py      # 文件名，.init表示当前文件名
@Time    : 2024/12/27         # 创建时间，2024/12/27表示当前时间
@Author  : <your_name>     # 作者
@Email   : <your_email>    # 作者电子邮件
@Desc    : <brief_description> # 文件的简要描述
"""
from envs.taxi import logger
from .run_taxi import run_taxi_dispatch
from .run_ice_skating import run_ice_skating_game
from .run_21_points import run_21_points_game
from .run_select_func import run_select_func
from .run_mountain_car import run_mountain_car
from .run_cartpole import run_cartpole
from .run_acrobot import run_acrobot

logger.info(f"Initializing run_function")
__all__ = [
    'run_taxi_dispatch',
    'run_ice_skating_game',
    'run_21_points_game',
    'run_select_func',
    'run_mountain_car',
    'run_cartpole',
    'run_acrobot'
]
