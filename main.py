# -*- coding: utf-8 -*-
"""
@File    : main.py
@Time    : 2024/12/20 17:50
@Author  : Zhang Jian
@Email   : your_email@example.com
@Desc    : study
"""

import run_function_by_class as select
from tools.printer_tool import PrintTool as printout
# 根记录器的日志
import logging
# 在主程序或入口文件中首先调用 setup_logging()，确保配置生效
from tools import logger_config

logger_config.setup_logging()
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    show_all_env = False
    if show_all_env:
        printout.print_all_env()
        printout.get_related_env(name="Taxi-v3")

    choose_env = {
        "冰面滑行": "FrozenLake-v1",
        "21点游戏": "Blackjack-v1",
        "出租车调度": "Taxi-v3",
        "小车上山": "MountainCar-v0",
        "平衡杆": "CartPole-v0",
        "双节倒立摆": "Acrobot-v1",
        # ...
    }

    choose = choose_env.get("双节倒立摆")
    if choose == "Blackjack-v1":
        select.run_21_points_game()
    if choose == "FrozenLake-v1":
        select.run_ice_skating_game()
    if choose == "Taxi-v3":
        select.run_taxi_dispatch()
    if choose == "MountainCar-v0":
        select.run_mountain_car()
    if choose == "CartPole-v0":
        select.run_cartpole()
    if choose == "Acrobot-v1":
        select.run_acrobot()
