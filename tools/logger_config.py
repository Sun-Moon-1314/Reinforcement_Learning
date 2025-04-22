# -*- coding: utf-8 -*-
"""
@File    : logger_config.py      # 文件名，logging表示当前文件名
@Time    : 2024/12/27         # 创建时间，2024/12/27表示当前时间
@Author  : <your_name>     # 作者
@Email   : <your_email>    # 作者电子邮件
@Desc    : <brief_description> # 文件的简要描述
"""
import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging():
    # 获取当前文件的目录路径
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # 如果 logs 目录不存在，创建它

    # 创建根记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 修改为 INFO 级别
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # 创建文件处理器（带文件大小轮转功能），日志文件将存储到 logs 目录
    log_file_path = os.path.join(log_dir, 'record.log')
    file_handler = RotatingFileHandler(log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 添加处理器到根记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 如果需要为某些模块单独设置
    module_logger = logging.getLogger('specific_module')
    module_logger.setLevel(logging.INFO)
    module_log_path = os.path.join(log_dir, 'specific_module.log')
    module_handler = logging.FileHandler(module_log_path)
    module_handler.setFormatter(file_formatter)
    module_logger.addHandler(module_handler)
