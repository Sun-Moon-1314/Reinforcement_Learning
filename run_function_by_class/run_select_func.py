# -*- coding: utf-8 -*-
"""
@File    : run_select_func.py      # 文件名，run_select_func表示当前文件名
@Time    : 2024/12/27         # 创建时间，2024/12/27表示当前时间
@Author  : <your_name>     # 作者
@Email   : <your_email>    # 作者电子邮件
@Desc    : <brief_description> # 文件的简要描述
"""
def run_select_func(get_function, number):
    """
    运行选择函数
    :param get_function:
    :param number:
    :return:
    """
    select_function = get_function.get(number)
    if select_function:
        select_function()
    else:
        print(f"Invalid key.")