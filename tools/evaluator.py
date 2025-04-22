# -*- coding: utf-8 -*-
"""
@File    : evaluator.py
@Time    : 2024/12/19 16:52
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    : 
"""

import numpy as np
class Evaluator:
    @staticmethod
    def evaluate_policy(policy1_info, policy2_info):
        """
        比较策略优劣
        :param policy1_info: (policy_name1, q_value1)
        :param policy2_info: (policy_name2, q_value2)
        :return:
        """
        policy1_sum = np.sum(policy1_info[1])
        policy2_sum = np.sum(policy2_info[1])
        if policy1_sum > policy2_sum:
            print("\n----{}->策略更好，其评估结果为{}>{}----".format(policy1_info[0], policy1_sum, policy2_sum))
        else:
            print("\n----{}->策略更好，其评估结果为{}>{}----".format(policy2_info[0], policy2_sum, policy1_sum))