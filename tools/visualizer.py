# -*- coding: utf-8 -*-
"""
@File    : evaluator.py
@Time    : 2024/12/19 16:52
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    :
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger(__name__)  # 使用当前模块名

color = {
    0: "hot",        # "hot" 配色方案，表示一种从黑色到红色、橙色、黄色的渐变色调，用于热图绘制
    1: "inferno",    # "inferno" 配色方案，表示一种从黑色到红色、黄色，再到白色的渐变色调，用于热图绘制
    2: "plasma",     # "plasma" 配色方案，表示一种从深蓝色到紫色、橙色、黄色的渐变色调，用于热图绘制
    3: "viridis"     # "viridis" 配色方案，表示一种从深蓝色到绿色，再到黄色的渐变色调，用于热图绘制
}

class Visualizer:

    avg_rewards_across_iterations = []  # 用于保存每个迭代的平均奖励
    iterations = []  # 用于保存每个迭代的编号
    save_picture = True # 保存图像开关
    print_by_step = 10 # 每隔N步打印一次图像

    picture_dir = os.path.join(os.path.dirname(__file__), 'picture_dir')
    if not os.path.exists(picture_dir):
        os.makedirs(picture_dir)  # 如果 logs 目录不存在，创建它

    @staticmethod
    def plot(input_data):
        """
        可视化
        :param input_data:
        :return:
        """
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        titles = ['without ace', 'with ace']
        have_aces = [0, 1]
        extent = [12, 22, 1, 11]
        for title, have_ace, axis in zip(titles, have_aces, axes):
            dat = input_data[extent[0]:extent[1], extent[2]:extent[3], have_ace].T
            img = axis.imshow(dat, extent=extent, origin='lower', aspect='auto', cmap='viridis')
            axis.set_xlabel('player sum')
            axis.set_ylabel('dealer showing')
            axis.set_title(title)
            # 添加颜色条并设置标签
            cbar = fig.colorbar(img, ax=axis, orientation='vertical')
            cbar.set_label('Value/Policy Intensity')
        plt.tight_layout()

        plt.show(block=False)  # Non-blocking display
        plt.pause(2)  # Pause for the specified duration
        plt.close()  # Close the plot after the duration

    @staticmethod
    def plot_policy_and_value(policy, value):
        """
        打印策略和价值
        :param policy: 策略
        :param value: 价值
        :return:
        """
        logger.info(f"打印图像")
        p = policy.argmax(-1)
        Visualizer.plot(p)
        v = value.max(axis = -1)
        Visualizer.plot(v)
        logger.info(f"图像关闭")

    @staticmethod
    def plot_episode_rewards(episode_rewards, iteration):
        """
        可视化 episode_rewards
        :param episode_rewards: 每一回合的奖励列表
        :param iteration: 当前迭代编号
        """

        if not isinstance(episode_rewards, (list, np.ndarray)):
            logger.error("episode_rewards must be a list or numpy array.")
            return

        if len(episode_rewards) == 0:
            logger.error("episode_rewards is empty.")
            return

        # 计算平均奖励
        avg_reward = np.mean(episode_rewards)

        # 将当前迭代的平均奖励保存到列表中
        Visualizer.avg_rewards_across_iterations.append(avg_reward)
        # Visualizer.iterations.append(iteration + 1)  # 迭代编号从 1 开始

        # 绘制奖励变化曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(episode_rewards)), episode_rewards, label='Episode Rewards',
                 color='blue', marker='o', markersize=3, linestyle='-')
        plt.axhline(y=avg_reward, color='red', linestyle='--', label=f'Avg Reward = {avg_reward:.2f}')
        plt.xlabel('Game Round', fontsize=12)
        plt.ylabel('Total Reward', fontsize=12)
        plt.title(f'Rewards per Episode (Iteration {iteration + 1})', fontsize=14)
        plt.legend()
        plt.grid(True)

        # 显示统计信息
        plt.annotate(f'Iteration: {iteration + 1}\nAvg Reward: {avg_reward:.2f}',
                     xy=(0.7, 0.1), xycoords='axes fraction', fontsize=10,
                     bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))

        plt.tight_layout()
        plt.pause(1)  # 显示 1 秒钟
        plt.close()  # 关闭图像

    @staticmethod
    def plot_cumulative_avg_rewards(episode_rewards, iteration, total_iterations, class_name, method_name):
        """
        绘制所有迭代的平均奖励曲线，每隔10轮更新一次图像，并在最后一步打印最终图像。
        """

        # 初始化保存平均奖励和迭代次数的列表
        if not hasattr(Visualizer, 'avg_rewards_across_iterations'):
            Visualizer.avg_rewards_across_iterations = []
            Visualizer.iterations = []

        # 计算当前迭代的平均奖励
        avg_reward = np.mean(episode_rewards)

        # 将当前迭代的平均奖励保存到列表中
        Visualizer.avg_rewards_across_iterations.append(avg_reward)
        Visualizer.iterations.append(iteration)

        # 每隔10轮或在最后一轮时更新图像
        if iteration % Visualizer.print_by_step == 0 or iteration == (total_iterations):
            logger.info(
                f"Plotting cumulative average rewards for {len(Visualizer.avg_rewards_across_iterations)} iterations."
            )

            # 绘制所有迭代的平均奖励曲线
            plt.figure(figsize=(10, 6))
            plt.plot(
                Visualizer.iterations,
                Visualizer.avg_rewards_across_iterations,
                label='Average Reward per Iteration',
                color='green',
                marker='o',
                markersize=5,
                linestyle='-',
                linewidth=2
            )

            plt.xlabel('Round', fontsize=12)
            plt.ylabel('Average Reward', fontsize=12)
            plt.title('Average Reward Across Iterations', fontsize=14)
            plt.legend()
            plt.grid(True)
            # 在图像右上角显示当前平均奖励（保留两位小数）
            plt.text(
                0.95, 0.05,
                f"Avg Reward: {avg_reward:.2f}",
                fontsize=12,
                color='blue',
                ha='right', va='top',
                transform=plt.gca().transAxes
            )
            plt.tight_layout()
            # plt.pause(0.001)  # 动态显示图像

            if Visualizer.save_picture:
                # 保存图像
                picture_dir = os.path.join(Visualizer.picture_dir, class_name)
                if not os.path.exists(picture_dir):
                    os.makedirs(picture_dir)
                Visualizer.save_dir = os.path.join(picture_dir, method_name)
                plt.savefig(Visualizer.save_dir)  # 保存为PNG文件
                logger.info("Cumulative average rewards plot saved.")

            # 如果不是最后一轮，则关闭图像
            # if iteration != total_iterations:
            plt.close()

    @staticmethod
    def plot_maintain_curve(positions, velocities):
        fig, ax = plt.subplots()
        if len(positions) > 0:
            ax.plot(positions, label='positions')
        if len(velocities) > 0:
            ax.plot(velocities, label='velocities')
        ax.legend()
        plt.show()