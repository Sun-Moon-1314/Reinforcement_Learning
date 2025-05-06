# -*- coding: utf-8 -*-
"""
@File    : complicated_pathplan.py      # 文件名，complicated_pathplan表示当前文件名
@Time    : 2025/1/3         # 创建时间，2025/1/3表示当前时间
@Author  : <your_name>     # 作者
@Email   : <your_email>    # 作者电子邮件
@Desc    : <brief_description> # 文件的简要描述
"""
from random import random

import pygame
import numpy as np
import time

# 默认障碍物列表
_default = [
    (1, 2),(1, 4),
    (2, 1),
    (3, 3), (3, 4),
    (4, 1)
]
from load_module_pic import ImageToGrid
# 定义智能体类
class PathPlanningEnv(ImageToGrid):
    def __init__(self, start=(60, 60), goal=(140, 120), grid_size=(150, 150), obstacles=None, render = True):
        super().__init__(start=start, goal=goal, obstacles=obstacles, grid_size=grid_size)
        self.screen = None
        self.state = self.start  # 当前状态（位置）
        self.open_render = render # 可视化控制开关
        self.car_entire_coord = []
        if self.open_render:
            self.env_show_init()

        self.agent_angle = 0  # 初始角度（车的朝向为右）
        self.action_translate = {
            0: "上",
            1: "下",
            2: "左",
            3: "右",
            4: "左上",
            5: "右上",
            6: "左下",
            7: "右下",
        }

    def reset(self):
        """重置环境"""
        self.state = self.start  # 重置为起点
        return self.state

    def get_action_angle(self, action):
        """
        获取动作对应的角度
        :param action: 动作编号
        :return: 动作的角度
        """
        action_angles = {
            0: 90,  # 上
            1: 270,  # 下
            2: 180,  # 左
            3: 0,  # 右
            4: 135,  # 左上
            5: 45,  # 右上
            6: 225,  # 左下
            7: 315  # 右下
        }
        return action_angles[action]

    def get_car_four_coord(self, car_center):
        """
        获取小车四个角的坐标
        :param car_center: (x, y) 小车中心在栅格图中的坐标
        :return: 四个角的实际栅格坐标列表 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        """
        cx, cy = car_center  # 小车中心坐标（整张图中的栅格位置）
        car_width = self.car_width / self.cell_size  # 小车宽度
        car_height = self.car_height / self.cell_size  # 小车高度
        angle_rad = np.deg2rad(self.agent_angle)  # 将角度转换为弧度

        # 定义矩形的四个顶点（相对于中心点）
        half_width = car_width / 2
        half_height = car_height / 2
        corners = np.array([
            [-half_width, -half_height],  # 左上角
            [half_width, -half_height],  # 右上角
            [half_width, half_height],  # 右下角
            [-half_width, half_height]  # 左下角
        ])

        # 创建旋转矩阵
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

        # 对矩形顶点进行旋转
        rotated_corners = np.dot(corners, rotation_matrix.T)

        # 平移到小车中心
        rotated_corners += [cx, cy]

        # 转换为整数栅格坐标
        grid_coordinates = np.round(rotated_corners).astype(int)

        return grid_coordinates.tolist()

    def step(self, action):
        """执行动作并更新状态"""
        x, y = self.state
        if action == 0:  # 上
            y -= 1
            self.agent_angle = 90  # 车朝上
        elif action == 1:  # 下
            y += 1
            self.agent_angle = 270  # 车朝下
        elif action == 2:  # 左
            x -= 1
            self.agent_angle = 180  # 车朝左
        elif action == 3:  # 右
            x += 1
            self.agent_angle = 0  # 车朝右
        elif action == 4:  # 左上
            x -= 1
            y -= 1
            self.agent_angle = 135  # 车朝左上
            # 检查与左上方向垂直的格子 (1, 0) 或 (0, 1) 是否有障碍物
            if self.is_catercorner_obstacle(x + 1, y) or self.is_catercorner_obstacle(x, y + 1):
                return self.state, False
        elif action == 5:  # 右上
            x += 1
            y -= 1
            self.agent_angle = 45  # 车朝右上
            # 检查与右上方向垂直的格子 (1, 0) 或 (0, -1) 是否有障碍物
            if self.is_catercorner_obstacle(x + 1, y) or self.is_catercorner_obstacle(x, y - 1):
                return self.state, False
        elif action == 6:  # 左下
            x -= 1
            y += 1
            self.agent_angle = 225  # 车朝左下
            # 检查与左下方向垂直的格子 (-1, 0) 或 (0, 1) 是否有障碍物
            if self.is_catercorner_obstacle(x - 1, y) or self.is_catercorner_obstacle(x, y + 1):
                return self.state, False
        elif action == 7:  # 右下
            x += 1
            y += 1
            self.agent_angle = 315  # 车朝右下
            # 检查与右下方向垂直的格子 (-1, 0) 或 (0, -1) 是否有障碍物
            if self.is_catercorner_obstacle(x - 1, y) or self.is_catercorner_obstacle(x, y - 1):
                return self.state, False
        print(f"<向————>{self.action_translate[action]}————移动>")
        print(f"旋转角度:{self.agent_angle}-度")
        # 到达目标则重新开始
        if (x, y) == self.goal:
            return self.reset()

        # 检查是否越界
        if x < 0 or x >= self.grid_size[0] or y < 0 or y >= self.grid_size[1]:
            return self.state, False  # 无效动作，状态不变
        if x < self.start[0] or x > self.goal[0] - 1:
            if x < self.start[0]:
                print(f"到达左边界")
                self.state = (x + 1, y)
                self.agent_angle = 0
            else:
                print(f"到达右边界")
                self.state = (x - 1, y)
                self.agent_angle = 180
        elif  y < self.start[1] or y > self.goal[1] - 1:
            if y < self.start[1]:
                print(f"到达上边界")
                self.state = (x, y + 1)
                self.agent_angle = 270
            else:
                print(f"到达下边界")
                self.state = (x, y - 1)
                self.agent_angle = 90

        # 检查是否撞到障碍物
        corner_coordinates = self.get_car_four_coord((x, y))
        corner_coordinates = list(map(lambda corner: tuple(corner), corner_coordinates))

        # 将较大的列表转为集合
        set1 = set(self.obstacle_lists)
        set2 = set(corner_coordinates)

        # 检查是否有交集，并打印交集（碰撞点）
        collision_points = set1 & set2
        if collision_points:
            print(f"检测到碰撞：{collision_points}，位置: {(x, y)}")

            # 可视化碰撞点
            for point in collision_points:
                print(f"碰撞点：{point}")  # 调试输出，查看碰撞点
                # 使用红色小圆点标记碰撞点
                self.light_blink_2(color=(255,140,0), p=list(point))
            # 更新显示
            pygame.display.flip()
            return self.state, False  # 碰到障碍物，状态不变
        else:
            if len(corner_coordinates) > 0:
                for corner in corner_coordinates:
                    # print(f"小车直角点：{corner}")  # 调试输出，查看碰撞点
                    # 使用红色小圆点标记碰撞点
                    self.light_blink(color=(128, 0, 0), p=list(corner))
                    # 更新显示
                    pygame.display.flip()

        self.state = (x, y)

        return self.state, False  # 动作有效，返回新的状态

    def is_catercorner_obstacle(self, x, y):
        """
        对角线方向是否存在障碍物
        :param x:
        :param y:
        :return:
        """
        if (x, y) in self.obstacles:
            print(f"障碍物挡住了路径 ({x}, {y})")
            return True
        return False

    def random_walk(self):
        """执行随机漫步并可视化"""
        self.reset()  # 重置环境
        running = True
        # 更新可视化
        if self.open_render:
            self.render(agent_angle = 0, coordinate = self.start)
        while running:
            # 获取当前角度
            current_angle = self.agent_angle

            # 随机选择多个动作
            actions = np.random.choice(range(0, 8), size = 8, replace = False)

            # 选择符合当前角度的有效动作
            valid_actions = [
                action for action in actions  # 遍历选择的动作
                if abs(self.get_action_angle(action) - current_angle) <= 45
            ]
            if len(valid_actions) > 0:
                # 随机选择一个符合条件的动作，使用归一化后的权重
                action = np.random.choice(valid_actions)

            # 执行动作
            self.step(action)

            # 更新可视化
            if self.open_render:
                self.render(agent_angle = self.get_action_angle(action), coordinate = self.state)

            # 延时，展示每步的变化
            time.sleep(0.2)

            # 处理退出事件（例如按下回车按钮）
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:  # 检测按键按下事件
                    if event.key == pygame.K_RETURN:  # 判断是否按下回车键
                        running = False

        # 程序结束时退出
        pygame.quit()

if __name__ == "__main__":
    # 创建环境实例并开始随机漫步
    # 设置为 5x5 的网格，并添加障碍物
    # 如果有目标图像路径传入，将加载目标图像
    env = PathPlanningEnv()
    env.random_walk()
