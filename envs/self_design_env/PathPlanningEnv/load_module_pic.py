# -*- coding: utf-8 -*-
"""
@File    : load_module_pic.py
@Time    : 2025/1/3
@Author  : <your_name>
@Email   : <your_email>
@Desc    : Render a small region of a large grid map with a car at a specified position and rotation.
"""

import pygame
import numpy as np
_default = [
    (30, 50), (30, 70), (50, 80), (100, 100),(100,30)
]
class ImageToGrid:
    def __init__(self, start, goal, obstacles=None, grid_size=None, car_width=10, car_height=10):
        self.obstacle_image = None
        self.goal_image = None
        self.cell_size = 5  # 每个格子的像素大小
        self.start = start if start else (car_width // 2, car_height // 2)# 起始点
        if self.start[0] < car_width // 2 or self.start[1] < car_height // 2:
            print("输入不符合规定")
            return
        self.goal = goal if goal else (grid_size[0], grid_size[1]) # 目标点
        self.state = self.start  # 当前状态（位置）
        self.grid_size = grid_size  # 地图的尺寸
        self.car_width = car_width * self.cell_size  # 小车的宽度
        self.car_height = car_height * self.cell_size  # 小车的高度
        self.obstacles = obstacles if obstacles else _default  # 障碍物列表
        self.obstacle_width = 20 * self.cell_size  # 障碍物的宽度
        self.obstacle_height = 20 * self.cell_size  # 障碍物的高度
        self.obstacle_lists = []
        self.goal_width = 10 * self.cell_size  # 障碍物的宽度
        self.goal_height = 10 * self.cell_size  # 障碍物的高度
        self.goal_lists = []
        self.screen_width = grid_size[1] * self.cell_size
        self.screen_height = grid_size[0] * self.cell_size

        # 设置颜色
        self.start_color = (60, 179, 113)  # 薄荷绿，表示起点
        self.goal_color = (	165,42,42)  # 珊瑚粉，表示终点
        self.obstacle_color = (124,252,0)  # 深紫罗兰，表示障碍物
        self.background_color = (119,136,153)  # 浅米灰，柔和的背景颜色
    def draw_grid(self, image, coordinate, angle):
        """
        绘制小车、终点，陷阱等图像
        :param coordinate: 中心坐标
        :param width: 宽度
        :param height: 长度
        :param angle: 旋转角度
        :return:
        """
        # 计算小车显示的范围
        center_x, center_y = coordinate
        # 旋转小车图像
        rotated_image = pygame.transform.rotate(image, angle)
        # 获取旋转后图像的尺寸，以确保正确位置
        rotated_rect = rotated_image.get_rect(center=(center_x * self.cell_size,
                                                      center_y * self.cell_size))

        # 绘制旋转后的图像
        self.screen.blit(rotated_image, rotated_rect.topleft)

    def env_show_init(self):
        """
        环境可视化
        :return:
        """
        # 初始化 Pygame
        pygame.init()
        # 初始化 Pygame 屏幕
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("***复杂场景-路径规划模拟器***")

        # 加载车的图像（可以是任意图形）
        try:
            self.agent_image = pygame.image.load("module_pic/agent_car.png")
            self.agent_image = pygame.transform.scale(self.agent_image, (self.car_width, self.car_height))  # 缩放到小车大小
        except pygame.error:
            print("加载图像失败，请检查图片路径和格式！")
            self.agent_image = pygame.Surface((self.cell_size, self.cell_size))  # 如果加载失败，使用默认图形
            self.agent_image.fill((0, 0, 255))  # 蓝色小方块作为默认图形

        # 加载目标图像（如果指定了目标图像路径）
        try:
            self.goal_image = pygame.image.load("module_pic/flag2.png")
            self.goal_image = pygame.transform.scale(self.goal_image, (self.goal_width, self.goal_height))
        except pygame.error:
            print("加载目标图像失败，请检查图片路径和格式！")
            self.goal_image = pygame.Surface((self.cell_size, self.cell_size))  # 默认的目标图形
            self.goal_image.fill((255, 0, 0))  # 红色方块作为默认目标图形

        # 加载障碍物图像（如果指定了障碍物图像路径）
        try:
            self.obstacle_image = pygame.image.load("module_pic/hourse.png")
            if self.obstacle_image:
                self.obstacle_image = pygame.transform.scale(self.obstacle_image, (self.obstacle_width, self.obstacle_height))
        except pygame.error:
            print("加载障碍物图像失败，请检查图片路径和格式！")
            self.obstacle_image = None  # 如果没有指定图像，则不使用图像，使用默认图形

    def render(self, agent_angle, coordinate):
        """
        图像显示
        :param agent_angle: 智能体旋转角度
        :param coordinate: 智能体坐标
        :return:
        """
        self.screen.fill(self.background_color)  # 填充背景颜色

        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                # 绘制格子
                pygame.draw.rect(self.screen, (105,105,105),
                                 (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size), 1)
        # 绘制障碍物
        for obstacle in self.obstacles:
            if self.obstacle_image:  # 如果有障碍物图片
                for width in range(self.obstacle_width):
                    for height in range(self.obstacle_height):
                        ob_x = (obstacle[0] * self.cell_size - self.obstacle_width // 2 + width + 1)
                        ob_y = (obstacle[1] * self.cell_size - self.obstacle_height // 2 + height + 1)
                        pygame.draw.rect(self.screen, self.obstacle_color,(ob_x, ob_y, self.cell_size, self.cell_size))
                        self.obstacle_lists.append((ob_x // self.cell_size, ob_y // self.cell_size))
                self.draw_grid(self.obstacle_image, obstacle, angle=0)
            else:  # 如果没有图片，绘制默认图形
                for width in range(self.obstacle_width):
                    for height in range(self.obstacle_height):
                        ob_x = (obstacle[0] * self.cell_size - self.obstacle_width // 2 + width + 1)
                        ob_y = (obstacle[1] * self.cell_size - self.obstacle_height // 2 + height + 1)
                        pygame.draw.rect(self.screen, self.obstacle_color,
                                         (ob_x, ob_y, self.cell_size, self.cell_size))
                        self.obstacle_lists.append((ob_x // self.cell_size, ob_y // self.cell_size))

        # 绘制起点
        pygame.draw.rect(self.screen, self.start_color,
                         (self.start[1] * self.cell_size, self.start[0] * self.cell_size, self.cell_size,
                          self.cell_size))

        # 绘制目标点,填充颜色
        for width in range(self.goal_width):
            for height in range(self.goal_height):
                goal_x = self.goal[0] * self.cell_size- self.goal_width // 2 + width + 1
                goal_y = self.goal[1] * self.cell_size - self.goal_width // 2 + height + 1
                pygame.draw.rect(self.screen, self.goal_color,
                                 (goal_x, goal_y , self.cell_size, self.cell_size))
                self.goal_lists.append((goal_x, goal_y))
        # 加载图片
        self.draw_grid(self.goal_image, self.goal, angle=0)

        # 绘制智能体
        self.draw_grid(self.agent_image, coordinate, angle=agent_angle)

        pygame.display.flip()  # 更新屏幕

    def light_blink(self, color ,p):
        """
        闪烁灯
        :param point:
        :return:
        """
        pygame.draw.circle(self.screen, color, (p[0] * self.cell_size, p[1] * self.cell_size), 5)
        p.clear()

    def light_blink_2(self, color, p):
        """
        闪烁灯
        :param point:
        :return:
        """
        pygame.draw.circle(self.screen, color, (p[0] * self.cell_size, p[1] * self.cell_size), 5)
        p.clear()
    def run(self):
        """主循环"""
        running = True
        while running:
            agent_angle = np.random.choice([0,45,90,135,180,225,270,315,360])
            coordinate = (np.random.randint(self.car_width, self.grid_size[0]-self.car_width), np.random.randint(self.car_height, self.grid_size[1]-self.car_height))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.render(agent_angle=agent_angle, coordinate=coordinate)
            # time.sleep(1)
            pygame.display.flip()

        pygame.quit()


# 示例运行
# image_to_grid = ImageToGrid(
#     start=(5, 5),  # 小车的中心坐标 (x, y)
#     goal=(190,190),
# )
# image_to_grid.env_show_init()
# image_to_grid.run()
