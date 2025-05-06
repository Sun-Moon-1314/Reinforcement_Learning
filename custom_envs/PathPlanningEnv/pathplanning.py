import math

import pygame
import random
import time

# 默认障碍物列表
_default = [
    (1, 2),(1, 4),
    (2, 1),
    (3, 3), (3, 4),
    (4, 1)
]

# 定义智能体类
class PathPlanningEnv:
    def __init__(self, grid_size=(5, 5), start=(0, 0), goal=None, cell_size=50, obstacles=None, render = True):
        self.screen = None
        self.grid_size = grid_size  # 网格大小
        self.start = start  # 起始点
        self.goal = goal if goal else (grid_size[0]-1, grid_size[1]-1) # 目标点
        self.state = self.start  # 当前状态（位置）
        self.cell_size = cell_size  # 每个格子的像素大小
        self.obstacles = obstacles if obstacles else _default  # 障碍物列表
        self.open_render = render # 可视化控制开关

        # 动态计算屏幕大小
        self.screen_width = grid_size[1] * cell_size
        self.screen_height = grid_size[0] * cell_size
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
        # 设置颜色
        self.start_color = (60,179,113)  # 薄荷绿，表示起点
        self.goal_color = (240, 128, 128)  # 珊瑚粉，表示终点
        self.obstacle_color = (147,112,219)  # 深紫罗兰，表示障碍物
        self.background_color = (245, 245, 245)  # 浅米灰，柔和的背景颜色

    def env_show_init(self):
        """
        环境可视化
        :return:
        """
        # 初始化 Pygame
        pygame.init()
        # 初始化 Pygame 屏幕
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("***路径规划***")

        # 加载车的图像（可以是任意图形）
        try:
            self.agent_image = pygame.image.load("module_pic/super_car.png")  # 假设图像朝右
            self.agent_image = pygame.transform.scale(self.agent_image, (self.cell_size, self.cell_size))  # 调整大小
        except pygame.error:
            print("加载图像失败，请检查图片路径和格式！")
            self.agent_image = pygame.Surface((self.cell_size, self.cell_size))  # 如果加载失败，使用默认图形
            self.agent_image.fill((0, 0, 255))  # 蓝色小方块作为默认图形

        # 加载目标图像（如果指定了目标图像路径）
        try:
            self.goal_image = pygame.image.load("module_pic/flag2.png")
            self.goal_image = pygame.transform.scale(self.goal_image, (self.cell_size, self.cell_size))
        except pygame.error:
            print("加载目标图像失败，请检查图片路径和格式！")
            self.goal_image = pygame.Surface((self.cell_size, self.cell_size))  # 默认的目标图形
            self.goal_image.fill((255, 0, 0))  # 红色方块作为默认目标图形

        # 加载障碍物图像（如果指定了障碍物图像路径）
        try:
            self.obstacle_image = pygame.image.load("module_pic/obstacle_pic.png")
            if self.obstacle_image:
                self.obstacle_image = pygame.transform.scale(self.obstacle_image, (self.cell_size, self.cell_size))
        except pygame.error:
            print("加载障碍物图像失败，请检查图片路径和格式！")
            self.obstacle_image = None  # 如果没有指定图像，则不使用图像，使用默认图形
    def reset(self):
        """重置环境"""
        self.state = self.start  # 重置为起点
        return self.state

    def step(self, action):
        """执行动作并更新状态"""
        x, y = self.state
        if action == 0:  # 上
            x -= 1
            self.agent_angle = 90  # 车朝上
        elif action == 1:  # 下
            x += 1
            self.agent_angle = 270  # 车朝下
        elif action == 2:  # 左
            y -= 1
            self.agent_angle = 180  # 车朝左
        elif action == 3:  # 右
            y += 1
            self.agent_angle = 0  # 车朝右
        elif action == 4:  # 左上
            x -= 1
            y -= 1
            self.agent_angle = 135  # 车朝左上
            # 检查与左上方向垂直的格子 (1, 0) 或 (0, 1) 是否有障碍物
            if self.is_catercorner_obstacle(x + 1, y) or self.is_catercorner_obstacle(x, y + 1):
                return self.state, False
        elif action == 5:  # 右上
            x -= 1
            y += 1
            self.agent_angle = 45  # 车朝右上
            # 检查与右上方向垂直的格子 (1, 0) 或 (0, -1) 是否有障碍物
            if self.is_catercorner_obstacle(x + 1, y) or self.is_catercorner_obstacle(x, y - 1):
                return self.state, False
        elif action == 6:  # 左下
            x += 1
            y -= 1
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
        # 到达目标则重新开始
        if (x, y) == self.goal:
            return self.reset()

        # 检查是否越界
        if x < 0 or x >= self.grid_size[0] or y < 0 or y >= self.grid_size[1]:
            return self.state, False  # 无效动作，状态不变

        # 检查是否撞到障碍物
        if (x, y) in self.obstacles:
            return self.state, False  # 碰到障碍物，状态不变

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

    def render(self):
        """更新可视化"""
        self.screen.fill(self.background_color)  # 填充背景颜色

        # 绘制网格并显示坐标
        font = pygame.font.SysFont("Arial", 10)  # 创建字体对象

        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                # 绘制格子
                pygame.draw.rect(self.screen, (200, 200, 200),
                                 (y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size), 1)

                # 只在第一行和第一列显示坐标
                if x == 0 or y == 0:
                    text_surface = font.render(f"{x if y == 0 else y}", True, (255,240,245))
                    self.screen.blit(text_surface, (y * self.cell_size + 5, x * self.cell_size + 5))  # 显示坐标

        # 绘制障碍物
        for obstacle in self.obstacles:
            if self.obstacle_image:  # 如果有障碍物图片
                pygame.draw.rect(self.screen, self.obstacle_color,
                                 (obstacle[1] * self.cell_size, obstacle[0] * self.cell_size, self.cell_size,
                                  self.cell_size))
                self.screen.blit(self.obstacle_image, (obstacle[1] * self.cell_size, obstacle[0] * self.cell_size))

            else:  # 如果没有图片，绘制默认图形
                pygame.draw.rect(self.screen, self.obstacle_color,
                                 (obstacle[1] * self.cell_size, obstacle[0] * self.cell_size, self.cell_size,
                                  self.cell_size))

        # 绘制起点
        pygame.draw.rect(self.screen, self.start_color,
                         (self.start[1] * self.cell_size, self.start[0] * self.cell_size, self.cell_size,
                          self.cell_size))

        # 绘制目标点
        pygame.draw.rect(self.screen, self.goal_color,
                         (self.goal[1] * self.cell_size, self.goal[0] * self.cell_size, self.cell_size,
                          self.cell_size))

        # 绘制智能体
        goal_rect = pygame.Rect(self.goal[1] * self.cell_size, self.goal[0] * self.cell_size, self.cell_size,
                                self.cell_size)
        self.screen.blit(self.goal_image, goal_rect.topleft)
        if self.agent_angle in [0, 45, 90, 270, 315]:  # 上下方向
            rotated_image = pygame.transform.rotate(self.agent_image, self.agent_angle)
        elif self.agent_angle in [180]:  # 左或右
            rotated_image = pygame.transform.flip(self.agent_image, self.agent_angle == 180, False)
        elif self.agent_angle in [135, 225]:  # 左侧对角线
            flipped_image = pygame.transform.flip(self.agent_image, True, False)  # 先水平镜像
            rotated_image = pygame.transform.rotate(flipped_image, self.agent_angle - 180)  # 再旋转

        # 获取旋转后或翻转后的图像的矩形并居中
        new_rect = rotated_image.get_rect(center=(self.state[1] * self.cell_size + self.cell_size // 2,
                                                  self.state[0] * self.cell_size + self.cell_size // 2))

        # 绘制图像
        self.screen.blit(rotated_image, new_rect.topleft)

        pygame.display.flip()  # 更新屏幕

    def random_walk(self):
        """执行随机漫步并可视化"""
        self.reset()  # 重置环境
        running = True
        # 更新可视化
        if self.open_render:
            self.render()
        while running:
            # 随机选择一个动作并执行
            action = random.randint(0, 7)  # 八个动作：上、下、左、右、左上、右上、左下、右下

            self.step(action)

            # 更新可视化
            if self.open_render:
                self.render()

            # 延时，展示每步的变化
            time.sleep(2)

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
    env = PathPlanningEnv(grid_size=(7, 7))
    env.random_walk()
