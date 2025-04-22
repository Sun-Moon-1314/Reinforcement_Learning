# -*- coding: utf-8 -*-
"""
@File    : planet_war_robot.py
@Time    : 2025/3/26 14:32
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    : 
"""
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math


class CircularMotionEnv(gym.Env):
    def __init__(self, render=None, radius=2.0, speed=5.0, center=None, max_steps=500):
        super(CircularMotionEnv, self).__init__()

        # 环境参数
        if center is None:
            center = [0, 0]
        self.radius = radius
        self.speed = speed
        self.center = center
        self.max_steps = max_steps
        self.current_step = 0

        # PyBullet 初始化
        if render == p.GUI:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)  # 使用 DIRECT 模式加速训练
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0.1])

        # 动作空间：控制小车的线速度和角速度
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # 状态空间：小车的位置、速度和目标位置
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        # 初始化目标点
        self.theta = 0.0
        self.update_target_position()

    def update_target_position(self):
        """更新目标点位置"""
        self.target_x = self.center[0] + self.radius * math.cos(self.theta)
        self.target_y = self.center[1] + self.radius * math.sin(self.theta)

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        self.current_step = 0
        self.theta = 0.0
        self.update_target_position()
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0]))
        return self._get_obs(), {}

    def step(self, action):
        """执行一步仿真"""
        self.current_step += 1

        # 解析动作
        linear_velocity = action[0] * self.speed  # 线速度
        angular_velocity = action[1] * math.pi  # 角速度（限制在 [-π, π]）

        # 获取当前小车位置和朝向
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        x, y, _ = pos
        yaw = p.getEulerFromQuaternion(orn)[2]

        # 更新小车位置
        new_x = x + linear_velocity * math.cos(yaw) * (1 / 240)
        new_y = y + linear_velocity * math.sin(yaw) * (1 / 240)
        new_yaw = yaw + angular_velocity * (1 / 240)
        p.resetBasePositionAndOrientation(self.robot_id, [new_x, new_y, 0.1], p.getQuaternionFromEuler([0, 0, new_yaw]))

        # 更新目标点
        self.theta += self.speed / self.radius * (1 / 240)
        self.update_target_position()

        # 计算奖励
        distance_to_target = math.sqrt((new_x - self.target_x) ** 2 + (new_y - self.target_y) ** 2)
        reward = -distance_to_target  # 奖励为与目标点的负距离

        # 判断是否结束
        done = self.current_step >= self.max_steps
        truncated = False  # 在本例中不使用截断

        return self._get_obs(), reward, done, truncated, {}

    def _get_obs(self):
        """获取当前状态"""
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        x, y, _ = pos
        obs = np.array([x, y, self.target_x, self.target_y, self.theta], dtype=np.float32)
        return obs

    def render(self, mode="human"):
        """渲染环境"""
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0],
                                                          distance=10,
                                                          yaw=50,
                                                          pitch=-35,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=320, height=240, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
        return px

    def close(self):
        """关闭环境"""
        p.disconnect()
