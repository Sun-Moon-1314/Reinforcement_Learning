# -*- coding: utf-8 -*-
"""
@File    : bullet_ray.py
@Time    : 2025/3/25 20:00
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    : 
"""

import gymnasium as gym
import ray
from gymnasium import spaces
import pybullet as p
import torch
import os
import pybullet_data
import math
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import numpy as np
from gymnasium.spaces import Box
import logging
from ray.rllib.core.rl_module import RLModule
from ray.rllib.algorithms.algorithm import Algorithm

logger = logging.getLogger(__name__)  # 使用当前模块名

print(f"------Ray Version: {ray.__version__}------")

# 设置 gymnasium 的日志级别
# 设置全局日志级别为 ERROR
logging.basicConfig(level=logging.ERROR)
# 修复 Box 类型
observation_space = Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                        high=np.array([1.0, 1.0], dtype=np.float32))

current_dir = os.getcwd()
# 设置保存路径为绝对路径
checkpoint = os.path.join(current_dir, "ray_results")


# 自定义环境
class CircularMotionEnv(gym.Env):
    def __init__(self, radius=2.0, speed=5.0, center=None, max_steps=500, render_mode=None):
        super(CircularMotionEnv, self).__init__()

        # 环境参数
        if center is None:
            center = [0, 0]
        self.target_x = None
        self.target_y = None
        self.radius = radius
        self.speed = speed
        self.center = center
        self.max_steps = max_steps
        self.current_step = 0

        # PyBullet 初始化
        if render_mode == p.GUI:
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
        if hasattr(self, 'client') and self.client is not None:
            try:
                import pybullet as p
                p.disconnect(self.client)
            except Exception as e:
                print(f"Error during disconnect: {e}")
            self.client = None

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            print(f"Exception ignored in __del__: {e}")


def ray_train(checkpoint):
    # 创建训练器，会自动调用ray.init()
    trainer = config.build()
    try:
        for i in range(10000):
            result = trainer.train()
            if "episode_reward_mean" in result:
                print(f"Iteration {i}: reward = {result['episode_reward_mean']}")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    # 保存模型
    trainer.save(checkpoint)
    print(f"Model saved at {checkpoint}")
    module = trainer.get_module("default_policy")
    module.save_to_path(checkpoint)
    print("Model restored!")

    print("Cleaning up resources...")
    # 停止训练器
    trainer.stop()
    # 关闭 Ray
    ray.shutdown()
    print("Resources cleaned up successfully.")


def ray_evaluate(checkpoint_path):
    # 评估初始化
    ray.init(ignore_reinit_error=True, num_cpus=1, local_mode=True)

    # 手动创建环境实例
    env_instance = CircularMotionEnv(render_mode=p.GUI)

    # 直接加载 RLModule
    module = RLModule.from_checkpoint(checkpoint_path)

    # 推理逻辑
    obs, _ = env_instance.reset()
    for _ in range(1000):
        obs_tensor = torch.tensor([obs], dtype=torch.float32)
        obs_batch = {"obs": obs_tensor}
        output = module.forward_inference(obs_batch)
        action = output["action_dist_inputs"][0].numpy()
        obs, reward, done, truncated, info = env_instance.step(action)
        env_instance.render()
        if done:
            obs, _ = env_instance.reset()

    env_instance.close()
    ray.shutdown()


# 定义环境创建函数
def env_creator(env_config):
    """
    创建 CircularMotionEnv 环境实例。

    Args:
        env_config (dict): 环境配置字典，包含 render_mode 等参数。

    Returns:
        CircularMotionEnv: 环境实例。
    """
    render_mode = env_config.get("render_mode", p.DIRECT)
    return CircularMotionEnv(render_mode=render_mode)


if __name__ == "__main__":
    # 注册自定义环境
    env_config = {
        "render_mode": p.DIRECT  # 默认使用 DIRECT 模式（无 GUI，适合训练）
    }
    register_env("CircularMotionEnv", env_creator)
    # 配置 PPO 算法
    config = (
        PPOConfig()
        .environment(env="CircularMotionEnv", env_config=env_config)  # 注册的环境名称
        .framework("torch")  # 使用 PyTorch 框架（也可以选择 "tf"）
        .env_runners(num_env_runners=1)  # 使用新版本的 env_runner API
        .training(
            lr=0.0001,  # 学习率
            train_batch_size=4000,  # 训练批次大小
            num_sgd_iter=10  # SGD 迭代次数
        )
    )
    ray_train(checkpoint)
    # 调用推理函数
    ray_evaluate(checkpoint)
