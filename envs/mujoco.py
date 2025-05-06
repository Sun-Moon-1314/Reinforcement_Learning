# -*- coding: utf-8 -*-
"""
@File    : fruitfly_mujoco_env.py
@Time    : 2025/4/28 19:22
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    : 基于 fruitfly.xml 的 MuJoCo 强化学习环境
"""
import mujoco
import mujoco.viewer
import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from ray.rllib.algorithms import PPOConfig


class FruitFlyEnv(gym.Env):
    def __init__(self):
        super(FruitFlyEnv, self).__init__()
        # 指定 XML 文件路径（根据您的实际路径调整）
        # 获取当前文件的绝对路径，并构建相对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "..", "env_scenes", "mujoco_menagerie", "flybody", "scene.xml")
        xml_path = os.path.abspath(xml_path)  # 确保路径是绝对路径
        print("XML 文件路径:", xml_path)

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML 文件未找到：{xml_path}")

        # 加载 MuJoCo 模型
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)  # 使用 mj_model 避免潜在冲突
        self.mj_data = mujoco.MjData(self.mj_model)

        self.viewer = None  # 初始化 viewer 为 None

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.mj_model.nu,), dtype=np.float32
        )
        obs_size = self.mj_model.nq + self.mj_model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self._get_obs(), {}

    def step(self, action):
        self.mj_data.ctrl[:] = action
        mujoco.mj_step(self.mj_model, self.mj_data)
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # 返回关节位置和速度
        return np.concatenate([self.mj_data.qpos.flat, self.mj_data.qvel.flat]).astype(np.float32)

    def _get_reward(self):
        # 示例奖励函数：可以根据您的任务自定义
        return -np.sum(np.square(self.mj_data.qpos))

    def render(self):
        # 如果 viewer 尚未初始化，则创建 viewer
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        else:
            # 更新 viewer
            self.viewer.sync()

    def close(self):
        # 关闭 viewer（如果存在）
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        super().close()


if __name__ == "__main__":
    # 创建并检查环境
    env = FruitFlyEnv()
    check_env(env)

    # 使用 PPO 训练智能体
    ppo_model = PPO("MlpPolicy", env, verbose=1)  # 使用 ppo_model 明确区分
    ppo_model.learn(total_timesteps=100000)

    # 测试训练好的智能体
    obs, _ = env.reset()
    for _ in range(10000):
        action, _states = ppo_model.predict(obs)  # 使用 ppo_model
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()  # 调用 render 方法，显示可视化窗口
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()  # 确保在结束时关闭 viewer
    print("测试完成")
