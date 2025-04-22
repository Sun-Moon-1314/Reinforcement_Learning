# -*- coding: utf-8 -*-
"""
@File    : bullet_SB3.py
@Time    : 2025/3/25 19:44
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    : 
"""
import pybullet as p

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from planet_war_robot import CircularMotionEnv


def train(render=p.DIRECT):
    # 检查环境
    env = CircularMotionEnv(render)
    check_env(env)

    # 使用 PPO 算法进行训练
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0001)
    model.learn(total_timesteps=1000)
    # 保存模型
    model.save("ppo_circular_motion")
    print("Model saved!")
    env.close()


def evaluate(render=p.GUI):
    # 加载模型
    loaded_model = PPO.load("ppo_circular_motion")
    print("Model loaded!")
    # 检查环境
    env = CircularMotionEnv(render)
    check_env(env)
    # 测试训练好的模型
    obs, _ = env.reset()
    for _ in range(1000):
        action, _states = loaded_model.predict(obs)
        obs, rewards, done, truncated, info = env.step(action)
        env.render()
        if done:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    train(p.DIRECT)
    evaluate(p.GUI)
