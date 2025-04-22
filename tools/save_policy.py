# -*- coding: utf-8 -*-
"""
@File    : save_policy.py      # 文件名，save_policy表示当前文件名
@Time    : 2024/12/31         # 创建时间，2024/12/31表示当前时间
@Author  : <your_name>     # 作者
@Email   : <your_email>    # 作者电子邮件
@Desc    : <brief_description> # 文件的简要描述
"""
import pickle
import torch
import keras
import numpy as np
import os

from envs.blackjack import logger


class Policy_loader:
    policy_dir = os.path.join(os.path.dirname(__file__), 'policy_dir')
    if not os.path.exists(policy_dir):
        os.makedirs(policy_dir)
    save_dir = None

    @staticmethod
    def save_policy(method_name, class_name, policy, **kwargs):
        step = kwargs.get("step", 1)
        if method_name is None:
            method_name = "default"
        policy_dir = os.path.join(Policy_loader.policy_dir, class_name)
        if not os.path.exists(policy_dir):
            os.makedirs(policy_dir)
        Policy_loader.save_dir = os.path.join(policy_dir, method_name)
        if isinstance(policy, dict) and "encoder" in policy:
            with open(f"{Policy_loader.save_dir}.pkl", "wb") as f:
                pickle.dump(policy, f)
        elif isinstance(policy, list):
            np.savetxt(f'{Policy_loader.save_dir}.csv', policy, delimiter=',', fmt='%.6f')
        elif "evaluate_net_pytorch" in policy:
            evaluate_net_dir_py = os.path.join(policy_dir, f'evaluate_net_pytorch')
            target_net_dir_py = os.path.join(policy_dir, f'target_net_pytorch')
            # 保存 evaluate_net_pytorch 和 target_net_pytorch 的模型权重（state_dict）
            torch.save({'model_state_dict': policy['evaluate_net_pytorch'].state_dict(),
                        'optimizer_state_dict': policy['optimizer'].state_dict(), }, f'{evaluate_net_dir_py}.pth')

            torch.save({'model_state_dict': policy['target_net_pytorch'].state_dict(),
                        'optimizer_state_dict': policy['optimizer'].state_dict(), }, f'{target_net_dir_py}.pth')
            logger.info(f"保存-->evaluate_net_pytorch+-->target_net_pytorch模型")
        elif "ddqn_evaluate_net_pytorch" in policy:
            evaluate_net_dir_py = os.path.join(policy_dir, 'ddqn_evaluate_net_pytorch')
            target_net_dir_py = os.path.join(policy_dir, 'ddqn_target_net_pytorch')
            # 保存 evaluate_net_pytorch 和 target_net_pytorch 的模型权重（state_dict）
            torch.save({'model_state_dict': policy['ddqn_evaluate_net_pytorch'].state_dict(),
                        'optimizer_state_dict': policy['ddqn_optimizer'].state_dict(), }, f'{evaluate_net_dir_py}.pth')

            torch.save({'model_state_dict': policy['ddqn_target_net_pytorch'].state_dict(),
                        'optimizer_state_dict': policy['ddqn_optimizer'].state_dict(), }, f'{target_net_dir_py}.pth')
            logger.info(f"保存-->ddqn_evaluate_net_pytorch+-->ddqn_target_net_pytorch模型")
            # torch.save(policy['target_net_pytorch'].state_dict(), f'{target_net_dir_py}.pth')
        elif "policy_net" in policy:
            policy_net_dir_py = os.path.join(policy_dir, f'policy_net')
            baseline_net_dir_py = os.path.join(policy_dir, f'baseline_net')
            # 保存 evaluate_net_pytorch 和 target_net_pytorch 的模型权重（state_dict）
            torch.save({'model_state_dict': policy['policy_net'].state_dict(),
                        'optimizer_state_dict': policy['policy_optimizer'].state_dict(), }, f'{policy_net_dir_py}.pth')
            torch.save({'model_state_dict': policy['baseline_net'].state_dict(),
                        'optimizer_state_dict': policy['baseline_optimizer'].state_dict(), },
                       f'{baseline_net_dir_py}.pth')
            logger.info(f"保存-->policy_net+-->baseline_net_dir_py模型")
        elif "off_policy_net" in policy:
            off_policy_net_dir_py = os.path.join(policy_dir, f'off_policy_net')
            off_baseline_net_dir_py = os.path.join(policy_dir, f'off_baseline_net')
            # 保存 evaluate_net_pytorch 和 target_net_pytorch 的模型权重（state_dict）
            torch.save({'model_state_dict': policy['off_policy_net'].state_dict(),
                        'optimizer_state_dict': policy['off_policy_optimizer'].state_dict(), },
                       f'{off_policy_net_dir_py}.pth')
            torch.save({'model_state_dict': policy['off_baseline_net'].state_dict(),
                        'optimizer_state_dict': policy['off_baseline_optimizer'].state_dict(), },
                       f'{off_baseline_net_dir_py}.pth')
            logger.info(f"保存-->off_policy_net+-->off_baseline_net模型")
        elif "actor" in policy:
            actor_dir_py = os.path.join(policy_dir, f'actor')
            critic_dir_py = os.path.join(policy_dir, f'critic')
            # 保存 evaluate_net_pytorch 和 target_net_pytorch 的模型权重（state_dict）
            torch.save({'model_state_dict': policy['actor'].state_dict(),
                        'optimizer_state_dict': policy['actor_optimizer'].state_dict(), }, f'{actor_dir_py}.pth')
            torch.save({'model_state_dict': policy['critic'].state_dict(),
                        'optimizer_state_dict': policy['critic_optimizer'].state_dict(), },
                       f'{critic_dir_py}.pth')
            logger.info(f"保存-->actor+-->critic模型")
        elif "ad_actor" in policy:
            actor_dir_py = os.path.join(policy_dir, f'ad_actor')
            critic_dir_py = os.path.join(policy_dir, f'ad_critic')
            # 保存 evaluate_net_pytorch 和 target_net_pytorch 的模型权重（state_dict）
            torch.save({'model_state_dict': policy['ad_actor'].state_dict(),
                        'optimizer_state_dict': policy['ad_actor_optimizer'].state_dict(), }, f'{actor_dir_py}.pth')
            torch.save({'model_state_dict': policy['ad_critic'].state_dict(),
                        'optimizer_state_dict': policy['ad_critic_optimizer'].state_dict(), },
                       f'{critic_dir_py}.pth')
            logger.info(f"保存-->ad_actor+-->ad_critic模型")
        elif "lambda_actor" in policy:
            actor_dir_py = os.path.join(policy_dir, f'lambda_actor')
            critic_dir_py = os.path.join(policy_dir, f'lambda_critic')
            # 保存 evaluate_net_pytorch 和 target_net_pytorch 的模型权重（state_dict）
            torch.save({'model_state_dict': policy['lambda_actor'].state_dict(),
                        'optimizer_state_dict': policy['lambda_actor_optimizer'].state_dict(),
                        'actor_e_traces': policy['actor_e_traces'], }, f'{actor_dir_py}.pth')
            torch.save({'model_state_dict': policy['lambda_critic'].state_dict(),
                        'critic_e_traces': policy['critic_e_traces'], },
                       f'{critic_dir_py}.pth')
            logger.info(f"保存-->lambda_actor+-->lambda_critic模型")
        elif "a2c_actor" in policy:
            actor_dir_py = os.path.join(policy_dir, f'a2c_actor')
            critic_dir_py = os.path.join(policy_dir, f'a2c_critic')
            # 保存 evaluate_net_pytorch 和 target_net_pytorch 的模型权重（state_dict）
            torch.save({'model_state_dict': policy['a2c_actor'].state_dict(),
                        'optimizer_state_dict': policy['a2c_actor_optimizer'].state_dict(), }, f'{actor_dir_py}.pth')
            torch.save({'model_state_dict': policy['a2c_critic'].state_dict(),
                        'optimizer_state_dict': policy['a2c_critic_optimizer'].state_dict(), },
                       f'{critic_dir_py}.pth')
            logger.info(f"保存-->a2c_actor+-->a2c_critic模型")
        elif "local_a3c_model" in policy:
            global_a3c_model_dir_py = os.path.join(policy_dir, f'global_a3c_model')
            torch.save({'model_state_dict': policy['global_a3c_model'].state_dict(),
                        'optimizer_state_dict': policy['global_optimizer'].state_dict(), },
                       f'{global_a3c_model_dir_py}.pth')
            logger.info(f"保存-->global_a3c_model模型")
        elif "ppo_actor" in policy:
            ppo_actor_dir_py = os.path.join(policy_dir, f'ppo_actor')
            ppo_critic_dir_py = os.path.join(policy_dir, f'ppo_critic')
            # 保存 evaluate_net_pytorch 和 target_net_pytorch 的模型权重（state_dict）
            torch.save({'model_state_dict': policy['ppo_actor'].state_dict(),
                        'optimizer_state_dict': policy['ppo_actor_optim'].state_dict(), }, f'{ppo_actor_dir_py}.pth')
            torch.save({'model_state_dict': policy['ppo_critic'].state_dict(),
                        'optimizer_state_dict': policy['ppo_critic_optim'].state_dict(), },
                       f'{ppo_critic_dir_py}.pth')
            logger.info(f"保存-->ppo_actor+-->ppo_critic模型")
        elif "sac_actor" in policy:
            sac_actor_dir_py = os.path.join(policy_dir, f'sac_actor')
            q0_net_dir_py = os.path.join(policy_dir, f'q0_net')
            q1_net_dir_py = os.path.join(policy_dir, f'q1_net')
            sac_critic_main_dir_py = os.path.join(policy_dir, f'sac_critic_main')
            sac_critic_target_dir_py = os.path.join(policy_dir, f'sac_critic_target')

            # 保存 evaluate_net_pytorch 和 target_net_pytorch 的模型权重（state_dict）
            torch.save({'model_state_dict': policy['sac_actor'].state_dict(),
                        'optimizer_state_dict': policy['sac_actor_optimizer'].state_dict(), },
                       f'{sac_actor_dir_py}.pth')
            torch.save({'model_state_dict': policy['q0_net'].state_dict(),
                        'optimizer_state_dict': policy['q0_net_optimizer'].state_dict(), },
                       f'{q0_net_dir_py}.pth')
            torch.save({'model_state_dict': policy['q1_net'].state_dict(),
                        'optimizer_state_dict': policy['q1_net_optimizer'].state_dict(), },
                       f'{q1_net_dir_py}.pth')
            torch.save({'model_state_dict': policy['sac_critic_main'].state_dict(),
                        'optimizer_state_dict': policy['sac_critic_main_optimizer'].state_dict(), },
                       f'{sac_critic_main_dir_py}.pth')
            torch.save({'model_state_dict': policy['sac_critic_target'].state_dict(),
                        'optimizer_state_dict': policy['sac_critic_target_optimizer'].state_dict(), },
                       f'{sac_critic_target_dir_py}.pth')

            logger.info(f"保存-->sac--模型")

    @staticmethod
    def load_policy(class_name, method_name):
        # 加载指定路径下的 CSV 文件
        policy_dir = os.path.join(Policy_loader.policy_dir, class_name)
        Policy_loader.save_dir = os.path.join(policy_dir, method_name)

        q_sa_loaded = np.loadtxt(f'{Policy_loader.save_dir}', delimiter=',')

        return q_sa_loaded

    @staticmethod
    def load_w_para(class_name, method_name):
        policy_dir = os.path.join(Policy_loader.policy_dir, class_name)
        Policy_loader.save_dir = os.path.join(policy_dir, method_name)

        with open(f"{Policy_loader.save_dir}", "rb") as f:
            data = pickle.load(f)
        print(f"模型已加载自 {Policy_loader.save_dir}")
        return data["weights"], data["encoder"]

    @staticmethod
    def load_dqn_network(dir):
        loaded_model = keras.models.load_model(filepath=f'{dir}.h5')
        print(f"模型已加载自 {Policy_loader.save_dir}")
        return loaded_model
