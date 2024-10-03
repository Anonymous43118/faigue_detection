import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO,DDPG,SAC,TD3,HerReplayBuffer
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise,NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, configure
from gymnasium.envs.registration import register
from gymnasium.wrappers import TransformObservation,TransformReward #獎勵歸一化工具TransformReward
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import pandas as pd
import sys

sys.path.append("G:\我的雲端硬碟\paper\code\DDPG_implementation")
# 注册自定義環境
register(
    id='ac_temp_humi_co2',
    entry_point='Env.MyEnv:ac_temp_humi_co2'  # 模型環境路徑
)

register(
    id='win_temp_humi_co2',
    entry_point='Env.MyEnv:win_temp_humi_co2'  # 模型環境路徑
)

def evaluate_model_ac(model, env_path, model_path, num_episodes, output_excel_path):
    """
    加載環境和模型，評估模型的表現，同時打印動作和環境狀態，並將數據保存到Excel文件。

    :param model: 使用的模型名稱 (ppo, sac, ddpg, td3)
    :param env_path: 保存的環境路徑
    :param model_path: 保存的模型路徑
    :param num_episodes: 要運行的回合數
    :param output_excel_path: 保存輸出的Excel文件路徑
    """

    # 加载环境
    env = gym.make("ac_temp_humi_co2")
    env = DummyVecEnv([lambda: env])  # 向量化環境
    env = VecNormalize.load(env_path, env)
    env.training = False  # 把環境設為評估模式
    env.norm_reward = True  # 不對獎勵進行normalization

    # 加载模型
    if model == "ppo":
        model = PPO.load(model_path, env=env)
    elif model == "sac":
        model = SAC.load(model_path, env=env)
    elif model == "ddpg":
        model = DDPG.load(model_path, env=env)
    elif model == "td3":
        model = TD3.load(model_path, env=env)

    all_data = []

    # 評估模型
    total_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]  # 使用索引來獲得單一動作獎勵值

            # 原始環境數值與獎勵值
            original_obs = env.get_original_obs()
            original_reward = env.get_original_reward()

            # 印出當前動作、狀態與獎勵
            print(f"动作: {action}, 状态: {original_obs}, 奖励: {reward}")

            # 保存數據
            step_data = {
                "episode": episode + 1,
                "action": action[0] if isinstance(action, (list, np.ndarray)) else action,
                "temperature": original_obs[0][0],
                "humidity": original_obs[0][1],
                "co2": original_obs[0][2],
                "reward": reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            }
            all_data.append(step_data)

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} total reward: {episode_reward}")

    # 計算平均獎勵
    average_reward = sum(total_rewards) / num_episodes
    print(f"Average reward over {num_episodes} episodes: {average_reward:.2f}")

    # 存到excel
    df = pd.DataFrame(all_data)
    df.to_excel(output_excel_path, index=False)

#主程式
if __name__ == "__main__":
    model = "ppo"
    env_dir = "G:\\我的雲端硬碟\\paper\\code\\DDPG_implementation\\compare_model\\ppo_train_result\\ppo_ac_temp_humi_co2_result\\ppo_ac_temp_humi_co2_env.pkl"
    model_dir = "G:\\我的雲端硬碟\\paper\\code\\DDPG_implementation\\compare_model\\ppo_train_result\\ppo_ac_temp_humi_co2_result\\ppo_ac_temp_humi_co2.zip"
    output_excel_path = "output.xlsx"
    evaluate_model_ac(model, env_dir, model_dir, num_episodes=10, output_excel_path=output_excel_path)
