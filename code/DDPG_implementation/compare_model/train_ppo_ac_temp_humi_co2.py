import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import sys

sys.path.append("G:\我的雲端硬碟\paper\code\DDPG_implementation")
# 注册自定義環境
register(
    id='ac_temp_humi_co2',
    entry_point='Env.MyEnv:ac_temp_humi_co2'  # 環境路徑
)
register(
    id='win_temp_humi_co2',
    entry_point='Env.MyEnv:win_temp_humi_co2'  # 環境路徑
)
#紀錄tensorboard的各項數值
class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # 初始化變量來存儲每個episode的累計獎勵和長度
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        # 獲取環境的最後一個獎勵
        last_rewards = np.array(self.locals['rewards'])
        # 累加當前step的獎勵和長度
        self.current_episode_reward += np.array(self.locals['rewards'])[0] 
        self.current_episode_length += 1
        # 對每個環境的最後一個獎勵進行記錄
        if self.locals['dones'][0]:  # 單一環境
            # Episode结束，記錄累計獎勵和長度
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # 紀錄episode的資訊
            self.logger.record('episode_reward', self.current_episode_reward)
            self.logger.record('episode_length', self.current_episode_length)
            
            # 重置累計獎勵與長度
            self.current_episode_reward = 0
            self.current_episode_length = 0

        for idx, reward in enumerate(last_rewards):
            self.logger.record(f'step_reward/reward_{idx}', reward)
        return True

def linear_schedule(initial_value): #學習率遞減
    """
    返回一個函數，該函數計算給定步驟下的學習率
    """
    def func(progress_remaining):
        """
        progress_remaining: 從1（訓練開始）線性遞減到0（訓練結束）
        """
        return initial_value * progress_remaining

    return func

def train_model_ac(name,base_save_path):
    log_dir = os.path.join(base_save_path)
    replay_buffer_path = os.path.join(base_save_path, str(name+"_buffer.pkl"))
    model_dir=os.path.join(base_save_path, name)
    env_dir=os.path.join(base_save_path,str(name+"_env.pkl"))
    tb_log_name=name

    # 创建环境
    env = gym.make("ac_temp_humi_co2")
    env = DummyVecEnv([lambda: env])  # 向量化環境
    env = VecNormalize(
        env, 
        norm_obs=True, #觀測值標準化
        norm_reward=True#獎勵標準化
        )
    # env=NormalizeActionWrapper(env)

    custom_callback=CustomCallback()
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=linear_schedule(3e-4), 
        batch_size=1024, 
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        seed=26,#目前26最優
        clip_range=0.2,
        verbose=1,
        tensorboard_log=log_dir,  # 訓練結果
        device='cuda',  # 根據需要選擇'cuda'或'auto'

    )
    # 訓練模型
    model.learn(total_timesteps=100000, log_interval=1, tb_log_name=tb_log_name,callback=custom_callback)

    # # 保存模型
    env.save(env_dir)
    model.save(model_dir)
    print("PPO_ac_temp_humi_co2訓練完成")
    
if __name__ == "__main__":
    name="ppo_ac_temp_humi_co2_seed30" #指定儲存檔案名稱
    base_save_path = "G:\我的雲端硬碟\paper\code\DDPG_implementation\compare_model/ppo_train_result" #指定要儲存的資料夾
    train_model_ac(name,base_save_path)


