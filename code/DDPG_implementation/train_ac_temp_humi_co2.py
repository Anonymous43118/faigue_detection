import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise,NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import pandas as pd

# 註冊自定義環境
register(
    id='ac_temp_humi_co2',
    entry_point='Env.MyEnv:ac_temp_humi_co2'  # 確保這裡是您的環境路徑
)
# 紀錄tensorboard的各項數值
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
        self.current_episode_reward += np.array(self.locals['rewards'])[0]  # 假設單環境
        self.current_episode_length += 1
        # 對每個環境的最後一個獎勵進行記錄
        if self.locals['dones'][0]:  # 假設單環境
            # Episode結束，記錄累計獎勵和長度
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # 在logger中記錄這個episode的信息
            self.logger.record('episode_reward', self.current_episode_reward)
            self.logger.record('episode_length', self.current_episode_length)
            
            # 重置累計獎勵和長度
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

def train_model(name,base_save_path):

    log_dir = os.path.join(base_save_path)
    replay_buffer_path = os.path.join(base_save_path, str(name+"_buffer.pkl"))
    model_dir=os.path.join(base_save_path, name)
    env_dir=os.path.join(base_save_path,str(name+"_env.pkl"))
    tb_log_name=name
    # 超參數
    policy="MlpPolicy"
    learning_rate = 1e-5 #嘗試學習率隨時間遞減或者設定得更小
    learning_starts=30000
    buffer_size = 2000000 #大一點比較好
    batch_size = 1024
    gamma = 0.99
    tau = 0.01
    seed=26
    gradient_steps=16 #可以設置為與train_freq相等或者略高
    train_freq=(8,'step')#降低更新頻率
    sigma = 0.2 #過大必然降低模型效能
    policy_kwargs={"net_arch": [64,32]}  

    # 创建环境
    env = gym.make("ac_temp_humi_co2")
    env = DummyVecEnv([lambda: env])  # 向量化環境
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=True
        )  
    # env=NormalizeActionWrapper(env)


    # 定義action noise
    n_actions = env.action_space.shape[-1]
    # action_noise=NormalActionNoise(mean=np.zeros(n_actions),sigma=sigma * np.ones(n_actions))
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=sigma * np.ones(n_actions))
    custom_callback=CustomCallback()

    # 创建DDPG模型
    model = DDPG(
        policy=policy,
        env=env,
        learning_rate=linear_schedule(learning_rate), 
        learning_starts=learning_starts,
        buffer_size=buffer_size,
        batch_size=batch_size, 
        tau=tau,
        gamma=gamma,
        seed=seed,
        action_noise=action_noise,
        gradient_steps=gradient_steps, 
        train_freq=train_freq, 
        verbose=1,
        tensorboard_log=log_dir,  # 日志目录
        device='cuda',  # 根据需要选择'cuda'或'auto'
        policy_kwargs=policy_kwargs
    )
    # 训练模型
    model.learn(total_timesteps=50000, log_interval=1, tb_log_name=tb_log_name,callback=custom_callback)

    # # 保存模型
    env.save(env_dir)
    model.save(model_dir)
    model.save_replay_buffer(replay_buffer_path)
    print("DDPG_ac_temp_humi_co2訓練完成")

def get_ac_temp_humi_co2_buffer(name,base_save_path): #取得前一百萬筆的replay buffer
    # 加載模型
    model = DDPG.load(os.path.join(base_save_path, name))

    # 訪問 replay buffer
    model.load_replay_buffer(os.path.join(base_save_path, str(name+"_buffer.pkl")))

    # 獲取 replay buffer 的大小
    buffer_size = model.replay_buffer.size()
    print("Replay buffer size:", buffer_size)

    # 確保有數據要被提取和保存
    if buffer_size > 0:
        # 設置最大記錄數
        max_records = min(1000000, buffer_size)

        # 提取數據
        observations = model.replay_buffer.observations[:max_records, 0, :] 
        actions = model.replay_buffer.actions[:max_records]
        rewards = model.replay_buffer.rewards[:max_records].squeeze()
        dones = model.replay_buffer.dones[:max_records].squeeze()
        print(actions.shape)
        # 處理觀測值
        temperature = observations[:, 0]  # 溫度是第一列
        humidity = observations[:, 1]  # 濕度是第二列
        co2 = observations[:, 2]  # 二氧化碳濃度是第三列

        
        # 分解動作數據為兩個獨立的特徵
        action_temperature = actions[:, 0, 0]
        action_window = actions[:, 0, 1]  

        # 創建 DataFrame
        data = {
            'temperature': temperature,
            'humidity': humidity,
            'co2': co2,
            'action_temperature': action_temperature,
            'action_air_flow': action_window,
            'rewards': rewards,
            'dones': dones
        }
        df = pd.DataFrame(data)

        # 檢查 DataFrame
        print(df.head())  # 打印 DataFrame 的前幾行以確認數據

        # 保存到 Excel 文件
        excel_path = os.path.join(base_save_path, str(name+"_buffer.xlsx"))
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)  # 寫入數據，不包括DataFrame的索引

            # 獲取工作表
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']  # 默認的工作表名稱是'Sheet1'

            # 凍結首行
            worksheet.freeze_panes = 'A2'  # 'A2'表示從A2開始向下和向右的單元格都是可滾動的，A1行被凍結
        print(f"Data saved to {excel_path}")
    else:
        print("Replay buffer is empty. Train your model before extracting the data.")
if __name__ == "__main__":
    name="ac_temp_humi_co2_50000"
    base_save_path = "G:\我的雲端硬碟\paper\code\DDPG_implementation\compare_model\ddpg_train_result"
    train_model(name,base_save_path)
    get_ac_temp_humi_co2_buffer(name,base_save_path)
