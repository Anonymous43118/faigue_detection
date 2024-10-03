from sklearn.preprocessing import MinMaxScaler
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
from DDPG_implementation.Env.humi_simu import Humidity_simulation
from DDPG_implementation.Env.temp_simu import TempSimulator
from DDPG_implementation.Env.CO2_simu import co2_simulator
from DDPG_implementation.Env.Rewardcalculate import RewardCalculator

def set_env_range(np_random, **kwargs): #設定所有環境中各個數值的範圍
    # 定義隨機數生成的範圍
    ranges = {
        "in_temperature": (5, 40),
        "in_humidity": (35, 85),
        "in_co2": (400, 1500),
        "out_temperature": (5, 40),
        "out_co2": (400, 800),
        "out_humidity": (30, 95)
    }
    
    # 使用np_random和範圍生成實際的初始化參數或使用傳入的指定參數
    environment_params = {
        "in_temperature": kwargs.get('custom_in_temperature', round(np_random.uniform(*ranges['in_temperature']), 1)),
        "in_humidity": kwargs.get('custom_in_humidity', round(np_random.uniform(*ranges['in_humidity']), 1)),
        "in_co2": kwargs.get('custom_in_co2', round(np_random.uniform(*ranges['in_co2']), 1)),
        "out_temperature": kwargs.get('custom_out_temperature', round(np_random.uniform(*ranges['out_temperature']), 1)),
        "out_co2": kwargs.get('custom_out_co2', round(np_random.uniform(*ranges['out_co2']), 1)),
        "out_humidity": kwargs.get('custom_out_humidity', round(np_random.uniform(*ranges['out_humidity']), 1))
    }
    
    return environment_params


class CustomCarEnvironment(gym.Env): #包含環境中4個數值
    def __init__(self):
        super(CustomCarEnvironment, self).__init__()
        # 定義環境空間
        self.observation_space = spaces.Box(low=np.array(
            [5, 35, 400, 0]), high=np.array([40, 90, 2000, 1]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array(
            [20, 0]), high=np.array([28, 1]), dtype=np.float32)  # 0為關窗、1為開窗
        self.reset()
        self.pre_state = None

    def reset(self, seed=None, option=None, **kwargs):
        # 固定種子碼
        super().reset(seed=seed)
        options = kwargs.pop('options', None)
        # 初始化室內環境參數
        self.in_temperature = round(self.np_random.uniform(5, 40), 1)
        self.in_humidity = round(self.np_random.uniform(35, 85), 1)
        self.in_co2 = round(self.np_random.uniform(400, 1500), 1)
        self.light = float(self.np_random.integers(0, 1))
        self.windows_open_rate = None  
        # 初始化室外環境參數
        self.out_temperature = round(self.np_random.uniform(5, 40), 1)
        self.out_co2 = round(self.np_random.uniform(400, 1500), 1)
        self.out_humidity = round(self.np_random.uniform(35, 85), 1)

        # 紀錄環境在特定step的狀態
        self.pre_state = np.array([self.in_temperature,
                                   self.in_humidity, self.in_co2])  # 記錄前一個step狀態
        self.first_state = np.array([self.in_temperature,
                                     self.in_humidity, self.in_co2])  # 記錄第一個step狀態

        if self.windows_open_rate == 0:
            self.out_temperature = self.in_temperature

        self.temp_simulator = TempSimulator(length=1, width=1, grid_size_x=20, grid_size_y=20,
                                            delta_t=10, init_temp=self.in_temperature, air_conditioner_temperature=26, out_temp=self.out_temperature)  # 溫度模擬

        self.humi_simulator = Humidity_simulation(in_humi=self.in_humidity, out_humi=self.out_humidity, win_h=50, win_w=60, car_length=100,
                                                  car_width=100, car_height=130, speed_per_hour=70, ave_temp=self.in_temperature)  # 濕度模擬

        self.CO2_simulator = co2_simulator(
            indoor_co2=self.in_co2, outdoor_co2=self.out_co2, ventilation_rate=self.humi_simulator.get_air_volume(windows_open_rate=1), V=self.humi_simulator.get_room_volume())  # CO2模擬

        # 初始化時間步數
        self.current_step = 0

        obs = np.array(
            [self.in_temperature, self.in_humidity, self.in_co2, self.light], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low,
                      self.observation_space.high)
        return (obs, {})

    def temp_reward(self):  # 溫度的獎勵計算
        temp_reward = 0
        temp_penalty = 0

        # 溫度在目標範圍內
        if 22 <= self.in_temperature <= 26:
            temp_reward += 13

        # 溫度不在目標範圍內
        else:
            # 計算溫度與目標範圍的差距
            if self.in_temperature < 22:
                temp_diff = 22 - self.in_temperature
            else:
                temp_diff = self.in_temperature - 26

            # 基本溫度懲罰
            temp_penalty += 1 * temp_diff

            # 如果有先前的狀態可以比較
            if self.pre_state is not None:
                temp_change = self.in_temperature - \
                    self.pre_state[0]  # 當前溫度與前一狀態的差異

                # 溫度往正確方向變化
                if (self.in_temperature < 22 and temp_change > 0) or (self.in_temperature > 26 and temp_change < 0):
                    temp_reward += 1  # 往正確方向前進的獎勵
                    # 根據變化速度調整獎勵，變化越快獎勵越多
                    temp_reward += abs(temp_change) * 5

                # 溫度往錯誤方向變化
                else:
                    temp_penalty += 1  # 往錯誤方向前進的懲罰
                    # 根據變化速度調整懲罰，變化越快懲罰越多
                    temp_penalty += abs(temp_change) * 5

        return 0.45*(temp_reward - temp_penalty)

    def humidity_reward(self):  # 濕度獎勵計算
        humidity_reward = 0
        humidity_penalty = 0

        # 濕度在目標範圍內
        if 40 <= self.in_humidity <= 60:
            humidity_reward += 23

        # 濕度不在目標範圍內
        else:
            # 計算濕度與目標範圍的差距
            if self.in_humidity < 40:
                humidity_diff = 40 - self.in_humidity
            else:
                humidity_diff = self.in_humidity - 60

            # 基本濕度懲罰
            humidity_penalty += 0.8 * humidity_diff

            # 如果有先前的狀態可以比較
            if self.pre_state is not None:
                humidity_change = self.in_humidity - \
                    self.pre_state[1]  # 當前濕度與前一狀態的差異

                # 濕度往正確方向變化
                if (self.in_humidity < 40 and humidity_change > 0) or (self.in_humidity > 60 and humidity_change < 0):
                    humidity_reward += 1  # 往正確方向前進的獎勵
                    # 根據變化速度調整獎勵，變化越快獎勵越多
                    humidity_reward += abs(humidity_change) * 10

                # 濕度往錯誤方向變化
                else:
                    humidity_penalty += 1  # 往錯誤方向前進的懲罰
                    # 根據變化速度調整懲罰，變化越快懲罰越多
                    humidity_penalty += abs(humidity_change) * 10
        return 0.1*(humidity_reward - humidity_penalty)

    def co2_reward(self):  # 二氧化碳獎勵計算
        co2_reward = 0
        co2_penalty = 0

        # 如果二氧化碳濃度小於1000 ppm
        if self.in_co2 < 1000:
            co2_reward += 10
        else:
            # 超過1000 ppm的懲罰
            co2_diff = self.in_co2 - 1000
            co2_penalty += 0.01 * co2_diff

            # 如果有先前的狀態可以比較
            if self.pre_state is not None:
                co2_change = self.in_co2 - self.pre_state[2]  # 當前CO2濃度與前一狀態的差異

                # 如果二氧化碳濃度往減少的方向變化
                if co2_change < 0:
                    # 根據變化速度調整獎勵，變化越快獎勵越多
                    co2_reward += abs(co2_change) * 1
                else:
                    # 如果二氧化碳濃度增加，則增加懲罰
                    co2_penalty += abs(co2_change) * 1

        return 0.5*(co2_reward - co2_penalty)


    def reward_compute(self, in_temp, in_humi, in_co2):  # 總獎勵計算
        # 總體獎勵
        # 獎勵可以考慮另外一點，如果環境狀態離目標狀態越近，則可以讓模型降低環境往目標值變化的速度，讓環境變化依舊往正確的方向前進，但速度減慢
        total_reward = self.temp_reward()+self.humidity_reward() + \
            self.co2_reward()
        return total_reward

    def check_if_done(self):  # done的判斷式
        done = False
        # 检查是否到达最大步数
        if self.current_step >= 2500:
            done = True
        # 检查环境是否达到最佳舒适区域
        elif 22 <= self.in_temperature <= 26 and 40 <= self.in_humidity <= 60 and self.in_co2 < 1000:
            done = True
        # # 到达一定时间后，如果环境没有改善，则提前结束
        # elif self.current_step >= 1250:
        #     if not self.is_improving():
        #         done = True
        return done

    def is_improving(self):
        # 检查环境是否在改善
        improving = True
        # 检查温度是否在向目标范围内改变
        if self.in_temperature < 22 and self.in_temperature <= self.first_state[0]:
            improving = False
        elif self.in_temperature > 26 and self.in_temperature >= self.first_state[0]:
            improving = False
        # 检查湿度是否在向目标范围内改变
        if self.in_humidity < 40 and self.in_humidity <= self.first_state[1]:
            improving = False
        elif self.in_humidity > 60 and self.in_humidity >= self.first_state[1]:
            improving = False
        # 检查CO2浓度是否在改善
        if self.in_co2 > 1000 and self.in_co2 >= self.first_state[2]:
            improving = False
        return improving

    def step(self, action):
        # 解析動作
        self.temp_simulator.air_conditioner_temperature, windows_open_rate = action

        # 更新環境狀態
        self.temp_simulator.simulate(duration=1)
        self.in_temperature = np.mean(self.temp_simulator.temperature_field)
        self.in_humidity = self.humi_simulator.humidity_simu(
            ave_temp=self.in_temperature, windows_open_rate=windows_open_rate)
        self.in_co2 = self.CO2_simulator.simulate_step(windows_open_rate)

        # 計算即時獎勵（這裡可以根據特定的目標定義獎勵函數）
        total_reward = self.reward_compute(
            action[0], self.in_humidity, self.in_co2)

        self.pre_state = np.array([self.in_temperature,
                                  self.in_humidity, self.in_co2])  # 紀錄當前狀態

        # 檢查是否到達目標狀態（這裡可以根據具體目標定義）
        done = self.check_if_done()

        self.current_step += 1
        info = {}
        obs = np.array(
            [self.in_temperature, self.in_humidity, self.in_co2, self.light], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low,
                      self.observation_space.high)
        # 返回新的環境狀態、即時獎勵、結束標記等信息
        return obs, total_reward, done, False, info

    def test_reset(self, seed=None,  # 連接tk.py來用UI模擬用的，可以自定義環境狀態和動作
                   custom_in_temperature=None,
                   custom_in_humidity=None,
                   custom_in_co2=None,
                   custom_light=None,
                   custom_out_temperature=None,
                   custom_out_co2=None,
                   custom_out_humidity=None):

        # 若有指定種子碼，則固定種子碼
        if seed is not None:
            super().reset(seed=seed)

        # 初始化室內環境參數，若未提供自定義值則使用預設範圍隨機產生
        self.in_temperature = custom_in_temperature if custom_in_temperature is not None else round(
            self.np_random.uniform(10, 40), 1)
        self.in_humidity = custom_in_humidity if custom_in_humidity is not None else round(
            self.np_random.uniform(40, 82), 1)
        self.in_co2 = custom_in_co2 if custom_in_co2 is not None else round(
            self.np_random.uniform(350, 1500), 1)
        self.light = custom_light if custom_light is not None else float(
            self.np_random.integers(0, 1))
        self.windows_open_rate = 0

        # 初始化室外環境參數，若未提供自定義值則使用預設範圍隨機產生
        self.out_temperature = custom_out_temperature if custom_out_temperature is not None else round(
            self.np_random.uniform(10, 40), 1)
        self.out_co2 = custom_out_co2 if custom_out_co2 is not None else round(
            self.np_random.uniform(350, 1500), 1)
        self.out_humidity = custom_out_humidity if custom_out_humidity is not None else round(
            self.np_random.uniform(40, 82), 1)
        # 紀錄環境在特定step的狀態
        self.pre_state = np.array([self.in_temperature,
                                   self.in_humidity, self.in_co2])  # 記錄前一個step狀態
        self.first_state = np.array([self.in_temperature,
                                    self.in_humidity, self.in_co2])  # 記錄第一個step狀態

        if self.windows_open_rate == 0:
            self.out_temperature = self.in_temperature  # 如果沒開窗，則冷氣溫度同室溫

        self.temp_simulator = TempSimulator(length=1, width=1, grid_size_x=20, grid_size_y=20,
                                            delta_t=10, init_temp=self.in_temperature, air_conditioner_temperature=26, out_temp=self.out_temperature)  # 溫度模擬

        self.humi_simulator = Humidity_simulation(in_humi=self.in_humidity, out_humi=self.out_humidity, win_h=50, win_w=60, car_length=100,
                                                  car_width=100, car_height=130, speed_per_hour=70, ave_temp=self.in_temperature)  # 濕度模擬

        self.CO2_simulator = co2_simulator(
            indoor_co2=self.in_co2, outdoor_co2=self.out_co2, ventilation_rate=self.humi_simulator.get_air_volume(windows_open_rate=1), V=self.humi_simulator.get_room_volume())  # CO2模擬

        # 初始化時間步數
        self.current_step = 0  # 假設開車一個半小時

        obs = np.array(
            [self.in_temperature, self.in_humidity, self.in_co2, self.light], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low,
                      self.observation_space.high)
        return (obs, {})


class win_temp_humi_co2(gym.Env): #車窗、溫濕度、二氧化碳濃度
    def __init__(self):
        super(win_temp_humi_co2, self).__init__()
        # 定義環境空間
        self.observation_space = spaces.Box(low=np.array(
            [5, 35, 400]), high=np.array([40, 90, 1500]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array(
            [-1, -1]), high=np.array([1, 1]), dtype=np.float32)  # 0為關窗、1為開窗
        self.reset()
        self.pre_state = None
        # 定义和拟合scaler

        self.scaler_temp = MinMaxScaler(feature_range=(20, 28))
        self.scaler_window = MinMaxScaler(feature_range=(0, 1))
        possible_actions_temp = np.array([[-1], [1]])
        possible_actions_window = np.array([[-1], [1]])
        self.scaler_temp.fit(possible_actions_temp)
        self.scaler_window.fit(possible_actions_window)

    def reset(self, **kwargs):  # seed=None, option=None,
        # 固定種子碼
        seed = kwargs.get('seed', None)
        super().reset(seed=seed)
        options = kwargs.pop('options', None)
        # 初始化室內環境參數
        np_random = self.np_random
        # 調用env_reset並獲取環境參數
        env_params = set_env_range(np_random, **kwargs)
        
        # 初始化室內環境參數
        self.in_temperature = env_params["in_temperature"]
        self.in_humidity = env_params["in_humidity"]
        self.in_co2 = env_params["in_co2"]
        self.out_temperature = env_params["out_temperature"]
        self.out_co2 = env_params["out_co2"]
        self.out_humidity = env_params["out_humidity"]

        self.windows_open_rate = None  # 暫定設置為None，原先為0
        # 紀錄環境在特定step的狀態
        self.pre_state = np.array([self.in_temperature,
                                   self.in_humidity, self.in_co2])  # 記錄前一個step狀態
        self.first_state = np.array([self.in_temperature,
                                     self.in_humidity, self.in_co2])  # 記錄第一個step狀態

        if self.windows_open_rate == 0:
            self.out_temperature = self.in_temperature

        self.temp_simulator = TempSimulator(length=1, width=1, grid_size_x=30, grid_size_y=30,
                                            delta_t=5, init_temp=self.in_temperature, air_conditioner_temperature=26, out_temp=self.out_temperature)  # 溫度模擬

        self.humi_simulator = Humidity_simulation(in_humi=self.in_humidity, out_humi=self.out_humidity, win_h=50, win_w=60, car_length=100,
                                                  car_width=100, car_height=130, speed_per_hour=70, ave_temp=self.in_temperature)  # 濕度模擬

        self.CO2_simulator = co2_simulator(
            indoor_co2=self.in_co2, outdoor_co2=self.out_co2, ventilation_rate=self.humi_simulator.get_air_volume(windows_open_rate=1), V=self.humi_simulator.get_room_volume())  # CO2模擬

        # 初始化時間步數
        self.current_step = 0

        obs = np.array(
            [self.in_temperature, self.in_humidity, self.in_co2], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low,
                      self.observation_space.high)
        return (obs, {})

    def temp_reward(self):  # 溫度的獎勵計算
        temp_reward = 0
        temp_penalty = 0
        comfortable_temp_min = 22
        comfortable_temp_max = 26
        # 當溫度在舒適範圍內時，給予正獎勵；否則，給予懲罰
        if comfortable_temp_min <= self.in_temperature <= comfortable_temp_max:
            # 在舒適範圍內，可以給予一個固定的正獎勵，或者根據溫度距離舒適區間邊緣的遠近給予不同的獎勵
            temp_reward += 20
        else:
            # 溫度離開舒適區間時，根據距離給予懲罰
            temp_diff = min(abs(self.in_temperature - comfortable_temp_min),
                            abs(self.in_temperature - comfortable_temp_max))
            temp_penalty += temp_diff

            # 判斷溫度是否向正確方向變化
            # 前一個狀態比當前狀態更接近舒適區間邊緣，則當前動作是正向的，應給予額外獎勵
            if self.pre_state is not None:
                prev_temp_diff = min(abs(
                    self.pre_state[0] - comfortable_temp_min), abs(self.pre_state[0] - comfortable_temp_max))
                if temp_diff < prev_temp_diff:  # 如果當前溫度比之前更接近舒適區間
                    temp_reward += 1  # 鼓勵向舒適區間靠近的行為
                elif temp_diff > prev_temp_diff:  # 如果當前溫度比之前更遠離舒適區間
                    temp_penalty += 1  # 懲罰遠離舒適區間的行為

        # 綜合獎勵
        return temp_reward - temp_penalty

    def humidity_reward(self):  # 濕度獎勵計算
        humidity_reward = 0
        humidity_penalty = 0

        # 濕度在目標範圍內
        if 40 <= self.in_humidity <= 60:
            humidity_reward += 20

        # 濕度不在目標範圍內
        else:
            # 計算濕度與目標範圍的差距
            if self.in_humidity < 40:
                humidity_diff = 40 - self.in_humidity
            else:
                humidity_diff = self.in_humidity - 60

            # 基本濕度懲罰
            humidity_penalty += 0.8 * humidity_diff

        #     # 如果有先前的狀態可以比較
            if self.pre_state is not None:
                humidity_change = self.in_humidity - \
                    self.pre_state[1]  # 當前濕度與前一狀態的差異

                # 濕度往正確方向變化
                if (self.in_humidity < 40 and humidity_change > 0) or (self.in_humidity > 60 and humidity_change < 0):
                    humidity_reward += 1  # 往正確方向前進的獎勵
                    # 根據變化速度調整獎勵，變化越快獎勵越多
                    # humidity_reward += abs(humidity_change) * 10

                # 濕度往錯誤方向變化
                else:
                    humidity_penalty += 1  # 往錯誤方向前進的懲罰
                    # 根據變化速度調整懲罰，變化越快懲罰越多
                    # humidity_penalty += abs(humidity_change) * 10
        return 0.5*(humidity_reward - humidity_penalty)

    def co2_reward(self):  # 二氧化碳獎勵計算
        co2_reward = 0
        co2_penalty = 0

        # 如果二氧化碳濃度小於1000 ppm
        if self.in_co2 < 700:
            co2_reward += 20
        elif 700 <= self.in_co2 <= 999:
            co2_reward += 15
        else:
            # 超過1000 ppm的懲罰
            # co2_diff = self.in_co2 - 1000
            # co2_penalty += 0.01 * co2_diff

            # 如果有先前的狀態可以比較
            if self.pre_state is not None:
                co2_change = self.in_co2 - self.pre_state[2]  # 當前CO2濃度與前一狀態的差異

                # # 如果二氧化碳濃度往減少的方向變化
                # if co2_change < 0:
                #     # 根據變化速度調整獎勵，變化越快獎勵越多
                #     co2_reward += abs(co2_change) * 1
                # else:
                #     # 如果二氧化碳濃度增加，則增加懲罰
                #     co2_penalty += abs(co2_change) * 1
                if self.in_co2 > 600 and co2_change < 0:
                    # 根據變化速度調整獎勵，變化越快獎勵越多
                    co2_reward += 1
                elif self.in_co2 > 1000 and co2_change > 0:
                    # 如果二氧化碳濃度增加，則增加懲罰
                    co2_penalty += 1

        return co2_reward - co2_penalty

    def reward_compute(self):  # 總獎勵計算
        # 總體獎勵
        # 獎勵可以考慮另外一點，如果環境狀態離目標狀態越近，則可以讓模型降低環境往目標值變化的速度，讓環境變化依舊往正確的方向前進，但速度減慢
        total_reward = self.temp_reward()+self.humidity_reward() + \
            self.co2_reward()

        return total_reward

    def check_if_done(self):  # done的判斷式
        done = False
        # 检查是否到最大步數
        if self.current_step >= 3000:
            done = True
        return done

    def step(self, action):
        # 解析動作
        scaled_temp = self.scaler_temp.transform([[action[0]]])[0][0]
        scaled_window = self.scaler_window.transform([[action[1]]])[0][0]
        self.temp_simulator.air_conditioner_temperature = scaled_temp
        windows_open_rate = scaled_window
        # 更新環境狀態
        self.temp_simulator.simulate(
            duration=1, windows_open_rate=scaled_window)
        self.in_temperature = np.mean(
            self.temp_simulator.temperature_field[-10:, :10])
        self.in_humidity = self.humi_simulator.humidity_simu(
            ave_temp=self.in_temperature, windows_open_rate=windows_open_rate)
        self.in_co2 = self.CO2_simulator.simulate_step(windows_open_rate)

        # 計算即時獎勵（這裡可以根據特定的目標定義獎勵函數）
        total_reward = self.reward_compute()

        self.pre_state = np.array([self.in_temperature,
                                  self.in_humidity, self.in_co2])  # 紀錄當前狀態

        # 檢查是否到達目標狀態（這裡可以根據具體目標定義）
        done = self.check_if_done()

        self.current_step += 1
        info = {}
        obs = np.array(
            [self.in_temperature, self.in_humidity, self.in_co2], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low,
                      self.observation_space.high)
        # 返回新的環境狀態、即時獎勵、結束標記等信息
        return obs, total_reward, done, False, info


class win_temp_humi(gym.Env):
    def __init__(self):
        super(win_temp_humi, self).__init__()
        # 定義環境空間
        self.observation_space = spaces.Box(low=np.array(
            [5, 35]), high=np.array([40, 90]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array(
            [-1, -1]), high=np.array([1, 1]), dtype=np.float32)  # 0為關窗、1為開窗
        self.reset()
        self.pre_state = None

        # 定義和擬合scaler
        self.scaler_temp = MinMaxScaler(feature_range=(20, 28))
        self.scaler_window = MinMaxScaler(feature_range=(0, 1))
        possible_actions_temp = np.array([[-1], [1]])
        possible_actions_window = np.array([[-1], [1]])
        self.scaler_temp.fit(possible_actions_temp)
        self.scaler_window.fit(possible_actions_window)

    def reset(self, **kwargs):  # seed=None, option=None,
        # 固定種子碼
        seed = kwargs.get('seed', None)
        super().reset(seed=seed)
        options = kwargs.pop('options', None)
        # 初始化室內環境參數
        np_random = self.np_random
        # 調用env_reset並獲取環境參數
        env_params = set_env_range(np_random, **kwargs)
        
        # 初始化室內環境參數
        self.in_temperature = env_params["in_temperature"]
        self.in_humidity = env_params["in_humidity"]
        self.out_temperature = env_params["out_temperature"]
        self.out_humidity = env_params["out_humidity"]
        self.windows_open_rate = None  # 暫定設置為None，原先為0
        # 紀錄環境在特定step的狀態
        self.pre_state = np.array([self.in_temperature,
                                   self.in_humidity])  # 記錄前一個step狀態
        self.first_state = np.array([self.in_temperature,
                                     self.in_humidity])  # 記錄第一個step狀態

        if self.windows_open_rate == 0:
            self.out_temperature = self.in_temperature

        self.temp_simulator = TempSimulator(length=1, width=1, grid_size_x=30, grid_size_y=30,
                                            delta_t=5, init_temp=self.in_temperature, air_conditioner_temperature=26, out_temp=self.out_temperature)  # 溫度模擬

        self.humi_simulator = Humidity_simulation(in_humi=self.in_humidity, out_humi=self.out_humidity, win_h=50, win_w=60, car_length=100,
                                                  car_width=100, car_height=130, speed_per_hour=70, ave_temp=self.in_temperature)  # 濕度模擬

        # 初始化時間步數
        self.current_step = 0  # 假設開車一個半小時

        obs = np.array(
            [self.in_temperature, self.in_humidity], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low,
                      self.observation_space.high)
        return (obs, {})

    def temp_reward(self):  # 溫度的獎勵計算
        temp_reward = 0
        temp_penalty = 0
        comfortable_temp_min = 22
        comfortable_temp_max = 26
        # 當溫度在舒適範圍內時，給予正獎勵；否則，給予懲罰
        if comfortable_temp_min <= self.in_temperature <= comfortable_temp_max:
            # 在舒適範圍內，可以給予一個固定的正獎勵，或者根據溫度距離舒適區間邊緣的遠近給予不同的獎勵
            temp_reward += 20
        else:
            # 溫度離開舒適區間時，根據距離給予懲罰
            temp_diff = min(abs(self.in_temperature - comfortable_temp_min),
                            abs(self.in_temperature - comfortable_temp_max))
            temp_penalty += temp_diff

            # 判斷溫度是否向正確方向變化
            # 前一個狀態比當前狀態更接近舒適區間邊緣，則當前動作是正向的，應給予額外獎勵
            if self.pre_state is not None:
                prev_temp_diff = min(abs(
                    self.pre_state[0] - comfortable_temp_min), abs(self.pre_state[0] - comfortable_temp_max))
                if temp_diff < prev_temp_diff:  # 如果當前溫度比之前更接近舒適區間
                    temp_reward += 1  # 鼓勵向舒適區間靠近的行為
                elif temp_diff > prev_temp_diff:  # 如果當前溫度比之前更遠離舒適區間
                    temp_penalty += 1  # 懲罰遠離舒適區間的行為

        # 綜合獎勵
        return temp_reward - temp_penalty


    def humidity_reward(self):  # 濕度獎勵計算
        humidity_reward = 0
        humidity_penalty = 0

        # 濕度在目標範圍內
        if 40 <= self.in_humidity <= 60:
            humidity_reward += 20

        # 濕度不在目標範圍內
        else:
            # 計算濕度與目標範圍的差距
            if self.in_humidity < 40:
                humidity_diff = 40 - self.in_humidity
            else:
                humidity_diff = self.in_humidity - 60

            # 基本濕度懲罰
            humidity_penalty += 0.8 * humidity_diff

        #     # 如果有先前的狀態可以比較
            if self.pre_state is not None:
                humidity_change = self.in_humidity - \
                    self.pre_state[1]  # 當前濕度與前一狀態的差異

                # 濕度往正確方向變化
                if (self.in_humidity < 40 and humidity_change > 0) or (self.in_humidity > 60 and humidity_change < 0):
                    humidity_reward += 1  # 往正確方向前進的獎勵
                    # 根據變化速度調整獎勵，變化越快獎勵越多
                    # humidity_reward += abs(humidity_change) * 10

                # 濕度往錯誤方向變化
                else:
                    humidity_penalty += 1  # 往錯誤方向前進的懲罰
                    # 根據變化速度調整懲罰，變化越快懲罰越多
                    # humidity_penalty += abs(humidity_change) * 10
        return humidity_reward - humidity_penalty

    def reward_compute(self):  # 總獎勵計算
        # 總體獎勵
        # 獎勵可以考慮另外一點，如果環境狀態離目標狀態越近，則可以讓模型降低環境往目標值變化的速度，讓環境變化依舊往正確的方向前進，但速度減慢
        total_reward = self.temp_reward()+self.humidity_reward()
        # b = RewardCalculator(
        #     pre_state=self.pre_state, in_temp=self.in_temperature, in_humi=self.in_humidity)
        # result = b.reward_compute()
        # print(f"預設環境數值:{self.in_temperature}, {self.in_humidity}")
        # print(f"預設獎勵:{total_reward}")
        # print(f"外部獎勵:{result}")
        return total_reward

    def check_if_done(self):  # done的判斷式
        done = False
        # 检查是否到达最大步数
        if self.current_step >= 3000:
            done = True
        return done

    def step(self, action):
        # 解析动作
        scaled_temp = self.scaler_temp.transform([[action[0]]])[0][0]
        scaled_window = self.scaler_window.transform([[action[1]]])[0][0]
        self.temp_simulator.air_conditioner_temperature = scaled_temp
        windows_open_rate = scaled_window
        # 更新環境狀態
        self.temp_simulator.simulate(
            duration=1, windows_open_rate=scaled_window)
        self.in_temperature = np.mean(
            self.temp_simulator.temperature_field[-10:, :10])
        self.in_humidity = self.humi_simulator.humidity_simu(
            ave_temp=self.in_temperature, windows_open_rate=windows_open_rate)

        # 計算即時獎勵（這裡可以根據特定的目標定義獎勵函數）
        total_reward = self.reward_compute()

        self.pre_state = np.array(
            [self.in_temperature, self.in_humidity])  # 紀錄當前狀態

        # 檢查是否到達目標狀態（這裡可以根據具體目標定義）
        done = self.check_if_done()

        self.current_step += 1
        info = {}
        obs = np.array(
            [self.in_temperature, self.in_humidity], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low,
                      self.observation_space.high)
        # 返回新的環境狀態、即時獎勵、結束標記等信息
        return obs, total_reward, done, False, info


class win_temp_co2(gym.Env):
    def __init__(self):
        super(win_temp_co2, self).__init__()
        # 定義環境空間
        self.observation_space = spaces.Box(low=np.array(
            [5, 400]), high=np.array([40, 800]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array(
            [-1, -1]), high=np.array([1, 1]), dtype=np.float32)  # 0為關窗、1為開窗
        self.reset()
        self.pre_state = None

        # 定义和拟合scaler

        self.scaler_temp = MinMaxScaler(feature_range=(20, 28))
        self.scaler_window = MinMaxScaler(feature_range=(0, 1))
        possible_actions_temp = np.array([[-1], [1]])
        possible_actions_window = np.array([[-1], [1]])
        self.scaler_temp.fit(possible_actions_temp)
        self.scaler_window.fit(possible_actions_window)

    def reset(self, **kwargs):  # seed=None, option=None,
        # 固定種子碼
        seed = kwargs.get('seed', None)
        super().reset(seed=seed)
        options = kwargs.pop('options', None)
        # 初始化室內環境參數
        np_random = self.np_random
        # 调用env_reset并获取环境参数
        env_params = set_env_range(np_random, **kwargs)
        
        # 初始化室內環境參數
        self.in_temperature = env_params["in_temperature"]
        self.in_humidity = env_params["in_humidity"]
        self.in_co2 = env_params["in_co2"]
        self.out_temperature = env_params["out_temperature"]
        self.out_co2 = env_params["out_co2"]
        self.out_humidity = env_params["out_humidity"]
        # self.in_temperature = kwargs.get(
        #     'custom_in_temperature', round(self.np_random.uniform(5, 40), 1))
        # self.in_humidity = kwargs.get(
        #     'custom_in_humidity', round(self.np_random.uniform(35, 85), 1))
        # self.in_co2 = kwargs.get('custom_in_co2', round(
        #     self.np_random.uniform(400, 1500), 1))
        # # 初始化室外環境參數
        # self.out_temperature = kwargs.get(
        #     'custom_out_temperature', round(self.np_random.uniform(5, 40), 1))
        # self.out_co2 = kwargs.get('custom_out_co2', round(
        #     self.np_random.uniform(400, 700), 1))
        # self.out_humidity = kwargs.get(
        #     'custom_out_humidity', round(self.np_random.uniform(35, 85), 1))
        self.windows_open_rate = None  # 暫定設置為None，原先為0
        # 紀錄環境在特定step的狀態
        self.pre_state = np.array(
            [self.in_temperature, self.in_co2])  # 記錄前一個step狀態
        self.first_state = np.array(
            [self.in_temperature, self.in_co2])  # 記錄第一個step狀態

        if self.windows_open_rate == 0:
            self.out_temperature = self.in_temperature

        self.temp_simulator = TempSimulator(length=1, width=1, grid_size_x=30, grid_size_y=30,
                                            delta_t=5, init_temp=self.in_temperature, air_conditioner_temperature=26, out_temp=self.out_temperature)  # 溫度模擬

        self.humi_simulator = Humidity_simulation(in_humi=self.in_humidity, out_humi=self.out_humidity, win_h=50, win_w=60, car_length=100,
                                                  car_width=100, car_height=130, speed_per_hour=70, ave_temp=self.in_temperature)  # 濕度模擬

        self.CO2_simulator = co2_simulator(
            indoor_co2=self.in_co2, outdoor_co2=self.out_co2, ventilation_rate=self.humi_simulator.get_air_volume(windows_open_rate=1), V=self.humi_simulator.get_room_volume())  # CO2模擬

        # 初始化時間步數
        self.current_step = 0  # 假設開車一個半小時

        obs = np.array(
            [self.in_temperature, self.in_co2], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low,
                      self.observation_space.high)
        return (obs, {})

    def temp_reward(self):  # 溫度的獎勵計算
        temp_reward = 0
        temp_penalty = 0
        comfortable_temp_min = 22
        comfortable_temp_max = 26
        # 当温度在舒适范围内时，给予正奖励；否则，给予惩罚
        if comfortable_temp_min <= self.in_temperature <= comfortable_temp_max:
            # 在舒适范围内，可以给予一个固定的正奖励，或者根据温度距离舒适区间边缘的远近给予不同的奖励
            temp_reward += 20
        else:
            # 温度离开舒适区间时，根据距离给予惩罚
            temp_diff = min(abs(self.in_temperature - comfortable_temp_min),
                            abs(self.in_temperature - comfortable_temp_max))
            temp_penalty += temp_diff

            # 判断温度是否向正确方向变化
            # 前一个状态比当前状态更接近舒适区间边缘，则当前动作是正向的，应给予额外奖励
            if self.pre_state is not None:
                prev_temp_diff = min(abs(
                    self.pre_state[0] - comfortable_temp_min), abs(self.pre_state[0] - comfortable_temp_max))
                if temp_diff < prev_temp_diff:  # 如果当前温度比之前更接近舒适区间
                    temp_reward += 1  # 鼓励向舒适区间靠近的行为
                elif temp_diff > prev_temp_diff:  # 如果当前温度比之前更远离舒适区间
                    temp_penalty += 1  # 惩罚远离舒适区间的行为

        # 综合奖励
        return temp_reward - temp_penalty

    def co2_reward(self):  # 二氧化碳獎勵計算
        co2_reward = 0
        co2_penalty = 0

        # 如果二氧化碳浓度小于1000 ppm
        if self.in_co2 < 700:
            co2_reward += 20
        elif 700 <= self.in_co2 <= 999:
            co2_reward += 15
        else:
            # 超过1000 ppm的惩罚
            # co2_diff = self.in_co2 - 1000
            # co2_penalty += 0.01 * co2_diff

            # 如果有先前的状态可以比较
            if self.pre_state is not None:
                co2_change = self.in_co2 - self.pre_state[1]  # 当前CO2浓度与前一状态的差异

                # # 如果二氧化碳浓度往减少的方向变化
                # if co2_change < 0:
                #     # 根据变化速度调整奖励，变化越快奖励越多
                #     co2_reward += abs(co2_change) * 1
                # else:
                #     # 如果二氧化碳浓度增加，则增加惩罚
                #     co2_penalty += abs(co2_change) * 1
                if self.in_co2 > 600 and co2_change < 0:
                    # 根据变化速度调整奖励，变化越快奖励越多
                    co2_reward += 1
                elif self.in_co2 > 1000 and co2_change > 0:
                    # 如果二氧化碳浓度增加，则增加惩罚
                    co2_penalty += 1

        return co2_reward - co2_penalty

    def reward_compute(self):  # 總獎勵計算
        # 總體獎勵
        # 獎勵可以考慮另外一點，如果環境狀態離目標狀態越近，則可以讓模型降低環境往目標值變化的速度，讓環境變化依舊往正確的方向前進，但速度減慢
        # total_reward = self.temp_reward()+self.co2_reward()
        reward_calculator = RewardCalculator(
            pre_state=self.pre_state, in_temp=self.in_temperature, in_co2=self.in_co2)
        result = reward_calculator.reward_compute()
        # print(
        #     f"預設環境數值:{self.in_temperature}, {self.in_humidity},{self.in_co2}")
        # print(f"預設獎勵:{total_reward}")
        # print(f"外部獎勵:{result}")
        return result

    def check_if_done(self):  # done的判斷式
        done = False
        # 检查是否到达最大步数
        if self.current_step >= 3000:
            done = True
        return done

    def step(self, action):
        # 解析动作
        scaled_temp = self.scaler_temp.transform([[action[0]]])[0][0]
        scaled_window = self.scaler_window.transform([[action[1]]])[0][0]
        self.temp_simulator.air_conditioner_temperature = scaled_temp
        windows_open_rate = scaled_window
        # 更新環境狀態
        self.temp_simulator.simulate(
            duration=1, windows_open_rate=scaled_window)
        self.temp_simulator.update_temperature_field(
            windows_open_rate=scaled_window)
        self.in_temperature = np.mean(
            self.temp_simulator.temperature_field[-10:, :10])
        self.in_co2 = self.CO2_simulator.simulate_step(windows_open_rate)

        # 計算即時獎勵（這裡可以根據特定的目標定義獎勵函數）
        total_reward = self.reward_compute()

        self.pre_state = np.array([self.in_temperature, self.in_co2])  # 紀錄當前狀態

        # 檢查是否到達目標狀態（這裡可以根據具體目標定義）
        done = self.check_if_done()

        self.current_step += 1
        info = {}
        obs = np.array(
            [self.in_temperature, self.in_co2], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low,
                      self.observation_space.high)
        # 返回新的環境狀態、即時獎勵、結束標記等信息
        return obs, total_reward, done, False, info


class ac_temp_humi(gym.Env):  # 冷氣溫度和冷氣風量+溫度和濕度
    def __init__(self):
        super(ac_temp_humi, self).__init__()
        # 定義環境空間
        self.observation_space = spaces.Box(low=np.array(
            [5, 35]), high=np.array([40, 90]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array(
            [-1, -1]), high=np.array([1, 1]), dtype=np.float32)  # 0為關窗、1為開窗
        self.reset()
        self.pre_state = None

        # 縮放動作數值
        self.scaler_temp = MinMaxScaler(feature_range=(20, 28))  # 冷氣溫度20-28度
        possible_actions_temp = np.array([[-1], [1]])
        self.scaler_temp.fit(possible_actions_temp)

        self.ac_air_flow = MinMaxScaler(
            feature_range=(1.01, 2.6))  # 冷氣風量1.01-2.6
        possible_actions_air_flow = np.array([[-1], [1]])
        self.ac_air_flow.fit(possible_actions_air_flow)

        self.ac_air_flow_for_humi = MinMaxScaler(
            feature_range=(10, 500))  # 冷氣風量縮放到10-50給濕度用
        actions_air_flow_for_humi = np.array([[1.01], [2.6]])
        self.ac_air_flow_for_humi.fit(actions_air_flow_for_humi)

    def reset(self, **kwargs):  # seed=None, option=None,
        # 固定種子碼
        seed = kwargs.get('seed', None)
        super().reset(seed=seed)
        options = kwargs.pop('options', None)
        # 初始化室內環境參數
        np_random = self.np_random
        # 调用env_reset并获取环境参数
        env_params = set_env_range(np_random, **kwargs)
        
        # 初始化室內環境參數
        self.in_temperature = env_params["in_temperature"]
        self.in_humidity = env_params["in_humidity"]
        self.out_temperature = env_params["out_temperature"]
        self.out_humidity = env_params["out_humidity"]
        # self.in_temperature = kwargs.get(
        #     'custom_in_temperature', round(self.np_random.uniform(5, 40), 1))
        # self.in_humidity = kwargs.get(
        #     'custom_in_humidity', round(self.np_random.uniform(35, 85), 1))

        # # 初始化室外環境參數
        # self.out_temperature = kwargs.get(
        #     'custom_out_temperature', round(self.np_random.uniform(5, 40), 1))
        # self.out_humidity = kwargs.get(
        #     'custom_out_humidity', round(self.np_random.uniform(35, 85), 1))

        # 紀錄環境在特定step的狀態
        self.pre_state = np.array([self.in_temperature,
                                   self.in_humidity])  # 記錄前一個step狀態
        self.first_state = np.array([self.in_temperature,
                                     self.in_humidity])  # 記錄第一個step狀態

        self.temp_simulator = TempSimulator(length=1, width=1, grid_size_x=30, grid_size_y=30,
                                            delta_t=5, init_temp=self.in_temperature, air_conditioner_temperature=26, ac_level=1.01)  # 溫度模擬

        self.humi_simulator = Humidity_simulation(in_humi=self.in_humidity, out_humi=self.out_humidity, win_h=50, win_w=60, car_length=100,
                                                  car_width=100, car_height=130, speed_per_hour=70, ave_temp=self.in_temperature)  # 濕度模擬
        # 初始化時間步數
        self.current_step = 0  # 假設開車一個半小時

        obs = np.array(
            [self.in_temperature, self.in_humidity], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low,
                      self.observation_space.high)
        return (obs, {})

    def temp_reward(self):  # 溫度的獎勵計算
        temp_reward = 0
        temp_penalty = 0
        comfortable_temp_min = 22
        comfortable_temp_max = 26
        # 当温度在舒适范围内时，给予正奖励；否则，给予惩罚
        if comfortable_temp_min <= self.in_temperature <= comfortable_temp_max:
            # 在舒适范围内，可以给予一个固定的正奖励，或者根据温度距离舒适区间边缘的远近给予不同的奖励
            temp_reward += 20
        else:
            # 温度离开舒适区间时，根据距离给予惩罚
            temp_diff = min(abs(self.in_temperature - comfortable_temp_min),
                            abs(self.in_temperature - comfortable_temp_max))
            temp_penalty += temp_diff

            # 判断温度是否向正确方向变化
            # 前一个状态比当前状态更接近舒适区间边缘，则当前动作是正向的，应给予额外奖励
            if self.pre_state is not None:
                prev_temp_diff = min(abs(
                    self.pre_state[0] - comfortable_temp_min), abs(self.pre_state[0] - comfortable_temp_max))
                if temp_diff < prev_temp_diff:  # 如果当前温度比之前更接近舒适区间
                    temp_reward += 1  # 鼓励向舒适区间靠近的行为
                elif temp_diff > prev_temp_diff:  # 如果当前温度比之前更远离舒适区间
                    temp_penalty += 1  # 惩罚远离舒适区间的行为

        # 综合奖励
        return temp_reward - temp_penalty

    def humidity_reward(self):  # 濕度獎勵計算
        humidity_reward = 0
        humidity_penalty = 0

        # 濕度在目標範圍內
        if 40 <= self.in_humidity <= 60:
            humidity_reward += 20

        # 濕度不在目標範圍內
        else:
            # 計算濕度與目標範圍的差距
            if self.in_humidity < 40:
                humidity_diff = 40 - self.in_humidity
            else:
                humidity_diff = self.in_humidity - 60

            # 基本濕度懲罰
            humidity_penalty += 0.8 * humidity_diff

        #     # 如果有先前的狀態可以比較
            if self.pre_state is not None:
                humidity_change = self.in_humidity - \
                    self.pre_state[1]  # 當前濕度與前一狀態的差異

                # 濕度往正確方向變化
                if (self.in_humidity < 40 and humidity_change > 0) or (self.in_humidity > 60 and humidity_change < 0):
                    humidity_reward += 1  # 往正確方向前進的獎勵
                    # 根據變化速度調整獎勵，變化越快獎勵越多
                    # humidity_reward += abs(humidity_change) * 10

                # 濕度往錯誤方向變化
                else:
                    humidity_penalty += 1  # 往錯誤方向前進的懲罰
                    # 根據變化速度調整懲罰，變化越快懲罰越多
                    # humidity_penalty += abs(humidity_change) * 10
        return (humidity_reward - humidity_penalty)

    def reward_compute(self):  # 總獎勵計算
        # 總體獎勵
        # 獎勵可以考慮另外一點，如果環境狀態離目標狀態越近，則可以讓模型降低環境往目標值變化的速度，讓環境變化依舊往正確的方向前進，但速度減慢
        total_reward = self.temp_reward()+self.humidity_reward()
        return total_reward

    def check_if_done(self):  # done的判斷式
        done = False
        # 检查是否到达最大步数
        if self.current_step >= 3000:
            done = True
        return done

    def step(self, action):
        # 解析动作
        scaled_temp = self.scaler_temp.transform(
            [[action[0]]])[0][0]  # 冷氣溫度[-1,1]縮放到[20,28]
        scaled_air_flow = self.ac_air_flow.transform(
            [[action[1]]])[0][0]  # 冷氣風量[-1,1]縮放到[1.01,2.6]

        reshaped_air_flow = scaled_air_flow.reshape(-1, 1)
        air_flow_for_humi = self.ac_air_flow_for_humi.transform(
            reshaped_air_flow).flatten()[0]  # 濕度[1.01,2.6]縮放到[10,500]

        # 更新環境狀態
        self.temp_simulator.air_conditioner_temperature = scaled_temp  # 更新冷氣溫度
        self.temp_simulator.update_temperature_field(
            ac_level=scaled_air_flow)  # 更新冷氣風量
        self.in_temperature = np.mean(
            self.temp_simulator.temperature_field[20:30, 5:25])  # 根據input決定採用溫度場的哪個範圍
        self.in_humidity = self.humi_simulator.humidity_simu(
            ave_temp=self.in_temperature, ac_level=air_flow_for_humi)

        # 計算即時獎勵（這裡可以根據特定的目標定義獎勵函數）
        total_reward = self.reward_compute()
        self.pre_state = np.array(
            [self.in_temperature, self.in_humidity])  # 紀錄當前狀態

        # 檢查是否到達目標狀態（這裡可以根據具體目標定義）
        done = self.check_if_done()

        self.current_step += 1

        info = {}
        obs = np.array(
            [self.in_temperature, self.in_humidity], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low,
                      self.observation_space.high)
        # 返回新的環境狀態、即時獎勵、結束標記等信息
        return obs, total_reward, done, False, info


class ac_temp_co2(gym.Env):
    def __init__(self):
        super(ac_temp_co2, self).__init__()
        # 定義環境空間
        self.observation_space = spaces.Box(low=np.array(
            [5, 400]), high=np.array([40, 800]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array(
            [-1, -1]), high=np.array([1, 1]), dtype=np.float32)  # 0為關窗、1為開窗
        self.reset()
        self.pre_state = None

        # 縮放動作數值
        self.scaler_temp = MinMaxScaler(feature_range=(20, 28))  # 冷氣溫度20-28度
        possible_actions_temp = np.array([[-1], [1]])
        self.scaler_temp.fit(possible_actions_temp)

        self.ac_air_flow = MinMaxScaler(
            feature_range=(1.01, 2.6))  # 冷氣風量1.01-2.6
        possible_actions_window = np.array([[-1], [1]])
        self.ac_air_flow.fit(possible_actions_window)

        self.ac_air_flow_for_co2 = MinMaxScaler(
            feature_range=(0, 50))  # 冷氣風量縮放到400-2000給co2用
        actions_air_flow_for_co2 = np.array([[1.01], [2.6]])
        self.ac_air_flow_for_co2.fit(actions_air_flow_for_co2)

    def reset(self, **kwargs):  # seed=None, option=None,
        # 固定種子碼
        seed = kwargs.get('seed', None)
        super().reset(seed=seed)
        options = kwargs.pop('options', None)
        # 初始化室內環境參數
        np_random = self.np_random
        # 调用env_reset并获取环境参数
        env_params = set_env_range(np_random, **kwargs)
        
        # 初始化室內環境參數
        self.in_temperature = env_params["in_temperature"]
        self.in_humidity = env_params["in_humidity"]
        self.in_co2 = env_params["in_co2"]
        self.out_temperature = env_params["out_temperature"]
        self.out_co2 = env_params["out_co2"]
        self.out_humidity = env_params["out_humidity"]

        # 紀錄環境在特定step的狀態
        self.pre_state = np.array(
            [self.in_temperature, self.in_co2])  # 記錄前一個step狀態
        self.first_state = np.array(
            [self.in_temperature, self.in_co2])  # 記錄第一個step狀態

        self.temp_simulator = TempSimulator(length=1, width=1, grid_size_x=30, grid_size_y=30,
                                            delta_t=5, init_temp=self.in_temperature, air_conditioner_temperature=26, out_temp=self.out_temperature)  # 溫度模擬

        self.humi_simulator = Humidity_simulation(in_humi=self.in_humidity, out_humi=self.out_humidity, win_h=50, win_w=60, car_length=100,
                                                  car_width=100, car_height=130, speed_per_hour=70, ave_temp=self.in_temperature)  # 濕度模擬

        self.CO2_simulator = co2_simulator(
            indoor_co2=self.in_co2, outdoor_co2=self.out_co2, ventilation_rate=self.humi_simulator.get_air_volume(ac_level=50), V=self.humi_simulator.get_room_volume())  # CO2模擬

        # 初始化時間步數
        self.current_step = 0  # 假設開車一個半小時

        obs = np.array(
            [self.in_temperature, self.in_co2], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low,
                      self.observation_space.high)
        return (obs, {})

    def temp_reward(self):  # 溫度的獎勵計算
        temp_reward = 0
        temp_penalty = 0
        comfortable_temp_min = 22
        comfortable_temp_max = 26
        # 当温度在舒适范围内时，给予正奖励；否则，给予惩罚
        if comfortable_temp_min <= self.in_temperature <= comfortable_temp_max:
            # 在舒适范围内，可以给予一个固定的正奖励，或者根据温度距离舒适区间边缘的远近给予不同的奖励
            temp_reward += 20
        else:
            # 温度离开舒适区间时，根据距离给予惩罚
            temp_diff = min(abs(self.in_temperature - comfortable_temp_min),
                            abs(self.in_temperature - comfortable_temp_max))
            temp_penalty += temp_diff

            # 判断温度是否向正确方向变化
            # 前一个状态比当前状态更接近舒适区间边缘，则当前动作是正向的，应给予额外奖励
            if self.pre_state is not None:
                prev_temp_diff = min(abs(
                    self.pre_state[0] - comfortable_temp_min), abs(self.pre_state[0] - comfortable_temp_max))
                if temp_diff < prev_temp_diff:  # 如果当前温度比之前更接近舒适区间
                    temp_reward += 1  # 鼓励向舒适区间靠近的行为
                elif temp_diff > prev_temp_diff:  # 如果当前温度比之前更远离舒适区间
                    temp_penalty += 1  # 惩罚远离舒适区间的行为

        # 综合奖励
        return temp_reward - temp_penalty

    def co2_reward(self):  # 二氧化碳獎勵計算
        co2_reward = 0
        co2_penalty = 0

        # 如果二氧化碳浓度小于1000 ppm
        if self.in_co2 < 700:
            co2_reward += 20
        elif 700 <= self.in_co2 <= 999:
            co2_reward += 15
        else:
            # 超过1000 ppm的惩罚
            # co2_diff = self.in_co2 - 1000
            # co2_penalty += 0.01 * co2_diff

            # 如果有先前的状态可以比较
            if self.pre_state is not None:
                co2_change = self.in_co2 - self.pre_state[1]  # 当前CO2浓度与前一状态的差异

                # # 如果二氧化碳浓度往减少的方向变化
                # if co2_change < 0:
                #     # 根据变化速度调整奖励，变化越快奖励越多
                #     co2_reward += abs(co2_change) * 1
                # else:
                #     # 如果二氧化碳浓度增加，则增加惩罚
                #     co2_penalty += abs(co2_change) * 1
                if self.in_co2 > 600 and co2_change < 0:
                    # 根据变化速度调整奖励，变化越快奖励越多
                    co2_reward += 1
                elif self.in_co2 > 1000 and co2_change > 0:
                    # 如果二氧化碳浓度增加，则增加惩罚
                    co2_penalty += 1

        return co2_reward - co2_penalty

    def reward_compute(self):  # 總獎勵計算
        # 總體獎勵
        # 獎勵可以考慮另外一點，如果環境狀態離目標狀態越近，則可以讓模型降低環境往目標值變化的速度，讓環境變化依舊往正確的方向前進，但速度減慢
        total_reward = self.temp_reward()+self.co2_reward()
        return total_reward

    def check_if_done(self):  # done的判斷式
        done = False
        # 检查是否到达最大步数
        if self.current_step >= 3000:
            done = True
        return done

    def step(self, action):
        # 解析動作
        scaled_temp = self.scaler_temp.transform(
            [[action[0]]])[0][0]  # 冷氣溫度[-1,1]縮放到[20,28]
        scaled_air_flow = self.ac_air_flow.transform(
            [[action[1]]])[0][0]  # 冷氣風量[-1,1]縮放到[1.01,2.6]
        reshaped_air_flow = scaled_air_flow.reshape(-1, 1)
        air_flow_for_co2 = self.ac_air_flow_for_co2.transform(
            reshaped_air_flow).flatten()[0]

        self.temp_simulator.air_conditioner_temperature = scaled_temp

        # 更新環境狀態
        self.temp_simulator.simulate(duration=1)
        self.temp_simulator.update_temperature_field(
            ac_level=scaled_air_flow)
        self.in_temperature = np.mean(
            self.temp_simulator.temperature_field[20:30, 5:25])

        self.in_co2 = self.CO2_simulator.simulate_step(
            ac_level=air_flow_for_co2)

        # 計算即時獎勵（這裡可以根據特定的目標定義獎勵函數）
        total_reward = self.reward_compute()

        self.pre_state = np.array([self.in_temperature, self.in_co2])  # 紀錄當前狀態

        # 檢查是否到達目標狀態（這裡可以根據具體目標定義）
        done = self.check_if_done()

        self.current_step += 1
        info = {}
        obs = np.array(
            [self.in_temperature, self.in_co2], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low,
                      self.observation_space.high)
        # 返回新的環境狀態、即時獎勵、結束標記等信息
        return obs, total_reward, done, False, info


class ac_temp_humi_co2(gym.Env):
    def __init__(self):
        super(ac_temp_humi_co2, self).__init__()
        # 定義環境空間
        self.observation_space = spaces.Box(low=np.array(
            [5, 35, 400]), high=np.array([40, 90, 1500]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array(
            [-1, -1]), high=np.array([1, 1]), dtype=np.float32)  # 0為關窗、1為開窗
        self.reset()
        self.pre_state = None

        # 定义和拟合scaler
        self.scaler_temp = MinMaxScaler(feature_range=(20, 28))
        possible_actions_temp = np.array([[-1], [1]])
        self.scaler_temp.fit(possible_actions_temp)

        self.air_flow_for_temp = MinMaxScaler(
            feature_range=(1.01, 2.6))  # 給溫度場的冷氣風量
        possible_actions_air_flow = np.array([[-1], [1]])
        self.air_flow_for_temp.fit(possible_actions_air_flow)

        self.air_flow_for_humi = MinMaxScaler(
            feature_range=(10, 500))  # 給濕度的冷氣風量
        possible_air_for_humi = np.array([[1.01], [2.6]])
        self.air_flow_for_humi.fit(possible_air_for_humi)

        self.air_flow_for_co2 = MinMaxScaler(feature_range=(0, 50))  # 給濕度的冷氣風量
        possible_air_for_co2 = np.array([[1.01], [2.6]])
        self.air_flow_for_co2.fit(possible_air_for_co2)

    def reset(self, **kwargs):  # seed=None, option=None,
        # 固定種子碼
        seed = kwargs.get('seed', None)
        super().reset(seed=seed)
        options = kwargs.pop('options', None)

        np_random = self.np_random
        # 取得環境參數
        env_params = set_env_range(np_random, **kwargs)
        
        # 初始化室內環境參數
        self.in_temperature = env_params["in_temperature"]
        self.in_humidity = env_params["in_humidity"]
        self.in_co2 = env_params["in_co2"]
        self.out_temperature = env_params["out_temperature"]
        self.out_co2 = env_params["out_co2"]
        self.out_humidity = env_params["out_humidity"]
        
        # 紀錄環境在特定step的狀態
        self.pre_state = np.array([self.in_temperature,
                                   self.in_humidity, self.in_co2])  # 記錄前一個step狀態
        self.first_state = np.array([self.in_temperature,
                                     self.in_humidity, self.in_co2])  # 記錄第一個step狀態

        self.temp_simulator = TempSimulator(length=1, width=1, grid_size_x=30, grid_size_y=30,
                                            delta_t=5, init_temp=self.in_temperature, air_conditioner_temperature=26, out_temp=self.out_temperature)  # 溫度模擬

        self.humi_simulator = Humidity_simulation(in_humi=self.in_humidity, out_humi=self.out_humidity, win_h=50, win_w=60, car_length=100,
                                                  car_width=100, car_height=130, speed_per_hour=70, ave_temp=self.in_temperature)  # 濕度模擬

        self.CO2_simulator = co2_simulator(
            indoor_co2=self.in_co2, outdoor_co2=self.out_co2, ventilation_rate=self.humi_simulator.get_air_volume(ac_level=50), V=self.humi_simulator.get_room_volume())  # CO2模擬

        # 初始化時間步數
        self.current_step = 0  # 假設開車一個半小時

        obs = np.array(
            [self.in_temperature, self.in_humidity, self.in_co2], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low,
                      self.observation_space.high)
        return (obs, {})

    def temp_reward(self):  # 溫度的獎勵計算
        temp_reward = 0
        temp_penalty = 0
        comfortable_temp_min = 20
        comfortable_temp_max = 26
        # 當溫度在舒適範圍內時，給予正獎勵；否則，給予懲罰
        if comfortable_temp_min <= self.in_temperature <= comfortable_temp_max:
            # 在舒適範圍內，可以給予一個固定的正獎勵，或者根據溫度距離舒適區間邊緣的遠近給予不同的獎勵
            temp_reward += 20
        else:
            # 溫度離開舒適區間時，根據距離給予懲罰
            temp_diff = min(abs(self.in_temperature - comfortable_temp_min),
                            abs(self.in_temperature - comfortable_temp_max))
            temp_penalty += temp_diff

            # 判斷溫度是否向正確方向變化
            # 前一個狀態比當前狀態更接近舒適區間邊緣，則當前動作是正向的，應給予額外獎勵
            if self.pre_state is not None:
                prev_temp_diff = min(abs(
                    self.pre_state[0] - comfortable_temp_min), abs(self.pre_state[0] - comfortable_temp_max))
                if temp_diff < prev_temp_diff:  # 如果當前溫度持續往舒適區間變化
                    temp_reward += 1  
                elif temp_diff > prev_temp_diff:  # 如果當前溫度正在舒適區間的反方向變化
                    temp_penalty += 1  

        # 綜合獎勵
        return temp_reward - temp_penalty

    def humidity_reward(self):  # 濕度獎勵計算
        humidity_reward = 0
        humidity_penalty = 0

        # 濕度在目標範圍內
        if 30 <= self.in_humidity <= 60:
            humidity_reward += 20

        # 濕度不在目標範圍內
        else:
            # 計算濕度與目標範圍的差距
            if self.in_humidity < 30:
                humidity_diff = 30 - self.in_humidity
            else:
                humidity_diff = self.in_humidity - 60

            # 基本濕度懲罰
            humidity_penalty += 0.8 * humidity_diff

        #     # 如果有先前的狀態可以比較
            if self.pre_state is not None:
                humidity_change = self.in_humidity - \
                    self.pre_state[1]  # 當前濕度與前一狀態的差異

                # 濕度往正確方向變化
                if (self.in_humidity < 30 and humidity_change > 0) or (self.in_humidity > 60 and humidity_change < 0):
                    humidity_reward += 1  # 往正確方向前進的獎勵


                # 濕度往錯誤方向變化
                else:
                    humidity_penalty += 1  # 往錯誤方向前進的懲罰

        return 0.5*(humidity_reward - humidity_penalty)

    def co2_reward(self):  # 二氧化碳獎勵計算
        co2_reward = 0
        co2_penalty = 0

        # 如果二氧化碳濃度小於1000 ppm
        if self.in_co2 < 700:
            co2_reward += 20
        elif 700 <= self.in_co2 <= 999:
            co2_reward += 15
        else:
            # 超過1000 ppm的懲罰
            # co2_diff = self.in_co2 - 1000
            # co2_penalty += 0.01 * co2_diff

            # 如果有先前的狀態可以比較
            if self.pre_state is not None:
                co2_change = self.in_co2 - self.pre_state[2]  # 當前CO2濃度與前一狀態的差異

                # # 如果二氧化碳濃度往減少的方向變化
                # if self.in_co2 > 600 and co2_change < 0:
                #     co2_reward += 1
                if self.in_co2 > 1000 and co2_change > 0:
                    # 如果二氧化碳濃度增加，則增加懲罰
                    co2_penalty += 1

        return co2_reward - co2_penalty


    def reward_compute(self):  # 總獎勵計算
        # 總體獎勵
        # 獎勵可以考慮另外一點，如果環境狀態離目標狀態越近，則可以讓模型降低環境往目標值變化的速度，讓環境變化依舊往正確的方向前進，但速度減慢
        total_reward = self.temp_reward()+self.humidity_reward() + \
            self.co2_reward()

        return total_reward

    def check_if_done(self):  # done的判斷式
        done = False
        # 检查是否到最大步數
        if self.current_step >= 3000:
            done = True
        return done

    def step(self, action):
        scaled_temp = self.scaler_temp.transform(
            [[action[0]]])[0][0]  # 冷氣溫度[-1,1]縮放到[20,28]
        self.temp_simulator.air_conditioner_temperature = scaled_temp

        scaled_air_flow_for_temp = self.air_flow_for_temp.transform(
            [[action[1]]])[0][0]  # 冷氣風量[-1,1]縮放到[1.01,2.6]
        reshaped_air_flow = scaled_air_flow_for_temp.reshape(-1, 1)
        air_flow_for_humi = self.air_flow_for_humi.transform(
            reshaped_air_flow).flatten()[0]  # 濕度[1.01,2.6]縮放到[10,500]
        air_flow_for_co2 = self.air_flow_for_co2.transform(reshaped_air_flow).flatten()[
            0]  # co2[1.01,2.6]縮放到[0,50]
        # 更新環境狀態
        self.temp_simulator.simulate(duration=1)
        self.temp_simulator.update_temperature_field(
            ac_level=scaled_air_flow_for_temp)  # 更新冷氣風量

        self.in_temperature = np.mean(
            self.temp_simulator.temperature_field[20:30, 5:25])
        self.in_humidity = self.humi_simulator.humidity_simu(
            ave_temp=self.in_temperature, ac_level=air_flow_for_humi)
        self.in_co2 = self.CO2_simulator.simulate_step(
            ac_level=air_flow_for_co2)

        # 計算即時獎勵（這裡可以根據特定的目標定義獎勵函數）
        total_reward = self.reward_compute()

        self.pre_state = np.array([self.in_temperature,
                                  self.in_humidity, self.in_co2])  # 紀錄當前狀態

        # 檢查是否到達目標狀態（這裡可以根據具體目標定義）
        done = self.check_if_done()

        self.current_step += 1
        info = {}
        obs = np.array(
            [self.in_temperature, self.in_humidity, self.in_co2], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low,
                      self.observation_space.high)
        # 返回新的環境狀態、即時獎勵、結束標記等信息
        return obs, total_reward, done, False, info


def test_ac_temp_humi():
    # 濕度測試
    env = ac_temp_humi()
    state, info = env.reset(custom_in_temperature=28, custom_in_humidity=60)
    print("初始状态:", state)
    for i in range(1000):
        obs, total_reward, done, truncate, info = env.step([1, 1])
        print(f"狀態: {obs}")

    print("初始状态:", state)
    print(f"室外濕度{env.out_humidity}")


def test_ac_temp_co2():
    # co2測試
    env = ac_temp_co2()
    state, info = env.reset(custom_in_temperature=28,
                            custom_in_co2=500, custom_out_co2=600)

    for i in range(1000):
        obs, total_reward, done, truncate, info = env.step([-1, 1])
        print(f"狀態: {obs}")

    print("初始状态:", state)
    print(f"室外co2:{env.out_co2}")


def test_ac_temp_humi_co2():
    # 三者混合測試
    env = ac_temp_humi_co2()
    state, info = env.reset(
        custom_in_temperature=28,
        custom_in_humidity=60,
        custom_out_humidity=50,
        custom_in_co2=500,
        custom_out_co2=800
    )

    for i in range(1000):
        obs, total_reward, done, truncate, info = env.step([-1, 1])
        print(f"狀態: {obs}")

    print("初始状态:", state)
    print(f"室外濕度:{env.out_humidity}室外co2:{env.out_co2}")


if __name__ == "__main__":
    test_ac_temp_humi_co2()

    # 三者混合測試
    # env = win_temp_co2()
    # state, info = env.reset(
    #     custom_in_temperature=30,
    #     custom_out_temperature=25,
    #     custom_in_humidity=60,
    #     custom_out_humidity=80,
    #     custom_in_co2=600,
    #     custom_out_co2=1000
    # )

    # for i in range(500):
    #     obs, total_reward, done, truncate, info = env.step([-1, 1])
    #     print(f"狀態: {obs}")
    # env.temp_simulator.plot_temperature_field()

    # print("初始状态:", state)
    # print(f"室外溫度:{env.out_temperature}室外濕度:{env.out_humidity}")

    # 定义一些要测试的动作
    # test_actions = np.array([
    #     [-1, -1],  # 最小值
    #     [0, 0],    # 中间值
    #     [1, 1]     # 最大值
    # ])
    # for action in test_actions:
    #     print(f"原始动作: {action}")
    #     env.step(action)
