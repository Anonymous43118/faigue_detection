import sys
import os
import numpy as np
import math
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
from DDPG_implementation.Env.temp_simu import TempSimulator


class Humidity_simulation:
    def __init__(self, in_humi, out_humi, win_w, win_h, car_length, car_width, car_height, speed_per_hour, ave_temp):
        self.win_h = win_h  # 車窗高度(cm)
        self.win_w = win_w  # 車窗寬度(cm)
        self.car_length = car_length  # 駕駛座長度(cm)
        self.car_width = car_width  # 駕駛座寬度(cm)
        self.car_height = car_height  # 駕駛座高度(cm)
        self.speed_per_hour = speed_per_hour  # 車輛時速(公里/小時)
        self.in_humi = in_humi  # 室內相對溼度(%)
        self.out_humi = out_humi  # 室外濕度(%)
        self.ave_temp = ave_temp  # 平均溫度(攝氏)

    def self_gen_steam(self):  # 模擬自生水氣
        self_gen_steam = 0.001  # 人體自生水氣(kg/s)，理論值為0.011
        air_indensity = 1.29  # 空氣密度(kg/m^3)

        result = self_gen_steam/(air_indensity*self.get_room_volume())
        return result

    def get_air_volume(self, windows_open_rate=None, ac_level=None):  # 風量計算，單位m^3/s
        if windows_open_rate is not None and ac_level is None:  # 考慮車窗風量
            window_area = (self.win_h/100)*(self.win_w/100) * \
                windows_open_rate  # 窗戶截面積(m^2)
            wind_speed = self.speed_per_hour*1000/3600  # 假設車速固定為時速70，並換算為m/s
            # print("車窗開啟、濕度被呼叫")
            return window_area*wind_speed
        else:  # 考慮冷氣風量
            ac_length = 15
            ac_high = 7.5
            area = (ac_length/100)*(ac_high/100)
            ac_wind_speed = ac_level  # 空調進風量，約1~5 m/s，設為10~50
            # print("冷氣風量開啟、濕度被呼叫")
            return area*ac_wind_speed

    def get_env_AH(self, current_temp, humi):  # 將相對濕度轉換為絕對濕度
        pe = 16.37379 - 3876.659 / (current_temp + 229.73)
        p = math.exp(pe) * 1000  # 水饱和蒸气压，pa
        ah = (18 * p / (8.314 * (current_temp + 273.15))) * \
            humi / 100  # AH，绝对湿度 mg/L
        return ah

    def get_env_RH(self, current_temp, ah):  # 將絕對溼度轉換為相對濕度
        # 轉換為相對濕度
        pe = 16.37379 - 3876.659 / (current_temp + 229.73)
        p = math.exp(pe) * 1000  # 水饱和蒸气压，pa
        rh = 100*ah/((18*p)/(8.314*current_temp+8.314*273.15))
        return rh

    def get_room_volume(self):
        # 體積(m^3)
        return (self.car_length/100)*(self.car_height/100)*(self.car_width/100)

    # 計算每個時間的濕度變化量(單位為絕對溼度kg/kg)
    def humidity_simu(self, ave_temp, windows_open_rate=None, ac_level=None):
        Q_n = self.get_air_volume(
            windows_open_rate=windows_open_rate, ac_level=ac_level)
        V = self.get_room_volume()
        self.ave_temp = ave_temp
        o = self.get_env_AH(self.ave_temp, self.out_humi)
        i = self.get_env_AH(self.ave_temp, self.in_humi)
        delta_h = Q_n/V/1000 * (o - i)+self.self_gen_steam()
        self.in_humi = self.in_humi+delta_h
        # if self.in_humi > self.out_humi and windows_open_rate > 0:
        #     self.in_humi = self.out_humi

        return self.in_humi


if __name__ == '__main__':
    duration = 1000  # 5400秒，一個半小時
    init_temp = 28  # 初始室內溫度
    car_length, car_width, car_height = 100, 100, 130

    # 使用範例
    temp_simulator = TempSimulator(length=1, width=1, grid_size_x=30, grid_size_y=30,
                                   delta_t=5, init_temp=init_temp, air_conditioner_temperature=28, out_temp=15, ac_level=2.6)
    humi_simulator = Humidity_simulation(in_humi=60, out_humi=50, win_h=50, win_w=60, car_length=car_length,
                                         car_width=car_width, car_height=car_height, speed_per_hour=70, ave_temp=init_temp)

    for i in range(duration):
        temp_simulator.simulate(duration=1)
        average_temperature = np.mean(temp_simulator.temperature_field)

        humi_in_t = humi_simulator.humidity_simu(
            ave_temp=average_temperature, windows_open_rate=0.5, ac_level=500)
        # print("平均溫度 {:.2f} 度".format(average_temperature))
        print("當前濕度 {:.2f}%".format(humi_in_t))

