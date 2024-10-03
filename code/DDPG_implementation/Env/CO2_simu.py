import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')))
from DDPG_implementation.Env.humi_simu import Humidity_simulation
from DDPG_implementation.Env.temp_simu import TempSimulator
import numpy as np



class co2_simulator:
    def __init__(self, indoor_co2, outdoor_co2,  ventilation_rate, V):
        self.outdoor_co2 = outdoor_co2  # 室外 CO2 濃度(ppm)
        self.indoor_co2 = indoor_co2  # 室內 CO2 濃度(ppm)
        self.ventilation_rate = ventilation_rate/3000  # 通風量(m^3/h)
        self.people_co2_gen = 0.0173  # 人產生的二氧化碳(m^3/h)
        self.v = V  # 體積(m^3)
        self.win_h = 50
        self.win_w = 60
        self.speed_per_hour = 70

    def get_air_volume(self, windows_open_rate=None, ac_level=None):  # 風量計算，單位m^3/s
        if windows_open_rate is not None and ac_level is None:  # 考慮車窗風量
            window_area = (self.win_h/100)*(self.win_w/100) * \
                windows_open_rate  # 窗戶截面積(m^2)
            wind_speed = self.speed_per_hour*1000/3600/360  # 假設車速固定為時速70，並換算為m/s
            return window_area*wind_speed

        else:  # 考慮冷氣風量
            if ac_level == 0:  # 如果動作為
                ac_level = 0.0001
            ac_length = 15
            ac_high = 7.5
            area = (ac_length/100)*(ac_high/100)
            ac_wind_speed = ac_level  # 進風量0-50
            return area*ac_wind_speed

    def simulate_step(self, windows_open_rate=None, ac_level=None):
        # 模擬每個時間步長的二氧化碳變化
        self.ventilation_rate = self.get_air_volume(
            windows_open_rate=windows_open_rate, ac_level=ac_level * 0.01 if ac_level is not None else None)
        if windows_open_rate is not None and ac_level is None:  # 車窗開啟
            dC_dt = ((self.outdoor_co2 - self.indoor_co2) *
                     (self.ventilation_rate) + self.people_co2_gen)/self.v
            # print("車窗開啟、co2被呼叫")
        else:  # 冷氣風量開啟
            dC_dt = ((self.outdoor_co2 - self.indoor_co2) *
                     self.ventilation_rate + self.people_co2_gen)/self.v
            # print("冷氣風量開啟、co2被呼叫")
        self.indoor_co2 += dC_dt

        return self.indoor_co2


if __name__ == '__main__':
    duration = 1000  # 單位:秒
    init_temp = 30  # 初始室內溫度
    car_length, car_width, car_height = 100, 100, 130

    # 使用範例
    temp_simulator = TempSimulator(length=1, width=1, grid_size_x=20, grid_size_y=20,
                                   delta_t=10, init_temp=init_temp, air_conditioner_temperature=15, out_temp=15)
    humi_simulator = Humidity_simulation(in_humi=50, out_humi=60, win_h=50, win_w=60, car_length=car_length,
                                         car_width=car_width, car_height=car_height, speed_per_hour=70, ave_temp=init_temp)
    CO2_simulator = co2_simulator(
        indoor_co2=500, outdoor_co2=600, ventilation_rate=humi_simulator.get_air_volume(ac_level=50), V=humi_simulator.get_room_volume())  # ac_level設為最大值，來取得最大風量

    for i in range(duration):
        temp_simulator.simulate(duration=1)
        average_temperature = np.mean(temp_simulator.temperature_field)

        humi_in_t = humi_simulator.humidity_simu(
            ave_temp=average_temperature, windows_open_rate=0.5, ac_level=50)
        CO2_in_t = CO2_simulator.simulate_step(
            windows_open_rate=None, ac_level=50)

        print("當前二氧化碳濃度 {:.2f} ppm".format(CO2_in_t))
