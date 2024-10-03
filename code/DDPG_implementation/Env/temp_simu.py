import numpy as np
import matplotlib.pyplot as plt


class TempSimulator:
    def __init__(self, length, width, grid_size_x, grid_size_y, delta_t, init_temp, air_conditioner_temperature, out_temp=None, windows_open_rate=None, ac_level=None):
        self.length = length
        self.width = width
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.delta_t = delta_t  # 以多少秒為一單位
        self.alpha = (0.0257 * delta_t) / \
            (1.225 * 1005 * ((length / grid_size_x) ** 2))  # 熱傳導係數，上限為0.25，超過則方程式無法收斂

        self.init_temp = init_temp
        self.temperature_field = init_temp * \
            np.ones((grid_size_x, grid_size_y))
        self.air_conditioner_temperature = air_conditioner_temperature  # 冷氣溫度
        self.windows_open_rate = windows_open_rate  # 車窗開啟幅度
        self.out_temp = out_temp  # 室外溫度
        self.ac_level = ac_level  # 數值介於1.01~2.6

    def update_temperature_field(self, windows_open_rate=None, ac_level=None):
        self.ac_level = ac_level
        self.windows_open_rate = windows_open_rate
        new_temperature_field = self.temperature_field.copy()

        if self.windows_open_rate == 0 or self.windows_open_rate == None or self.out_temp == None:
            self.temperature_field[:, 0] = self.init_temp  # 0代表沒開窗戶
        elif self.windows_open_rate > 0 and self.out_temp is not None:
            self.temperature_field[:, 0] = self.out_temp  # 左邊的窗戶

        self.temperature_field[self.grid_size_x - 1,
                               :] = self.air_conditioner_temperature  # 正前方冷氣
        if self.ac_level is not None and self.windows_open_rate is None:  # 有冷氣風量的變數的話
            for i in range(1, self.grid_size_x - 1):
                for j in range(1, self.grid_size_y - 1):
                    new_temperature_field[i, j] = self.temperature_field[i, j] + self.alpha * self.ac_level * (
                        (self.temperature_field[i + 1, j] + self.temperature_field[i - 1, j]) +
                        self.temperature_field[i, j + 1] + self.temperature_field[i, j - 1] - 4 *
                        self.temperature_field[i, j])
            # print("冷氣風量開啟、溫度被呼叫")
        elif self.ac_level is None and self.windows_open_rate is not None:  # 沒有冷氣風量變數
            for i in range(1, self.grid_size_x - 1):
                for j in range(1, self.grid_size_y - 1):
                    new_temperature_field[i, j] = self.temperature_field[i, j] + self.alpha * (
                        (self.temperature_field[i + 1, j] + self.temperature_field[i - 1, j]) +
                        self.temperature_field[i, j + 1] + self.temperature_field[i, j - 1] - 4 *
                        self.temperature_field[i, j])
            # print("車窗開啟、溫度被呼叫")
        self.temperature_field = new_temperature_field

    def plot_temperature_field(self):
        plt.imshow(self.temperature_field, cmap='hot',
                   origin='lower', extent=[0, self.width, 0, self.length])
        plt.colorbar(label='Temperature (°C)')
        plt.title('Temperature Distribution in the Car Interior')
        plt.xlabel('Width (m)')
        plt.ylabel('Length (m)')
        plt.show()

    def simulate(self, duration, windows_open_rate=None, ac_level=None):  # 返回每個step的溫度
        for t in range(duration):
            self.update_temperature_field(
                windows_open_rate=windows_open_rate, ac_level=ac_level)

        if self.out_temp is None and self.windows_open_rate is None:  # 沒考慮車窗的開啟幅度，上方正中間
            temp_result = np.mean(self.temperature_field[25:30, 10:20])
        elif self.ac_level is not None:  # 車窗開啟幅度
            temp_result = np.mean(self.temperature_field[20:30, 5:25])
        else:
            temp_result = np.mean(
                self.temperature_field[-10:, :10])  # 有考慮車窗開啟幅度，所以為左上角

        return temp_result


if __name__ == '__main__':
    # 使用範例
    simulator1 = TempSimulator(length=1, width=1, grid_size_x=30, grid_size_y=30,
                               delta_t=5, init_temp=30, air_conditioner_temperature=20, out_temp=25)  # delta_t可以調整溫度擴散速度，往加增擴散速度，往下減少，預設為5
    simulator1.simulate(duration=2500, windows_open_rate=0)  # 模擬2500步
    middle_top_1 = np.mean(simulator1.temperature_field[-10:, :10])
    print(f"沒冷氣風流的:{middle_top_1}")
    simulator1.plot_temperature_field()

