import numpy as np


class RewardCalculator():
    def __init__(self, in_temp, pre_state, in_humi=None, in_co2=None) -> None:
        self.in_temp = in_temp  # 當前室內溫度
        self.in_humi = in_humi  # 當前室內濕度
        self.in_co2 = in_co2  # 當前室內co2濃度
        self.pre_state = pre_state
        # print(self.pre_state)
        # print(f"外部環境數值:{self.in_temp}, {self.in_humi}, {self.in_co2}")

    def temp_reward(self):  # 溫度的獎勵計算
        temp_reward = 0
        temp_penalty = 0
        comfortable_temp_min = 22
        comfortable_temp_max = 26
        # 当温度在舒适范围内时，给予正奖励；否则，给予惩罚
        if comfortable_temp_min <= self.in_temp <= comfortable_temp_max:
            # 在舒适范围内，可以给予一个固定的正奖励，或者根据温度距离舒适区间边缘的远近给予不同的奖励
            temp_reward += 20
        else:
            # 温度离开舒适区间时，根据距离给予惩罚
            temp_diff = min(abs(self.in_temp - comfortable_temp_min),
                            abs(self.in_temp - comfortable_temp_max))
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
        if self.in_humi is None:  # 如果沒有傳入
            return 0
        else:
            humi_index = 1 if len(self.pre_state) == 2 else 2
        humidity_reward = 0
        humidity_penalty = 0

        # 濕度在目標範圍內
        if 40 <= self.in_humi <= 60:
            humidity_reward += 20

        # 濕度不在目標範圍內
        else:
            # 計算濕度與目標範圍的差距
            if self.in_humi < 40:
                humidity_diff = 40 - self.in_humi
            else:
                humidity_diff = self.in_humi - 60

            # 基本濕度懲罰
            humidity_penalty += humidity_diff

        #     # 如果有先前的狀態可以比較
            if self.pre_state is not None:
                humidity_change = self.in_humi - \
                    self.pre_state[humi_index]  # 當前濕度與前一狀態的差異

                # 濕度往正確方向變化
                if (self.in_humi < 40 and humidity_change > 0) or (self.in_humi > 60 and humidity_change < 0):
                    humidity_reward += 1  # 往正確方向前進的獎勵
                    # 根據變化速度調整獎勵，變化越快獎勵越多
                    # humidity_reward += abs(humidity_change) * 10

                # 濕度往錯誤方向變化
                else:
                    humidity_penalty += 1  # 往錯誤方向前進的懲罰
                    # 根據變化速度調整懲罰，變化越快懲罰越多
                    # humidity_penalty += abs(humidity_change) * 10
        return humidity_reward - humidity_penalty

    def co2_reward(self):  # 二氧化碳獎勵計算
        if self.in_co2 is None:  # 如果沒有傳入溫度
            return 0
        else:
            co2_index = 1 if len(self.pre_state) == 2 else 2
        
        co2_reward = 0
        co2_penalty = 0

        # 如果二氧化碳浓度小于1000 ppm
        # if self.in_co2 < 800:
        #     co2_reward += 25
        co2_reward = self.in_co2 *-0.1

        # else:
        #     # 超过1000 ppm的惩罚
        #     co2_diff = self.in_co2 - 1000
        #     co2_penalty += 0.02 * co2_diff

        #     # 如果有先前的状态可以比较
        #     if self.pre_state is not None:
        #         co2_change = self.in_co2 - \
        #             self.pre_state[co2_index]  # 当前CO2浓度与前一状态的差异

        #         if self.in_co2 > 600 and co2_change < 0:
        #             # 根据变化速度调整奖励，变化越快奖励越多
        #             co2_reward += 1
        #         elif self.in_co2 > 1000 and co2_change > 0:
        #             # 如果二氧化碳浓度增加，则增加惩罚
        #             co2_penalty += 1

        return co2_reward - co2_penalty

    def reward_compute(self):  # 總獎勵計算
        # 總體獎勵
        # 獎勵可以考慮另外一點，如果環境狀態離目標狀態越近，則可以讓模型降低環境往目標值變化的速度，讓環境變化依舊往正確的方向前進，但速度減慢
        total_reward = self.temp_reward()+self.humidity_reward() + self.co2_reward()
        return total_reward

# if __name__ == "__main__":
