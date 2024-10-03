"""用於將開啟攝像頭以及連結樹莓派，來進行受試者資料取得"""
import warnings
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import sys
from collections import deque
import threading
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
import gymnasium as gym

sys.path.append("H:\我的雲端硬碟\paper\code")
from face_implementation.receive_from_pi import DataReceiver
# 忽略特定類型的警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
register(
    id="ac_temp_humi_co2",
    entry_point='DDPG_implementation.Env.MyEnv:ac_temp_humi_co2'
)

def model_env_load():
    # 加载环境
    env = gym.make("ac_temp_humi_co2")
    env = DummyVecEnv([lambda: env])  # 向量化環境
    env = VecNormalize.load("H:\我的雲端硬碟\paper\code\DDPG_implementation\compare_model\ppo_train_result\ppo_ac_temp_humi_co2_result\ppo_ac_temp_humi_co2_env.pkl", env)
    env.training = False  # 把環境設為評估模式
    env.norm_reward = True  # 不對獎勵進行normalization
    model = PPO.load("H:\我的雲端硬碟\paper\code\DDPG_implementation\compare_model\ppo_train_result\ppo_ac_temp_humi_co2_result\ppo_ac_temp_humi_co2.zip", env=env)
    return env, model

def sound_alarm(path):
    # play an alarm sound
    playsound.playsound("./Alert_Sound/alert1.alert1.mp3")

def mouth_open_alert(mouth, threshold=0.8):
    # 根據嘴巴長寬比設定閥值
    mar = mouth_aspect_ratio(mouth)
    if mar > threshold:
        return True
    else:
        return False

def mouth_aspect_ratio(mouth):
    # 計算嘴巴的長寬比
    A = np.linalg.norm(mouth[3] - mouth[9])  # 上嘴唇的寬度
    B = np.linalg.norm(mouth[2] - mouth[10])  # 下嘴唇的寬度
    C = np.linalg.norm(mouth[0] - mouth[6])  # 嘴巴的高度
    mar = (A + B) / (2.0 * C)  # 嘴巴的長寬比
    return mar

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

def set_sliding_window():
    # 初始化 blink_queue 和 yawn_queue
    blink_queue = deque([0] * 1800, maxlen=1800)
    yawn_queue = deque([0] * 1800, maxlen=1800)

    return blink_queue, yawn_queue

def get_blink_yawn_ratio(blink_queue,yawn_queue):
    # 計算 blink_ratio 和 yawn_ratio
    blink_ratio = sum(blink_queue) / len(blink_queue)
    yawn_ratio = sum(yawn_queue) / len(yawn_queue)
    return blink_ratio, yawn_ratio

def handle_data_received(temp, humi, co2, light):
    global temperature, humidity, co2_level, light_status
    temperature, humidity, co2_level, light_status = temp, humi, co2, light

def is_fatigued(blink_ratio, yawn_ratio, model_result_sum):# 判斷疲勞
    # 計算加權
    blink_score = (blink_ratio / BLINK_RATIO_THRESHOLD) * BLINK_WEIGHT
    yawn_score = (yawn_ratio / YAWN_RATIO_THRESHOLD) * YAWN_WEIGHT
    model_score = (model_result_sum / MODEL_ACTION_THRESHOLD) * MODEL_WEIGHT
    total_score = blink_score + yawn_score + model_score

    fatigue_threshold = 1.0  # 總分大於1即為疲勞
    return total_score > fatigue_threshold

#接收樹梅派數值設定-------------------------------------------------------------
receiver = DataReceiver(on_data_received=handle_data_received)
receiver_thread = threading.Thread(target=receiver.run)
receiver_thread.start()


#設定調用此檔案所需參數-------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="shape_predictor_68_face_landmarks.dat")
ap.add_argument("-a", "--alarm", type=str, default="",
                help="Alert_Sound\alert1.mp3")
ap.add_argument("-w", "--webcam", type=int, default=1,
                help="1")
args = vars(ap.parse_args())

#設定疲勞辨識所需閥值-------------------------------------------------------------
EYE_AR_THRESH = 0.24#眼睛長寬比
EYE_AR_CONSEC_FRAMES = 48 # 設定連續多少幀後觸發警報(此數值可忽略)

MOUTH_AR_THRESH = 0.7  # 嘴巴長寬比閥值
MOUTH_AR_CONSEC_FRAMES = 30  # 設定連續多少幀後觸發警報(此數值可忽略)

# 設定疲勞閾值-------------------------------------------------------------
BLINK_RATIO_THRESHOLD = 0.2  # 假設眨眼占比30%為疲勞的閾值
YAWN_RATIO_THRESHOLD = 0.06  # 假設打哈欠占比15%為疲勞的閾值
MODEL_ACTION_THRESHOLD=1 #模型閥值(模型兩動作範圍介於0-1之間，而模型動作總和不該大於1)
FINAL_FATIGUE_THRESHOLD = 2.05 #總疲勞閥值
blink_ratio,yawn_ratio=0,0 #初始化眨眼和哈欠頻率
# 設定判別權重-------------------------------------------------------------
BLINK_WEIGHT=0.35 #眼部權重
YAWN_WEIGHT=0.35 #嘴部權重
MODEL_WEIGHT=0.3 #模型動作權重
# 調用模型-------------------------------------------------------------
env,model=model_env_load()
COUNTER = 0
ALARM_ON = False

# 初始化計數器
MOUTH_COUNTER = 0
MOUTH_ALARM_ON = False

# 調用Dlib模型
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# 取得左眼右眼Index
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
mouth = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
# 啟動影像串流
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# 設定一分鐘區間queue的sliding window來判斷眨眼和哈欠頻率
blink_queue,yawn_queue = set_sliding_window()

# Initialize timer for sending data
last_sent_time = time.time()
send_interval = 2  # seconds

try:
    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        # 顯示接收到的數據
        if temperature is not None and humidity is not None and co2_level is not None:
            text = f"Temp: {temperature}C, Hum: {humidity}%, CO2: {co2_level} ppm, Light: {light_status}"
            cv2.putText(frame, text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        

        for rect in rects:
            # 臉部座標轉Numpy數組
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # 提取座標，並計算EAR
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # 雙眼平均EAR
            ear = (leftEAR + rightEAR) / 2.0
            # 嘴巴開口偵測
            mouth = shape[48:68]
            # 判斷嘴巴是否張開
            if mouth_open_alert(mouth):
                # 執行相應的動作，例如發出警報
                cv2.putText(frame, "Mouth OPEN!", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # 繪製嘴巴部位的輪廓
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            # 可視化眼睛部位
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # 確認ERA是否超過閥值
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                blink_queue.append(1)  # EAR 小於閾值，認為在眨眼，加入 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    #聲音警告
                    if not ALARM_ON:
                        ALARM_ON = True
                        # 確認參數是否包含警告音檔，如果有就使用
                        if args["alarm"] != "":
                            t = Thread(target=sound_alarm,
                                    args=(args["alarm"],))
                            t.deamon = True
                            t.start()
                    # draw an alarm on the frame
                    # cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # 眼睛縱橫比沒有低於眨眼閾值，就重置計數器和警報
            else: 
                COUNTER = 0
                ALARM_ON = False
                blink_queue.append(0)  # EAR 大於等於閾值，不在眨眼，加入 0

            # 嘴巴開啟檢測
            mouth = shape[48:68]  # 獲取嘴巴的landmarks
            mar = mouth_aspect_ratio(mouth)  # 計算嘴巴長寬比
            if mar > MOUTH_AR_THRESH:
                yawn_queue.append(1)  # MAR 大於閾值，認為在打哈欠，加入 1
                MOUTH_COUNTER += 1
                if MOUTH_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                    # 如果警報未開啟，則開啟警報
                    if not MOUTH_ALARM_ON:
                        MOUTH_ALARM_ON = True
                        # 播放警報聲音
                        if args["alarm"] != "":
                            t = Thread(target=sound_alarm, args=(args["alarm"],))
                            t.deamon = True
                            t.start()
                    # cv2.putText(frame, "YAWNING ALERT!", (10, 80),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                yawn_queue.append(0)  # MAR 小於等於閾值，不在打哈欠，加入 0
                MOUTH_COUNTER = 0
                MOUTH_ALARM_ON = False
            # 在畫面上繪製計算出的眼睛縱橫比，以幫助
            # 調試並設置正確的眼睛縱橫比閾值和幀計數器
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
            # 計算眨眼和哈欠占比
            blink_ratio, yawn_ratio = get_blink_yawn_ratio(blink_queue,yawn_queue)
            #取得模型動作總和
            obs=np.array([temperature,humidity,co2_level])
            obs = env.normalize_obs(obs) #標準化環境數值
            action, _states = model.predict(obs, deterministic=True) #把數值餵給模型
            action_sum=np.sum(action)
            final_result=is_fatigued(blink_ratio=blink_ratio,yawn_ratio=yawn_ratio,model_result_sum=action_sum)
            
            #調用thread執行發送給pi的任務
            # 在指定时间间隔后发送数据
            current_time = time.time()
            if current_time - last_sent_time > send_interval:
                send_thread = threading.Thread(target=receiver.send_data_to_pi, args=(final_result,))
                send_thread.start()
                last_sent_time = current_time
        # 顯示眨眼率和打哈欠率
        cv2.putText(frame, f"Blink Ratio: {blink_ratio:.2f}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,    (0, 255, 0), 2)
        cv2.putText(frame, f"Yawn Ratio: {yawn_ratio:.2f}",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
            
        # 按q終止執行
        if key == ord("q"):
            receiver.stop()
            break
finally:
    # 清理並關閉線程和資源
    receiver.stop()
    receiver_thread.join()
    cv2.destroyAllWindows()
    vs.stop()

cv2.destroyAllWindows()
vs.stop()
