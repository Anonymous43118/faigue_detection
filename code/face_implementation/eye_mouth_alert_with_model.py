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
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import DDPG
import gymnasium as gym
import sys
import pyodbc
from collections import deque
import pandas as pd
sys.path.append("H:\我的雲端硬碟\paper\code")

# 忽略特定類型的警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

def register_env(env_id):
    register(id=env_id, entry_point=f'DDPG_implementation.Env.MyEnv:{env_id}')


def sound_alarm(path):
    # play an alarm sound
    playsound.playsound("./Alert_Sound/alert1.alert1.mp3")


def win_temp_humi_co2_reset(env_name):
    model_path = f"DDPG_implementation/{env_name}/{env_name}" 
    vec_env_path = f"DDPG_implementation/{env_name}/{env_name}_env.pkl"
    env = gym.make(env_name)
    model = DDPG.load(model_path)
    env = DummyVecEnv([lambda: env])
    # 載入並應用訓練時的 VecNormalize 環境狀態
    env = VecNormalize.load(
        vec_env_path, env)
    obs = env.reset()
    first_obs = obs

    return env, model, obs


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


def set_sliding_window(): #用queue建立一分鐘的sliding windows
    blink_timestamps = deque()
    yawn_timestamps = deque()
    frame_timestamps = deque()
    last_checked_time = time.time()
    return blink_timestamps, yawn_timestamps, frame_timestamps, last_checked_time


def get_blink_yawn_ratio(blink_timestamps, frame_timestamps): #計算眨眼和哈欠頻率
    blink_ratio = len(blink_timestamps) / \
        len(frame_timestamps) if frame_timestamps else 0
    yawn_ratio = len(yawn_timestamps) / \
        len(frame_timestamps) if frame_timestamps else 0
    return blink_ratio, yawn_ratio


def get_data(env_name, data_df, original_obs, action, blink_ratio, yawn_ratio, tired_or_not):
    # Start by determining which columns to include based on env_name
    if "win" in env_name:
        ac_temp = float(action[0][0])
        win_open_rate = float(action[0][1])
        action_data = {
            'ac_temp': ac_temp,
            'win_open_rate': win_open_rate
        }
    elif "ac" in env_name:
        ac_temp = float(action[0][0])
        ac_air_flow = float(action[0][1])
        action_data = {
            'ac_temp': ac_temp,
            'ac_air_flow': ac_air_flow
        }
    else:
        # Default action if no prefix is recognized
        ac_temp = float(action[0][0])
        action_data = {
            'ac_temp': ac_temp
        }

    # Now prepare the observation data based on suffix
    if "temp_humi_co2" in env_name:
        temp, humi, co2 = original_obs[0]
        obs_data = {
            'temp': float(temp),
            'humi': float(humi),
            'co2': float(co2)
        }
    elif "temp_humi" in env_name:
        temp, humi = original_obs[0]
        obs_data = {
            'temp': float(temp),
            'humi': float(humi)
        }
    elif "temp_co2" in env_name:
        temp, co2 = original_obs[0]
        obs_data = {
            'temp': float(temp),
            'co2': float(co2)
        }
    else:
        # Default observation data if no suffix is recognized
        temp = original_obs[0][0]
        obs_data = {
            'temp': float(temp)
        }

    # Merge all data together
    new_data = {**obs_data, **action_data,
                'blink_ratio': float(blink_ratio),
                'yawn_ratio': float(yawn_ratio),
                'tired_or_not': int(tired_or_not)}

    # Create a new DataFrame row and append it to the existing DataFrame
    new_row = pd.DataFrame([new_data])
    data_df = pd.concat([data_df, new_row], ignore_index=True)
    return data_df



def fatigue_or_not(ALARM_ON, blink_ratio, yawn_ratio, action, FINAL_FATIGUE_THRESHOLD): #判別是否疲勞
    if blink_ratio + yawn_ratio+abs(action[0][0])+abs(action[0][1]) > FINAL_FATIGUE_THRESHOLD:
        if not ALARM_ON:
            ALARM_ON = True
            if args["alarm"] != "":
                t = Thread(target=sound_alarm, args=(args["alarm"],))
                t.daemon = True
                t.start()
            cv2.putText(frame, "FATIGUE DETECTED!", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tired_or_not = True
    else:
        ALARM_ON = False
        tired_or_not = False
    return ALARM_ON, tired_or_not

# --------------------------------------------------------------------
# 連接paper資料庫，給筆電用的
connection = pyodbc.connect(
    driver='{SQL server}',
    SERVER='localhost',
    DATABASE='paper',
    UID='paper',
    trust_server_certificate='yes',
    # trusted_connection='yes',
    PWD='Anonymous43118')
# 連接paper資料庫，給桌電用的
# connection = pyodbc.connect(
#     driver='{ODBC Driver 17 for SQL Server}',
#     SERVER='localhost',
#     DATABASE='paper',
#     UID='paper',
#     trust_server_certificate='yes',
#     # trusted_connection='yes',
#     PWD='Anonymous43118')
cursor = connection.cursor()
# --------------------------------------------------------------------
# 調用環境模擬
env_name = "ac_temp_humi_co2" 
register_env(env_name)
env, model, obs = win_temp_humi_co2_reset(env_name=env_name)
# 調用panda的dataframe
# data_columns = ['Temperature', 'Humidity',
#                 'CO2', 'AC_Temp', 'AC_Flow', 'Blink_Ratio', 'Yawn_Ratio', 'tired_or_not']
# data_df = pd.DataFrame(columns=data_columns)

# 定義不同的環境配置表
table_config = {
    'ac_temp_humi_co2': {
        'table_name': 'ac_temp_humi_co2',
        'columns': 'temp, humi, co2, ac_temp, ac_air_flow, blink_ratio, yawn_ratio, tired_or_not'
    },
    'ac_temp_humi': {
        'table_name': 'ac_temp_humi',
        'columns': 'temp, humi, ac_temp, ac_air_flow, blink_ratio, yawn_ratio, tired_or_not'
    },
    'ac_temp_co2': {
        'table_name': 'ac_temp_co2',
        'columns': 'temp, co2, ac_temp, ac_air_flow, blink_ratio, yawn_ratio, tired_or_not'
    },
    'win_temp_humi_co2': {
        'table_name': 'win_temp_humi_co2',
        'columns': 'temp, humi, co2, ac_temp, win_open_rate, blink_ratio, yawn_ratio, tired_or_not'
    },
    'win_temp_humi': {
        'table_name': 'win_temp_humi',
        'columns': 'temp, humi, ac_temp, win_open_rate, blink_ratio, yawn_ratio, tired_or_not'
    },
    'win_temp_co2': {
        'table_name': 'win_temp_co2',
        'columns': 'temp, co2, ac_temp, win_open_rate, blink_ratio, yawn_ratio, tired_or_not'
    }
}
# 檢查環境名稱是否有在配置表中
if env_name in table_config:
    config = table_config[env_name]
    table_name = config['table_name']
    columns = config['columns'].split(', ')
    data_df = pd.DataFrame(columns=columns)
# --------------------------------------------------------------------

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="shape_predictor_68_face_landmarks.dat")
ap.add_argument("-a", "--alarm", type=str, default="",
                help="Alert_Sound\alert1.mp3")
ap.add_argument("-w", "--webcam", type=int, default=1,
                help="1")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

# 在程式碼開頭定義新的常數
MOUTH_AR_THRESH = 0.6  # 設定判斷嘴巴開啟的長寬比閾值
MOUTH_AR_CONSEC_FRAMES = 30  # 設定連續多少幀後觸發警報

# 初始化計數器
MOUTH_COUNTER = 0
MOUTH_ALARM_ON = False


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
mouth = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# 初始化隊列和時間戳
# 設定一分鐘區間 queue 的 sliding window 來判斷眨眼和哈欠頻率
blink_timestamps, yawn_timestamps, frame_timestamps, last_checked_time = set_sliding_window()

# 設定疲勞閾值
BLINK_RATIO_THRESHOLD = 0.3  # 假設眨眼佔比30%為疲勞的閾值
YAWN_RATIO_THRESHOLD = 0.15  # 假設打哈欠佔比15%為疲勞的閾值
FINAL_FATIGUE_THRESHOLD = 2.05


# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    current_time = time.time()  # 紀錄當前時間
    frame_timestamps.append(current_time)  # 把當前時間寫入queue
    # 只保留一分鐘內的內容
    while frame_timestamps and (current_time - frame_timestamps[0] > 60):
        frame_timestamps.popleft()
    # 模型和環境判斷
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    original_obs = env.get_original_obs()
    print("環境狀態:"+str(original_obs), "動作:"+str(action))

# loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
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

    # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            blink_timestamps.append(current_time)  # 紀錄眨眼的幀數
            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True
                    # check to see if an alarm file was supplied,
                    # and if so, start a thread to have the alarm
                    # sound played in the background
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm,
                                   args=(args["alarm"],))
                        t.deamon = True
                        t.start()
                # draw an alarm on the frame
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            ALARM_ON = False

        # 嘴巴開啟檢測
        mouth = shape[48:68]  # 獲取嘴巴的 landmarks
        mar = mouth_aspect_ratio(mouth)  # 計算嘴巴長寬比
        if mar > MOUTH_AR_THRESH:
            yawn_timestamps.append(current_time)
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
                cv2.putText(frame, "YAWNING ALERT!", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            MOUTH_COUNTER = 0
            MOUTH_ALARM_ON = False
        # draw the computed eye aspect ratio on the frame to help
        # with debugging and setting the correct eye aspect ratio
        # thresholds and frame counters
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # 維護事件隊列，僅保留最近一分鐘內的內容
        while blink_timestamps and (current_time - blink_timestamps[0] > 60):
            blink_timestamps.popleft()
        while yawn_timestamps and (current_time - yawn_timestamps[0] > 60):
            yawn_timestamps.popleft()
    # 計算眨眼和哈欠佔比
    blink_ratio, yawn_ratio = get_blink_yawn_ratio(
        blink_timestamps, frame_timestamps)


    # 顯示眨眼頻率和哈欠頻率
    cv2.putText(frame, f"Blink Ratio: {blink_ratio:.2f}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Yawn Ratio: {yawn_ratio:.2f}",
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    ALARM_ON, tired_or_not = fatigue_or_not(
        ALARM_ON, blink_ratio, yawn_ratio, action, FINAL_FATIGUE_THRESHOLD)

    # 抓取寫回sqlserver的資料
    data_df = get_data(env_name,data_df, original_obs, action, blink_ratio,
                       yawn_ratio, tired_or_not)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        print(data_df)
        break


data_tuples = [tuple(x) for x in data_df.to_numpy()]
column_names = ', '.join(columns)
sql_insert_query = f"INSERT INTO {table_name}({column_names}) VALUES ({', '.join(['?']*len(columns))})"
print(sql_insert_query)
try:
    cursor.executemany(sql_insert_query, data_tuples)
    connection.commit()
    print("Data inserted successfully")
except pyodbc.Error as e:
    print("Error occurred:", e)
finally:
    cursor.close()
    connection.close()
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
