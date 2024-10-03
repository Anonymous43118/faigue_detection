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
import pyodbc
from collections import deque
import pandas as pd
import threading
sys.path.append("H:\我的雲端硬碟\paper\code")
from face_implementation.receive_from_pi import DataReceiver
# 忽略特定類型的警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

def insert_data_to_db(participant_name,temp, humi, co2, light,blink_ratio,yawn_ratio,feeling):
    try:
        cursor.execute("""
            INSERT INTO paper.[dbo].[additional_data] (participant_name, temp, humi,co2, light,blink_ratio,yawn_ratio,subjective_feeling)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (participant_name, temp, humi,co2, light,blink_ratio,yawn_ratio,feeling))
        connection.commit()
        print("Data inserted successfully")
    except pyodbc.Error as e:
        print("Database error:", e)

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

# Setup data receiver
receiver = DataReceiver(on_data_received=handle_data_received)
receiver_thread = threading.Thread(target=receiver.run)
receiver_thread.start()
participant_name="禎晟"
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


EYE_AR_THRESH = 0.24#眼睛長寬比
EYE_AR_CONSEC_FRAMES = 48 # 設定連續多少幀後觸發警報

MOUTH_AR_THRESH = 0.7  # 嘴巴長寬比閥值
MOUTH_AR_CONSEC_FRAMES = 30  # 設定連續多少幀後觸發警報

# 設定疲勞閾值
BLINK_RATIO_THRESHOLD = 0.3  # 假設眨眼占比30%為疲勞的閾值
YAWN_RATIO_THRESHOLD = 0.15  # 假設打哈欠占比15%為疲勞的閾值
FINAL_FATIGUE_THRESHOLD = 2.05
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

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


# 設定一分鐘區間queue的sliding window來判斷眨眼和哈欠頻率
blink_queue,yawn_queue = set_sliding_window()

# 幀計數器
frame_counter = 0  # 初始化幀計數器
feeling = 0  # 初始感覺設為0，表示沒有按`t`
blink_ratio,yawn_ratio=0,0
try:
    # loop over frames from the video stream
    while True:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale channels)
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = detector(gray, 0)
        # 顯示接收到的數據
        if temperature is not None and humidity is not None and co2_level is not None:
            text = f"Temp: {temperature}C, Hum: {humidity}%, CO2: {co2_level} ppm, Light: {light_status}"
            cv2.putText(frame, text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        
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
                blink_queue.append(1)  # EAR 小於閾值，認為在眨眼，加入 1
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
                    # cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
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
            # draw the computed eye aspect ratio on the frame to help
            # with debugging and setting the correct eye aspect ratio
            # thresholds and frame counters
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
            # 增加幀計數器
            frame_counter += 1
            # 計算眨眼和哈欠占比
            blink_ratio, yawn_ratio = get_blink_yawn_ratio(blink_queue,yawn_queue)
            # 如果幀計數器達到60，寫入數據並重置計數器
            if frame_counter >= 60:
                # insert_data_to_db(participant_name, temperature, humidity, co2_level, light_status, blink_ratio, yawn_ratio, feeling)
                frame_counter = 0  # 重置幀計數器
                print(blink_queue)

            if feeling==1:
                cv2.putText(frame, "set tired", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "set not tired", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # 顯示眨眼率和打哈欠率
        cv2.putText(frame, f"Blink Ratio: {blink_ratio:.2f}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,    (0, 255, 0), 2)
        cv2.putText(frame, f"Yawn Ratio: {yawn_ratio:.2f}",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # 檢查是否按下`t`鍵
        if key == ord('t'):
            if feeling == 1:
                feeling = 0  # 0為不疲勞，1為疲勞
            else:
                feeling =1
            
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            receiver.stop()
            break
finally:
    # 清理並關閉線程和資源
    receiver.stop()
    receiver_thread.join()
    cursor.close()
    connection.close()
    cv2.destroyAllWindows()
    vs.stop()

cv2.destroyAllWindows()
vs.stop()
