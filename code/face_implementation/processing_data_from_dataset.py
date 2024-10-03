"""從資料庫取出資料，並使用模型和影像識別來進行判別，並建構混淆矩陣"""
import warnings
import numpy as np
import sys
import pyodbc
from stable_baselines3 import DDPG,SAC,PPO,TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.envs.registration import register
import gymnasium as gym
import pandas as pd
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,accuracy_score,confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append("H:\我的雲端硬碟\paper\code")
from face_implementation.receive_from_pi import DataReceiver
# 忽略特定類型警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore")
def register_env(env_id,vec_env_path=None):
    # 檢查是否已經註冊過環境
    if env_id not in gym.envs.registry:
        # 如果未註冊，則進行註冊
        register(
            id=env_id,
            entry_point=f'DDPG_implementation.Env.MyEnv:{env_id}'
        )
    env=gym.make(env_id)
    env = DummyVecEnv([lambda: env])
    # 載入標準化統計數據
    if vec_env_path==None:
        env = VecNormalize.load(f"H:\我的雲端硬碟\paper\code\DDPG_implementation\{env_id}\{env_id}_env.pkl", env)
    else:
        env=VecNormalize.load(vec_env_path,env)
    # 現在使用標準化的環境進行預測
    obs = env.reset()
    return env
def plot_confusion_matrix(cm, title,table_name):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{table_name} Confusion Matrix for {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show() 
def connect_to_db():
    # 連接paper資料庫，給筆電用的
    connection = pyodbc.connect(
        driver='{SQL server}',
        SERVER='localhost',
        DATABASE='paper',
        UID='paper',
        trust_server_certificate='yes',
        # trusted_connection='yes',
        PWD='Anonymous43118')
    return connection
   
def model_and_whole_detect(BLINK_RATIO_THRESHOLD,YAWN_RATIO_THRESHOLD,MODEL_THRESHOLD,BLINK_WEIGHT,YAWN_WEIGHT,MODEL_WEIGHT): #根據資料庫內的環境數值，並調用模型和使用設定閥值來判別是否疲勞，並把兩個結果分別寫回資料表

    # 模型和資料表的對應字典
    models = {
        # "ac_temp_co2": DDPG.load("H:\我的雲端硬碟\paper\code\DDPG_implementation/ac_temp_co2/ac_temp_co2"),
        # "ac_temp_humi": DDPG.load("H:\我的雲端硬碟\paper\code\DDPG_implementation/ac_temp_humi/ac_temp_humi"),
        "ac_temp_humi_co2": DDPG.load("H:\我的雲端硬碟\paper\code\DDPG_implementation/ac_temp_humi_co2/ac_temp_humi_co2"),
        "td3_ac_temp_humi_co2": TD3.load("H:\我的雲端硬碟\paper\code\DDPG_implementation\compare_model/td3_train_result/td3_ac_temp_humi_co2_result/td3_ac_temp_humi_co2.zip"),
        "sac_ac_temp_humi_co2": SAC.load("H:\我的雲端硬碟\paper\code\DDPG_implementation\compare_model\sac_train_result\sac_ac_temp_humi_co2_result\sac_ac_temp_humi_co2.zip"),
        "ppo_ac_temp_humi_co2": PPO.load("H:\我的雲端硬碟\paper\code\DDPG_implementation\compare_model\ppo_train_result\ppo_ac_temp_humi_co2_result\ppo_ac_temp_humi_co2.zip"),
        # "win_temp_co2": DDPG.load("H:\我的雲端硬碟\paper\code\DDPG_implementation\win_temp_co2\win_temp_co2"),
        # "win_temp_humi": DDPG.load("H:\我的雲端硬碟\paper\code\DDPG_implementation\win_temp_humi\win_temp_humi"),
        "win_temp_humi_co2": DDPG.load("H:\我的雲端硬碟\paper\code\DDPG_implementation\win_temp_humi_co2\win_temp_humi_co2.zip"),
        "td3_win_temp_humi_co2": TD3.load("H:\我的雲端硬碟\paper\code\DDPG_implementation\compare_model/td3_train_result/td3_win_temp_humi_co2_result/td3_win_temp_humi_co2.zip"),
        "sac_win_temp_humi_co2": SAC.load("H:\我的雲端硬碟\paper\code\DDPG_implementation\compare_model\sac_train_result\sac_win_temp_humi_co2_result\sac_win_temp_humi_co2.zip"),
        "ppo_win_temp_humi_co2": PPO.load("H:\我的雲端硬碟\paper\code\DDPG_implementation\compare_model\ppo_train_result\ppo_win_temp_humi_co2_result\ppo_win_temp_humi_co2.zip")
    }

    # 讀取資料
    connection=connect_to_db()
    cursor = connection.cursor()
    sql_query = "SELECT ID, temp, humi, co2,blink_ratio,yawn_ratio FROM paper.dbo.data_from_participant_backup"
    cursor.execute(sql_query)
    rows = cursor.fetchall()

    # 用于判断疲劳的函数
    def is_fatigued(blink_ratio, yawn_ratio, model_result_sum):
        # 計算加權
        blink_score = (blink_ratio / BLINK_RATIO_THRESHOLD) * BLINK_WEIGHT
        yawn_score = (yawn_ratio / YAWN_RATIO_THRESHOLD) * YAWN_WEIGHT
        model_score = (model_result_sum / MODEL_THRESHOLD) * MODEL_WEIGHT
        total_score = blink_score + yawn_score + model_score
        # 判断是否疲劳
        fatigue_threshold = 1.0  # 疲勞閥值
        return total_score > fatigue_threshold
    # 遍歷每個模型及其對應的資料表
    for model_name, model in models.items():
        # 解析模型名稱，確定需要的輸入參數
        inputs = []
        if "temp" in model_name:
            inputs.append("temp")
        if "humi" in model_name:
            inputs.append("humi")
        if "co2" in model_name:
            inputs.append("co2")
        if "ac_temp_humi_co2" in model_name:
            env=register_env("ac_temp_humi_co2")
        elif "win_temp_humi_co2" in model_name:
            env=register_env("win_temp_humi_co2")  
        else:
            env=register_env(model_name)  
        # 構建更新查詢
        table_name = "detect_result_" + model_name
        update_query = f"""
        UPDATE paper.dbo.{table_name}
        SET model_detect = ?, whole_detect = ?,face_detect=?
        WHERE ID = ?
        """
        for row in rows:
            # 整理資料格式
            data_list = [getattr(row, input) for input in inputs]  # 從行中提取需要的欄位
            data = np.array(data_list).reshape(1, -1)  # 格式化為模型輸入
            normalized_data = env.normalize_obs(data)
            # 使用模型進行預測
            model_output = model.predict(normalized_data, deterministic=True)
            model_result_sum = abs(model_output[0][0][0]) + abs(model_output[0][0][1])

            face_detect=1 if row.blink_ratio+row.yawn_ratio>=BLINK_RATIO_THRESHOLD+YAWN_RATIO_THRESHOLD else 0
            whole_result = is_fatigued(row.blink_ratio, row.yawn_ratio, model_result_sum)
            model_result = 1 if model_result_sum > MODEL_THRESHOLD else 0

            # 將預測結果寫回資料庫
            cursor.execute(update_query, (int(model_result), int(whole_result),int(face_detect), row.ID))

        # 提交更新到資料庫
        connection.commit()

    # 關閉連接
    cursor.close()
    connection.close()

def for_four_situation(model_path,vec_env_path,env_id, table_name,  
                           BLINK_RATIO_THRESHOLD, YAWN_RATIO_THRESHOLD, MODEL_THRESHOLD, 
                           BLINK_WEIGHT, YAWN_WEIGHT, MODEL_WEIGHT):
    connection=connect_to_db()
    cursor = connection.cursor()
    env=register_env(env_id,vec_env_path=vec_env_path)
    
    sql_query = f"SELECT ID, temp, humi, co2, blink_ratio, yawn_ratio FROM paper.dbo.{table_name}"
    cursor.execute(sql_query)
    rows = cursor.fetchall()
    
    def is_fatigued(blink_ratio, yawn_ratio, model_result_sum):# 判斷疲勞
        # 計算加權
        blink_score = (blink_ratio / BLINK_RATIO_THRESHOLD) * BLINK_WEIGHT
        yawn_score = (yawn_ratio / YAWN_RATIO_THRESHOLD) * YAWN_WEIGHT
        model_score = (model_result_sum / MODEL_THRESHOLD) * MODEL_WEIGHT
        total_score = blink_score + yawn_score + model_score
    
        fatigue_threshold = 1.0  # 總分大於1即為疲勞
        return total_score > fatigue_threshold
    
    update_query = f"""
    UPDATE paper.dbo.{table_name}
    SET model_detect = ?, whole_detect = ?, face_detect=?
    WHERE ID = ?
    """

    if "ppo" in model_path:
        model=PPO.load(model_path)
    elif "sac" in model_path:
        model=SAC.load(model_path)
    elif "td3" in model_path:
        model=TD3.load(model_path)
    else:
        model=DDPG.load(model_path)             
    
    for row in rows:
        data_list = [row.temp, row.humi, row.co2]  # Ensure these attributes exist
        data = np.array(data_list).reshape(1, -1)
        normalized_data = env.normalize_obs(data)
        model_output = model.predict(normalized_data, deterministic=True)
        model_result_sum = abs(model_output[0][0][0]) + abs(model_output[0][0][1])

        face_detect=1 if row.blink_ratio+row.yawn_ratio>=BLINK_RATIO_THRESHOLD+YAWN_RATIO_THRESHOLD else 0
        whole_result = is_fatigued(row.blink_ratio, row.yawn_ratio, model_result_sum)
        model_result = 1 if model_result_sum > MODEL_THRESHOLD else 0
        # 將預測結果寫回資料庫
        cursor.execute(update_query, (int(model_result), int(whole_result),int(face_detect), row.ID))

    connection.commit()
    cursor.close()


def build_confusion_matric(table_name): #調用資料表中的subjective_feeling,face_detect, ddpg_detect, whole_detect來建構混淆矩陣
    connection = connect_to_db()
    query = f"""
    SELECT ID,participant_name,temp,humi,co2,subjective_feeling, model_detect, whole_detect,face_detect
    FROM [paper].[dbo].[{table_name}]
    """
    data = pd.read_sql(query, connection)
    print(data.head(10))
    # 基於 face_detect 計算混淆矩陣並生成圖片
    cm_face = confusion_matrix(data['subjective_feeling'], data['face_detect'])
    
    print(cm_face)
    face_precision=precision_score(data['subjective_feeling'], data['face_detect'])
    face_recall=recall_score(data['subjective_feeling'], data['face_detect'])
    face_f1=f1_score(data['subjective_feeling'], data['face_detect'])
    face_accuracy=accuracy_score(data['subjective_feeling'], data['face_detect'])
    print(f"{table_name} face Precision: {face_precision:.2f}")
    print(f"{table_name} face Recall: {face_recall:.2f}")
    print(f"{table_name} face F1 Score: {face_f1:.2f}")
    print(f"{table_name} face accuracy: {face_accuracy:.2f}")
    print(f"{table_name} Confusion Matrix for Face Detect:")
    print(classification_report(data['subjective_feeling'], data['face_detect']))
    
    plot_confusion_matrix(cm_face, "Face Detect",table_name=table_name)
    # 基於 whole_detect 計算混淆矩陣並生成圖片
    cm_whole = confusion_matrix(data['subjective_feeling'], data['whole_detect'])
    whole_precision=precision_score(data['subjective_feeling'], data['whole_detect'])
    whole_recall=recall_score(data['subjective_feeling'], data['whole_detect'])
    whole_f1=f1_score(data['subjective_feeling'], data['whole_detect'])
    whole_accuracy=accuracy_score(data['subjective_feeling'], data['whole_detect'])
    print(f"{table_name} whole Precision: {whole_precision:.2f}")
    print(f"{table_name} whole Recall: {whole_recall:.2f}")
    print(f"{table_name} whole f1_score: {whole_f1:.2f}")
    print(f"{table_name} whole accuracy: {whole_accuracy:.2f}")
    print(f"{table_name} Confusion Matrix for Whole Detect:")
    print(classification_report(data['subjective_feeling'], data['whole_detect']))
    
    print(cm_whole)
    plot_confusion_matrix(cm_whole, "Whole Detect",table_name=table_name)
    


    connection.close()
    
if __name__ == "__main__":
    # model_and_whole_detect( #專門給統一情境的
    #     # 设置疲劳阈值
    #     BLINK_RATIO_THRESHOLD = 0.38,  # 假设眨眼占比30%为疲劳的阈值
    #     YAWN_RATIO_THRESHOLD = 0.06,  # 假设打哈欠占比15%为疲劳的阈值
    #     MODEL_THRESHOLD=2, #模型動作的閥值
    
    #     #設定權重
    #     BLINK_WEIGHT = 0.4, #眨眼的權重
    #     YAWN_WEIGHT = 0.4, #哈欠的權重
    #     MODEL_WEIGHT = 0.2 #模型判別結果的權重 
    # )
    
    env_path="H:\我的雲端硬碟\paper\code\DDPG_implementation\compare_model/ddpg_train_result/ddpg_win_temp_humi_co2_result/ddpg_win_temp_humi_co2_env.pkl"
    model_path="H:\我的雲端硬碟\paper\code\DDPG_implementation\compare_model/ddpg_train_result/ddpg_win_temp_humi_co2_result/ddpg_win_temp_humi_co2.zip"
    table_name="additional_data_copy"
    env_id="win_temp_humi_co2"
    blink_ratio_threshold=0.2
    yawn_ratio_threshold=0.06
    model_threshold=1
    
    blink_weight=0.35
    yawn_weight=0.35
    model_weight=0.3
    for_four_situation( #寫入四個不同情境資料表的code
        env_id=env_id,
        model_path=model_path,
        vec_env_path=env_path,
        table_name=table_name,
        # 设置疲劳阈值
        BLINK_RATIO_THRESHOLD = blink_ratio_threshold,  # 假设眨眼占比30%为疲劳的阈值
        YAWN_RATIO_THRESHOLD = yawn_ratio_threshold,  # 假设打哈欠占比15%为疲劳的阈值
        MODEL_THRESHOLD=model_threshold, #模型動作的閥值
    
        #設定權重
        BLINK_WEIGHT = blink_weight, #眨眼的權重
        YAWN_WEIGHT = yawn_weight, #哈欠的權重
        MODEL_WEIGHT = model_weight #模型判別結果的權重 
    )
    build_confusion_matric(table_name=table_name)
    table_name="additional_daytime_rainy"
    for_four_situation( #寫入四個不同情境資料表的code
        env_id=env_id,
        model_path=model_path,
        vec_env_path=env_path,
        table_name=table_name,
        # 设置疲劳阈值
        BLINK_RATIO_THRESHOLD = blink_ratio_threshold,  # 假设眨眼占比30%为疲劳的阈值
        YAWN_RATIO_THRESHOLD = yawn_ratio_threshold,  # 假设打哈欠占比15%为疲劳的阈值
        MODEL_THRESHOLD=model_threshold, #模型動作的閥值
    
        #設定權重
        BLINK_WEIGHT = blink_weight, #眨眼的權重
        YAWN_WEIGHT = yawn_weight, #哈欠的權重
        MODEL_WEIGHT = model_weight #模型判別結果的權重 
    )
    build_confusion_matric(table_name=table_name)
    table_name="additional_daytime_sunny"
    for_four_situation( #寫入四個不同情境資料表的code
        env_id=env_id,
        model_path=model_path,
        vec_env_path=env_path,
        table_name=table_name,
        # 设置疲劳阈值
        BLINK_RATIO_THRESHOLD = blink_ratio_threshold,  # 假设眨眼占比30%为疲劳的阈值
        YAWN_RATIO_THRESHOLD = yawn_ratio_threshold,  # 假设打哈欠占比15%为疲劳的阈值
        MODEL_THRESHOLD=model_threshold, #模型動作的閥值
    
        #設定權重
        BLINK_WEIGHT = blink_weight, #眨眼的權重
        YAWN_WEIGHT = yawn_weight, #哈欠的權重
        MODEL_WEIGHT = model_weight #模型判別結果的權重 
    )
    build_confusion_matric(table_name=table_name)    
    table_name="additional_night_rainy"
    for_four_situation( #寫入四個不同情境資料表的code
        env_id=env_id,
        model_path=model_path,
        vec_env_path=env_path,
        table_name=table_name,
        # 设置疲劳阈值
        BLINK_RATIO_THRESHOLD = blink_ratio_threshold,  # 假设眨眼占比30%为疲劳的阈值
        YAWN_RATIO_THRESHOLD = yawn_ratio_threshold,  # 假设打哈欠占比15%为疲劳的阈值
        MODEL_THRESHOLD=model_threshold, #模型動作的閥值
    
        #設定權重
        BLINK_WEIGHT = blink_weight, #眨眼的權重
        YAWN_WEIGHT = yawn_weight, #哈欠的權重
        MODEL_WEIGHT = model_weight #模型判別結果的權重 
    )
    build_confusion_matric(table_name=table_name)
    table_name="additional_night_sunny"
    for_four_situation( #寫入四個不同情境資料表的code
        env_id=env_id,
        model_path=model_path,
        vec_env_path=env_path,
        table_name=table_name,
        # 设置疲劳阈值
        BLINK_RATIO_THRESHOLD = blink_ratio_threshold,  # 假设眨眼占比30%为疲劳的阈值
        YAWN_RATIO_THRESHOLD = yawn_ratio_threshold,  # 假设打哈欠占比15%为疲劳的阈值
        MODEL_THRESHOLD=model_threshold, #模型動作的閥值
    
        #設定權重
        BLINK_WEIGHT = blink_weight, #眨眼的權重
        YAWN_WEIGHT = yawn_weight, #哈欠的權重
        MODEL_WEIGHT = model_weight #模型判別結果的權重 
    )
    build_confusion_matric(table_name=table_name)