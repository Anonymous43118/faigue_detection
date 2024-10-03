# faigue_detection
結合影像辨識+PPO模型判別+環境數值
1.	get_data_from_subjective.py：啟動攝像頭並且連結到樹莓派以及筆電的資料庫，並將受試者的眼睛長寬比(EAR)、嘴部長寬比(MAR)、溫濕度、二氧化碳濃度都記錄下來，並寫回資料庫，執行指令如下:
python { get_data_from_subjective.py的檔案路徑} -p {shape_predictor_68_face_landmarks的路徑} -w {要調用的攝像頭編號，預設為0，有外接可設為1}

2.	DDPG_implementation資料夾：裡面包含DDPG,TD3,SAC,PPO的所有訓練的code
  compare_model資料夾：保存所有訓練好的模型(包含比較對象)、訓練時的環境檔、replay buffer的內容(如果模型本身有)、以及訓練之後的結果(可用tensorboard的logdir指令調用)
    i.	model_eval.py：對模型進行多個episode的平均獎勵評估用
    ii.	train_ppo_ac_temp_humi_co2.py：訓練PPO模型的code
    iii.	train_sac_ac_temp_humi_co2.py：訓練SAC模型的code
    iv.	train_td3_ac_temp_humi_co2.py：訓練TD3模型的code

  Env資料夾：包含所有訓練模型時會用到的環境，包含溫度(temp_simu.py)、濕度(humi_simu.py)、二氧化碳濃度模擬模組(co2_simu.py)以及獎勵函數計算(Rewardcalcuate.py，此檔案可以忽略)、調用所有模組的最終訓練環境(MyEnv.py)

  custom_wrapper.py：用於模型功能測試，可忽略
  train_ac_temp_co2.py到train_win_temp_humi.py：共六個檔案，用於訓練不同環境條件數值下的DDPG模型，可忽略

3.	face_implementation資料夾:包含調用攝像頭、調用訓練好的PPO模型以及執行實際疲勞判別的所有code
  did_not_use資料夾：為最初影像辨識原作者包含的所有python檔案，因為論文並未使用，所以存放於此資料夾中，對於原作者的影像識別功能有疑問，可以參閱此資料夾以及introduction資料夾
  dlib_model資料夾：存放用於定位人臉的dlib預訓練模型以及用於偵測臉部特徵點的預訓練模型shape_predictor
  eye_close_alert.py：用於偵測是否眨眼
  eye_mouth_alert.py：同時偵測眼部和嘴部狀態用
  eye_mouth_alert_with_model.py：從資料庫擷取數據集後，同時偵測眼部和嘴部狀態，並聯合其他強化學習模型進行共同判別用，再把判別結果寫回資料庫中，可忽略
  processing_data_from_dataset.py：將不同情境的偵測結果進行處理並存入sqlserver資料庫
  receive_from_pi.py：用於從樹莓派中接收環境數據用
  requirement.txt：為原作者的影像偵測的python環境需求
  start.py：論文整個疲勞駕駛系統的啟動檔案，於終端機執行以下指令：
  python {start.py的檔案路徑} -p {shape_predictor_68_face_landmarks的路徑} -w {要調用的攝像頭編號，預設為0，有外接可設為1}
  請依照自己的情況取代{}內的內容，並且需同時於pi上執行connect_to_windows.py，檔案內的ip位址可自行設定為自己所需的
