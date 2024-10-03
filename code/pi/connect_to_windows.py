import adafruit_dht
import board
import time
import RPi.GPIO as GPIO
from ccs811_test import CCS811
import socket
GPIO.setwarnings(False)
# 创建 socket 对象
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 连接到电脑端IP和端口
s.connect(('172.20.10.5', 12345))
# Setup GPIO and DHT
pin_pqrs = 17 
LED_green_pin=16 #red_LED
LED_red_pin=26 #green_LED
buzzer_pin=12
GPIO.setup(LED_red_pin,GPIO.OUT)
GPIO.setup(LED_green_pin,GPIO.OUT)
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer_pin,GPIO.OUT) #buzzer pin

GPIO.setup(pin_pqrs, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) #light sensor
dht = adafruit_dht.DHT11(board.D4)
p=GPIO.PWM(buzzer_pin,1)

# Initialize CCS811
c = CCS811()
c.setup()  # Explicitly call setup

final_result="0"
def read_dht11():
    retries = 3
    while retries > 0:
        try:
            t = dht.temperature
            h = dht.humidity
            if t is not None and h is not None:
                return t, h
        except RuntimeError as error:
            print("Error reading DHT11 sensor:", error)
            retries -= 1
            time.sleep(2)  # 等待一段时间后重试
    return None, None

while True:
    try:
        t,h=read_dht11()
        # Read DHT11 sensor
        if t is not None and h is not None:
            print('溫度:', t, end=' ')
            print('濕度:', h)
        else:
            print("DHT11 fail")
        # Read light sensor state
        status = GPIO.input(pin_pqrs)
        if status == False:
            print('亮度:', status)
        else:
            print('亮度:', status)
            
        if final_result == "False":
            GPIO.output(LED_red_pin, 0)
            GPIO.output(LED_green_pin, 1)

            p.stop()

        elif final_result =q= "True":
            GPIO.output(LED_red_pin, 1)
            GPIO.output(LED_green_pin, 0)
            p.start(50)
            p.ChangeFrequency(523)
            

        print("get final_result:",final_result,"type:",type(final_result))
        # Check if data is available from CCS811
        try:
            # 嘗試讀取數據
            if c.data_available():
                c.read_logorithm_results()
                print("二氧化碳:[%d]" % c.CO2)
        except Exception as e:
            print("CCS811 error:", e)
        finally:
            data = f"Temperature: {t}, Humidity: {h}, CO2: {c.CO2}, light: {status}\n"
            s.sendall(data.encode('utf-8'))
            
            # 接收來自Windows的回應
            response = s.recv(1024)
            result=response.decode('utf-8')
            final_result=result
            print("Received response:",result)
            time.sleep(1)

    except RuntimeError as error:
        print(error)
    except KeyboardInterrupt:
        GPIO.cleanup()
        break
