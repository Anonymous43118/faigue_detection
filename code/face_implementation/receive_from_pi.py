import socket
import threading

class DataReceiver(threading.Thread):
    def __init__(self, host='0.0.0.0', port=12345, on_data_received=None):
        super().__init__()
        self.host = host
        self.port = port
        self.on_data_received = on_data_received
        self.server_socket = None
        self.conn = None
        self.addr = None
        self.temperature = None
        self.humidity = None
        self.co2 = None
        self.light = None
        self.running = True

    def run(self):
        self.start_server()
        self.receive_data()

    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print("Waiting for connection...")
        self.conn, self.addr = self.server_socket.accept()
        print(f"Connected by {self.addr}")

    def receive_data(self):
        try:
            while self.running:
                data = self.conn.recv(1024)
                if not data:
                    break
                decoded_data = data.decode('utf-8').strip()
                # print("Received data:", decoded_data)
                self.parse_data(decoded_data)
        finally:
            self.close_connection()

    def parse_data(self, data):
        try:
            data_parts = data.split(',')
            temp_data = {}
            for item in data_parts:
                key, value = item.strip().split(':')
                key = key.strip()
                value = value.strip()
                temp_data[key] = float(value)

            self.temperature = temp_data.get('Temperature', self.temperature)
            self.humidity = temp_data.get('Humidity', self.humidity)
            self.co2 = temp_data.get('CO2', self.co2)
            self.light = temp_data.get('light', self.light)
            
            if self.on_data_received:
                self.on_data_received(self.temperature, self.humidity, self.co2, self.light)

        except ValueError as e:
            print(f"Error parsing data: {e} - on item: {item}")
    
    def send_data_to_pi(self,result):
        try:
            response=str(result)
            self.conn.sendall(response.encode('utf-8'))
            print(f"Sent data to Pi: {response}")  # 添加调试信息
        except Exception as e:
            print(f"Error sending data back to Pi: {e}")
        
    def close_connection(self):
        if self.conn:
            self.conn.close()
        if self.server_socket:
            self.server_socket.close()
        self.running = False

    def stop(self):
        self.running = False
        self.close_connection()