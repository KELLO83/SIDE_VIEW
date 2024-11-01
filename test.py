import socket
import requests
import base64

# UDP 서버 설정
UDP_IP = "127.0.0.1"  # 로컬 IP
UDP_PORT = 5000       # UDP 포트

# HTTP 서버 설정 (ngrok을 통해 생성된 URL을 사용)
HTTP_URL = "https://dd8b-211-210-158-226.ngrok-free.app"

# UDP 소켓 생성 및 수신 대기
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}...")

# while True:
#     # UDP 데이터 수신
#     data, addr = sock.recvfrom(1024)  # 1024는 버퍼 사이즈
#     print(f"Received message: {data} from {addr}")

#     # 데이터를 Base64로 인코딩하여 HTTP POST로 전송
#     encoded_data = base64.b64encode(data).decode('utf-8')  # Base64 인코딩
#     response = requests.post(HTTP_URL, json={"udp_data": encoded_data})
#     print(f"HTTP Response: {response.status_code}, {response.text}")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(b"Test message", ("127.0.0.1", 5000))