import socket
import threading
import cv2

# Example positions: (zoom, pitch, yaw)
POSITIONS = [
    (2, 10, 10),
    (3, 13, 14),
    (5, 14, 11),
]

HOST = '192.168.0.161'  # IP of the RC Plus device
PORT = 8989
RTSP_URL = "rtsp://user:192.168.0.160@192.168.0.161:8554/streaming/live/1"


def _control_loop(sock):
    for zoom, pitch, yaw in POSITIONS:
        cmd = f"SET {yaw} {pitch} {zoom}\n"
        sock.sendall(cmd.encode())
        sock.sendall(b"GET\n")
        resp = sock.recv(1024).decode().strip()
        print("Response:", resp)


def _stream_loop():
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("Failed to open RTSP stream")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("H20 Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    with socket.create_connection((HOST, PORT)) as sock:
        control_thread = threading.Thread(target=_control_loop, args=(sock,), daemon=True)
        control_thread.start()
        _stream_loop()
        control_thread.join()

if __name__ == '__main__':
    main()
