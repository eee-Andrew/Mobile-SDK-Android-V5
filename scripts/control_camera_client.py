import socket
import time
import cv2
from ultralytics import YOLO

# Example positions as (zoom, pitch, yaw)
POSITIONS = [
    (2, 10, 10),
    (3, 13, 14),
    (5, 14, 11),
]

HOST = "192.168.0.161"  # IP of the RC Plus device
PORT = 8989
RTSP_URL = "rtsp://user:192.168.0.160@192.168.0.161:8554/streaming/live/1"

model = YOLO("best.pt")


def parse_response(resp: str) -> dict:
    """Parse server response into a dictionary."""
    tokens = resp.split()
    data = {}
    for i in range(0, len(tokens) - 1, 2):
        key = tokens[i]
        try:
            value = float(tokens[i + 1])
        except ValueError:
            continue
        data[key] = value
    return data


def main():
    last_index = None
    last_resp = ""
    sock = socket.create_connection((HOST, PORT))
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("Failed to open RTSP stream")
        sock.close()
        return
    try:
        for idx, (zoom, pitch, yaw) in enumerate(POSITIONS):
            cmd = f"SET {yaw} {pitch} {zoom}\n"
            sock.sendall(cmd.encode())
            time.sleep(0.5)
            detected = False
            for _ in range(30):
                ret, frame = cap.read()
                if not ret:
                    break
                res = model(frame, verbose=False)[0]
                for cls_id in res.boxes.cls:
                    if res.names[int(cls_id)] == "truck":
                        sock.sendall(b"GET\n")
                        resp = sock.recv(1024).decode().strip()
                        data = parse_response(resp)
                        last_index = idx
                        last_resp = resp
                        print(
                            f"Detected truck: range {data.get('RANGE', -1)} m "
                            f"lat {data.get('LAT', 0)} lon {data.get('LON', 0)}"
                        )
                        detected = True
                        break
                cv2.imshow("H20 Stream", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    detected = True
                    break
                if detected:
                    break
            if not detected:
                print(f"No truck detected at index {idx}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.close()
        if last_index is not None:
            data = parse_response(last_resp)
            print(
                f"Last detection index {last_index}: range {data.get('RANGE', -1)} m "
                f"lat {data.get('LAT', 0)} lon {data.get('LON', 0)}"
            )
        else:
            print("No trucks detected")

if __name__ == '__main__':
    main()
