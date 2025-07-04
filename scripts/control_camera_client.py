import socket

# Example positions: (zoom, pitch, yaw)
POSITIONS = [
    (2, 10, 10),
    (3, 13, 14),
    (5, 14, 11),
]

HOST = '192.168.0.161'  # IP of the RC Plus device
PORT = 8989

def main():
    with socket.create_connection((HOST, PORT)) as sock:
        for zoom, pitch, yaw in POSITIONS:
            cmd = f"SET {yaw} {pitch} {zoom}\n"
            sock.sendall(cmd.encode())
            sock.sendall(b"GET\n")
            resp = sock.recv(1024).decode().strip()
            print('Response:', resp)

if __name__ == '__main__':
    main()
