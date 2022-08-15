import socket

# s为监听对象，c为连接对象
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 1234))
    s.listen()
    c, addr = s.accept()
    with c:
        print(addr, "connected")

        while True:
            data = c.recv(1)
            print(data)
            print(str(data))
            if not data:
                break
            c.sendall(data)
