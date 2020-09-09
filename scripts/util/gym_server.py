from multiprocessing import connection
import socket
import sys

hostname = socket.gethostbyname(socket.gethostname())
print(f"HOSTNAME: {hostname}", file=sys.stderr)


address = (hostname, 6000)  # family is deduced to be 'AF_INET'
listener = connection.Listener(address, authkey=b"authkey")
conn = listener.accept()
print("connection accepted from", listener.last_accepted)
while True:
    msg = conn.recv()
    # do something with msg
    if msg == "close":
        conn.close()
        break
listener.close()
