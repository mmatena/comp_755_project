import socket
import sys
import rpyc


class OpenAiGymService(rpyc.Service):
    def __init__(self):
        super().__init__()
        self.value = None

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_set_value(self, value):
        self.value = value

    def exposed_get_value(self):
        return self.value


if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer

    hostname = socket.gethostbyname(socket.gethostname())
    print(f"HOSTNAME: {hostname}", file=sys.stderr)

    t = ThreadedServer(OpenAiGymService, port=18861)
    t.start()
