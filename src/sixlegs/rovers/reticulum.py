"""Optional real Reticulum framing over an isolated simulated radio interface.

Every rover has a separate stack process. No LAN, Internet or physical-radio
interfaces are enabled. Plain broadcasts exercise framing/delivery, not routing
or encrypted sessions. The actual emitted bytes are charged LoRa airtime.
"""

import multiprocessing
import tempfile
import threading
from pathlib import Path


def _endpoint(connection, index):
    import RNS
    from RNS.Interfaces.Interface import Interface

    with tempfile.TemporaryDirectory(prefix=f"rover-rns-{index}-") as directory:
        Path(directory, "config").write_text(
            "[reticulum]\nshare_instance = No\nenable_transport = No\n\n[logging]\nloglevel = 0\n\n[interfaces]\n"
        )
        RNS.Reticulum(configdir=directory, loglevel=0)
        RNS.Transport.USE_INBOUND_QUEUE = (
            False  # process packets synchronously on the simulation clock
        )
        outgoing = []
        incoming = []
        lock = threading.Lock()

        class SimulatedInterface(Interface):
            def __init__(self):
                super().__init__()
                self.name = f"isolated-rover-{index}"
                self.IN = True
                self.OUT = True
                self.online = True
                self.mode = Interface.MODE_FULL
                self.bitrate = 5000
                self.HW_MTU = 255
                self.ifac_size = 0
                self.ifac_identity = None

            def process_outgoing(self, data):
                with lock:
                    outgoing.append(bytes(data))

            def __str__(self):
                return self.name

        interface = SimulatedInterface()
        RNS.Transport.interfaces.append(interface)
        destination = RNS.Destination(
            None, RNS.Destination.IN, RNS.Destination.PLAIN, "rovercomms", "inspection"
        )
        destination.set_packet_callback(
            lambda data, packet: incoming.append(bytes(data))
        )
        send_to = RNS.Destination(
            None, RNS.Destination.OUT, RNS.Destination.PLAIN, "rovercomms", "inspection"
        )
        connection.send(("ready", RNS.__version__))
        while True:
            command, data = connection.recv()
            if command == "close":
                break
            if command == "encode":
                outgoing.clear()
                RNS.Packet(send_to, data).send()
                connection.send(("encoded", list(outgoing)))
            elif command == "decode":
                incoming.clear()
                RNS.Transport.inbound(data, interface)
                connection.send(("decoded", list(incoming)))
        connection.close()


class ReticulumBridge:
    def __init__(self, count=6):
        try:
            import RNS  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "Install optional stack with uv sync --extra reticulum"
            ) from error
        context = multiprocessing.get_context("spawn")
        self.connections = []
        self.processes = []
        self.version = None
        try:
            for i in range(count):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_endpoint, args=(child, i), daemon=True
                )
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for connection in self.connections:
                if not connection.poll(15):
                    raise RuntimeError("Reticulum endpoint did not start")
                status, self.version = connection.recv()
                if status != "ready":
                    raise RuntimeError(status)
        except BaseException:
            self.close()
            raise

    def exchange(self, node, command, data):
        connection = self.connections[node]
        connection.send((command, data))
        if not connection.poll(5):
            raise RuntimeError(f"Reticulum {command} timed out on rover {node}")
        status, result = connection.recv()
        if status != ("encoded" if command == "encode" else "decoded"):
            raise RuntimeError(status)
        return result

    def encode(self, node, payload):
        return self.exchange(node, "encode", payload)

    def decode(self, node, frame):
        return self.exchange(node, "decode", frame)

    def close(self):
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for connection in self.connections:
            connection.close()
        self.connections = []
        self.processes = []
