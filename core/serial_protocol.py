"""Unified CRC-protected UART output for all puzzle solvers."""

try:
    from maix import pinmap, uart
except ImportError:
    pinmap = uart = None


UART_DEVICE = "/dev/ttyS1"
UART_BAUDRATE = 115200


def crc8_ascii(payload):
    crc = 0
    for value in payload.encode("ascii"):
        crc ^= value
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def encode_action(action):
    payload = "A,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f" % (
        action.piece_id,
        action.pick_x,
        action.pick_y,
        action.pick_angle,
        action.place_x,
        action.place_y,
        action.place_angle,
    )
    return "$%s*%02X\r\n" % (payload, crc8_ascii(payload))


class ActionSender:
    def __init__(self, serial=None):
        self.serial = serial

    def open(self):
        if self.serial is not None:
            return self.serial
        if uart is None:
            raise RuntimeError("MaixPy UART is unavailable")
        pinmap.set_pin_function("A18", "UART1_RX")
        pinmap.set_pin_function("A19", "UART1_TX")
        self.serial = uart.UART(UART_DEVICE, UART_BAUDRATE)
        return self.serial

    def send(self, actions):
        serial = self.open()
        frames = []
        for action in actions:
            frame = encode_action(action)
            serial.write(frame.encode("ascii"))
            frames.append(frame)
        return frames
