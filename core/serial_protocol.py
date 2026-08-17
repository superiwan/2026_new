"""UART output for STM32 puzzle start/rotation/target commands."""

import math

try:
    from maix import pinmap, uart
except ImportError:
    pinmap = uart = None


UART_DEVICE = "/dev/ttyS0"
UART_BAUDRATE = 115200
ACK_TOKEN = "<ok>"
OVER_FRAME = "<over>\r\n"
PIXEL_X_RANGE = (27, 495)
PIXEL_Y_RANGE = (96, 453)


def _pixel_point(point):
    if point is None or len(point) != 2:
        raise ValueError("coordinate must contain x and y")
    return int(float(point[0])), int(float(point[1]))


def is_valid_pixel_point(point):
    try:
        x, y = _pixel_point(point)
    except (TypeError, ValueError, OverflowError):
        return False
    return (PIXEL_X_RANGE[0] <= x <= PIXEL_X_RANGE[1]
            and PIXEL_Y_RANGE[0] <= y <= PIXEL_Y_RANGE[1])


def _clockwise_degree(angle):
    angle = float(angle)
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    return int(round(angle)) % 360


def encode_position_pair(green, degree, red):
    """Encode one piece's start/clockwise rotation/target command."""
    if not is_valid_pixel_point(green):
        raise ValueError("green coordinate outside calibrated A4 range")
    if not is_valid_pixel_point(red):
        raise ValueError("red coordinate outside calibrated A4 range")
    green_x, green_y = _pixel_point(green)
    degree = _clockwise_degree(degree)
    red_x, red_y = _pixel_point(red)
    return "gre:(%d,%d)\ndeg:(%d)\nred:(%d,%d)\n" % (
        green_x, green_y, degree, red_x, red_y)


class PositionSender:
    def __init__(self, serial=None):
        self.serial = serial
        self.receive_buffer = ""

    def open(self):
        if self.serial is not None:
            return self.serial
        if uart is None:
            raise RuntimeError("MaixPy UART is unavailable")
        pinmap.set_pin_function("A17", "UART0_RX")
        pinmap.set_pin_function("A16", "UART0_TX")
        self.serial = uart.UART(UART_DEVICE, UART_BAUDRATE)
        return self.serial

    def poll_ack(self):
        """Consume at most one possibly fragmented ``<ok>`` token."""
        token_index = self.receive_buffer.find(ACK_TOKEN)
        if token_index < 0:
            data = self.open().read()
            if data:
                if isinstance(data, bytes):
                    data = data.decode("ascii", "ignore")
                self.receive_buffer += str(data)
            token_index = self.receive_buffer.find(ACK_TOKEN)
        if token_index < 0:
            # Only a token prefix can be useful on the next UART read.
            self.receive_buffer = self.receive_buffer[-(len(ACK_TOKEN) - 1):]
            return False
        # Retries accumulated before this recognition cycle are duplicates,
        # not permission to advance another piece.
        self.receive_buffer = ""
        return True

    def discard_input(self):
        """Drop ACK retries accumulated during a long recognition cycle."""
        self.receive_buffer = ""
        serial = self.open()
        while serial.read():
            pass

    def send(self, position_pairs):
        serial = None
        frames = []
        for green, degree, red in position_pairs:
            if not (is_valid_pixel_point(green)
                    and is_valid_pixel_point(red)):
                continue
            try:
                frame = encode_position_pair(green, degree, red)
            except (TypeError, ValueError, OverflowError):
                continue
            if serial is None:
                serial = self.open()
            # One write keeps the same piece's three fields adjacent.
            serial.write(frame.encode("ascii"))
            frames.append(frame)
        return frames

    def send_over(self):
        """Notify STM32 that every piece in the current cycle was sent."""
        self.open().write(OVER_FRAME.encode("ascii"))
        return OVER_FRAME
