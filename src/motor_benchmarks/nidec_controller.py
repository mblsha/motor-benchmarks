"""
Nidec-24H motor controller via Jumperless V5 (MicroPython REPL paste mode).

Design:
- A small MicroPython helper is uploaded once per connection.
- Speed changes only call a pre-defined function on the board.
"""

from __future__ import annotations

import time
import textwrap
from dataclasses import dataclass
from typing import Optional

import serial


_PROMPT = b">>> "


class MicroPythonPasteSession:
    """Minimal, more reliable paste-mode executor for MicroPython over serial."""

    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None

    def open(self) -> None:
        if self.ser and self.ser.is_open:
            return

        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,          # small read timeout; we implement our own waiting
                write_timeout=2.0,
            )
            time.sleep(0.5)
            self.interrupt()
            self.flush()
        except Exception as e:
            raise RuntimeError(f"Failed to open serial port {self.port}: {e}") from e

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()

    def flush(self) -> None:
        if not self.ser:
            return
        # More explicit than read_all()
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def interrupt(self) -> None:
        """Try to stop any running code (Ctrl-C twice)."""
        self._write(b"\x03\x03")
        self._drain(0.25)

    def exec_paste(self, code: str, timeout: float = 5.0) -> str:
        """
        Execute code via paste mode (Ctrl-E ... Ctrl-D). Waits until the REPL prompt returns.
        """
        self._ensure_open()

        # Clear any pending output so we start clean.
        self.flush()

        # Enter paste mode
        self._write(b"\x05")  # Ctrl-E
        # Some ports print a paste-mode banner; we don't rely on exact text.
        self._drain(0.15)

        # Send code
        if not code.endswith("\n"):
            code += "\n"
        self._write(code.encode("utf-8"))

        # Exit paste mode and execute
        self._write(b"\x04")  # Ctrl-D

        # Read until prompt comes back
        raw = self._read_until(_PROMPT, timeout=timeout)
        return raw.decode("utf-8", errors="ignore")

    def _read_until(self, needle: bytes, timeout: float) -> bytes:
        self._ensure_open()
        assert self.ser is not None

        deadline = time.monotonic() + timeout
        buf = bytearray()

        while time.monotonic() < deadline:
            chunk = self.ser.read(1024)
            if chunk:
                buf.extend(chunk)
                if needle in buf:
                    return bytes(buf)
            else:
                # no data this tick; keep looping until deadline
                pass

        raise TimeoutError(
            f"Timed out waiting for {needle!r}. Got: {bytes(buf)[-300:]!r}"
        )

    def _drain(self, duration: float) -> bytes:
        self._ensure_open()
        assert self.ser is not None

        deadline = time.monotonic() + duration
        buf = bytearray()
        while time.monotonic() < deadline:
            chunk = self.ser.read(1024)
            if chunk:
                buf.extend(chunk)
        return bytes(buf)

    def _write(self, data: bytes) -> None:
        self._ensure_open()
        assert self.ser is not None
        self.ser.write(data)

    def _ensure_open(self) -> None:
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Serial session not open. Call open() first.")


@dataclass(frozen=True)
class NidecConfig:
    pwm_freq: int = 20_000
    invert_duty_cycle: bool = True
    # If you ever need other pins, keep them in one place:
    change_pin: int = 33
    orange_pwm: int = 25
    green_brake: int = 28
    yellow_dir: int = 29
    white_5v: int = 30


class NidecMotorController:
    """
    Controls a Nidec-24H motor through Jumperless V5 running MicroPython.

    Uploads a helper to the board once per connection; subsequent calls only update PWM duty.
    """

    _DEVICE_HELPER = textwrap.dedent(
        """
        import jumperless as j

        CHANGE_PIN  = {change_pin}
        ORANGE_PWM  = {orange_pwm}
        GREEN_BRAKE = {green_brake}
        YELLOW_DIR  = {yellow_dir}
        WHITE_5V    = {white_5v}

        _PWM_PIN = j.GPIO_1
        _PWM_FREQ = {pwm_freq}

        def _set_pin(pin, high):
            if high:
                j.disconnect(j.GND, pin)
                j.connect(j.TOP_RAIL, pin)
            else:
                j.disconnect(j.TOP_RAIL, pin)
                j.connect(j.GND, pin)

        def nidec_init():
            j.nodes_clear()
            j.dac_set(j.TOP_RAIL, 3.3)

            j.connect(_PWM_PIN, ORANGE_PWM)

            # Static control lines
            _set_pin(GREEN_BRAKE, True)  # brake released
            _set_pin(YELLOW_DIR, True)
            _set_pin(WHITE_5V, True)     # start

        def nidec_set(duty):
            # Toggle for timing measurements
            _set_pin(CHANGE_PIN, True)
            j.pwm(_PWM_PIN, _PWM_FREQ, duty)
            _set_pin(CHANGE_PIN, False)

        nidec_init()
        print("__NIDEC_READY__")
        """
    ).strip() + "\n"

    def __init__(self, port: str, baudrate: int = 115200, config: NidecConfig = NidecConfig()):
        self._mp = MicroPythonPasteSession(port=port, baudrate=baudrate)
        self._cfg = config
        self._ready = False

    def connect(self) -> None:
        self._mp.open()
        out = self._mp.exec_paste(self._DEVICE_HELPER.format(**self._cfg.__dict__), timeout=6.0)
        if "__NIDEC_READY__" not in out:
            raise RuntimeError(f"Device init did not confirm readiness. Output:\n{out}")
        self._ready = True

    def disconnect(self) -> None:
        # Best-effort stop without reinitializing the whole board
        try:
            if self._ready:
                self.stop()
        finally:
            self._ready = False
            self._mp.close()

    def set_speed(self, speed: float) -> None:
        """
        Set motor speed as 0.0..1.0 (human-friendly).
        With invert_duty_cycle=True (Nidec behavior): speed↑ => duty↓.
        """
        self._ensure_ready()

        if not 0.0 <= speed <= 1.0:
            raise ValueError(f"speed must be between 0.0 and 1.0, got {speed}")

        duty = (1.0 - speed) if self._cfg.invert_duty_cycle else speed

        # Unique sentinel that we can check reliably
        code = f"nidec_set({duty:.6f})\nprint('__NIDEC_OK__')\n"
        out = self._mp.exec_paste(code, timeout=3.0)

        if "__NIDEC_OK__" not in out:
            raise RuntimeError(f"Speed update did not confirm. Output:\n{out}")

    def stop(self) -> None:
        # “stop” means speed=0.0 in our API
        self.set_speed(0.0)

    def _ensure_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("Not connected/initialized. Call connect() first.")

    def __enter__(self) -> "NidecMotorController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
