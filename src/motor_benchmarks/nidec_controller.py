"""Nidec-24H motor controller via Jumperless V5 using paste mode."""

import time
import serial


class NidecMotorController:
    """Control Nidec-24H motor via Jumperless V5.

    Sends complete initialization script as a template for each PWM change.
    Uses pyboard.py to communicate with MicroPython.
    """

    # Nidec initialization script template
    # Template parameters: {pwm_pin}, {pwm_freq}, {duty_cycle}
    NIDEC_INIT_TEMPLATE = """
import jumperless as j
import time

print("nidec start")

pwm_pin = j.GPIO_{pwm_pin}
pwm_freq = {pwm_freq}

def set_pin(pin, high):
    if high:
        j.disconnect(j.GND, pin)
        j.connect(j.TOP_RAIL, pin)
    else:
        j.disconnect(j.TOP_RAIL, pin)
        j.connect(j.GND, pin)

CHANGE_PIN  = 33  # to measure how long pwm-change to encoder takes to propagate
ORANGE_PWM  = 25  # Speed command input
GREEN_BRAKE = 28  # Brake: High-Released
YELLOW_DIR  = 29  # Direction
WHITE_5V    = 30  # Start/stop

j.nodes_clear()
j.dac_set(j.TOP_RAIL, 3.3)

j.connect(pwm_pin, ORANGE_PWM)

set_pin(CHANGE_PIN, True)
j.pwm(pwm_pin, pwm_freq, {duty_cycle})

set_pin(GREEN_BRAKE, True) # brake
set_pin(YELLOW_DIR, True)
set_pin(WHITE_5V, True) # start
set_pin(CHANGE_PIN, False)

print('done')
"""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        pwm_pin: int = 1,  # GPIO_1
        pwm_freq: int = 20_000,
        invert_duty_cycle: bool = True
    ):
        """Initialize Nidec motor controller.

        Args:
            port: Serial port path (e.g., '/dev/ttyACM0')
            baudrate: Serial baud rate
            pwm_pin: Jumperless GPIO pin for PWM output
            pwm_freq: PWM frequency in Hz (20kHz works well)
            invert_duty_cycle: If True, 0=max speed, 1=stop (Nidec behavior)
        """
        self.port = port
        self.baudrate = baudrate
        self.pwm_pin = pwm_pin
        self.pwm_freq = pwm_freq
        self.invert_duty_cycle = invert_duty_cycle
        self._serial = None

    def connect(self) -> None:
        """Connect to Jumperless via serial."""
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=2.0
            )
            time.sleep(0.5)

            # Send Ctrl-C to interrupt any running code
            self._serial.write(b'\x03')
            time.sleep(0.2)
            self._serial.read_all()  # Clear buffer

            print(f"  Connected to Jumperless on {self.port}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Jumperless on {self.port}: {e}")

    def disconnect(self) -> None:
        """Stop motor and close connection."""
        if self._serial and self._serial.is_open:
            try:
                # Stop motor (duty cycle 1.0 = stopped for inverted mode)
                stop_duty = 1.0 if self.invert_duty_cycle else 0.0
                self.set_pwm_duty_cycle(stop_duty)
                time.sleep(0.3)
            except:
                pass  # Best effort to stop
            finally:
                try:
                    self._serial.close()
                except:
                    pass

    def _exec(self, code: str) -> str:
        """Execute MicroPython code using paste mode (Ctrl-E).

        Args:
            code: Python code to execute (can be multi-line)

        Returns:
            Output from the code execution
        """
        if not self._serial or not self._serial.is_open:
            raise RuntimeError("Not connected to Jumperless. Call connect() first.")

        try:
            # Clear any existing output
            self._serial.read_all()

            # Enter paste mode (Ctrl-E)
            self._serial.write(b'\x05')
            time.sleep(0.3)
            # Read and discard paste mode banner
            self._serial.read_all()

            # Send the code (preserve original formatting including indentation)
            self._serial.write(code.encode('utf-8'))
            self._serial.write(b'\n')  # Extra newline
            time.sleep(0.3)

            # Exit paste mode and execute (Ctrl-D)
            self._serial.write(b'\x04')

            # Wait for execution to complete
            time.sleep(1.5)

            # Read all output
            output = self._serial.read_all().decode('utf-8', errors='ignore')
            return output
        except Exception as e:
            raise RuntimeError(f"Execution failed: {e}")

    def set_pwm_duty_cycle(self, duty_cycle: float) -> None:
        """Set motor PWM duty cycle by sending complete initialization script.

        Args:
            duty_cycle: Duty cycle from 0.0 to 1.0
                       If invert_duty_cycle=True: 0.0=max speed, 1.0=stopped
                       If invert_duty_cycle=False: 1.0=max speed, 0.0=stopped
        """
        if not 0.0 <= duty_cycle <= 1.0:
            raise ValueError(f"Duty cycle must be between 0.0 and 1.0, got {duty_cycle}")

        # Apply inversion if needed (Nidec: low duty = fast)
        actual_duty = (1.0 - duty_cycle) if self.invert_duty_cycle else duty_cycle

        # Expand template with parameters
        script = self.NIDEC_INIT_TEMPLATE.format(
            pwm_pin=self.pwm_pin,
            pwm_freq=self.pwm_freq,
            duty_cycle=f"{actual_duty:.6f}"
        )

        # Send script via pyboard
        result = self._exec(script)

        # Verify completion
        if "done" not in result:
            raise RuntimeError(f"Motor initialization failed. Output: {result}")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
