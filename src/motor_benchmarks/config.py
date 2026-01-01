"""Configuration for motor parameter sweeps."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from saleae import mso_api


@dataclass
class SweepConfig:
    """Configuration for a motor parameter sweep.

    Attributes:
        duty_cycle_start: Starting PWM duty cycle (0.0-1.0)
        duty_cycle_end: Ending PWM duty cycle (0.0-1.0)
        duty_cycle_steps: Number of steps in the sweep
        acquisition_duration: How long to capture data at each point (seconds)
        settle_time: Time to wait for motor to stabilize before capture (seconds)
        output_dir: Directory to save results
        motor_name: Name/identifier for the motor being tested
    """

    # Sweep parameters
    duty_cycle_start: float = 0.1
    duty_cycle_end: float = 1.0
    duty_cycle_steps: int = 10

    # Timing
    acquisition_duration: float = 10.0  # seconds
    settle_time: float = 2.0  # seconds

    # Serial/Jumperless settings
    serial_port: str = "/dev/ttyACM0"
    serial_baudrate: int = 115200

    # Nidec motor PWM settings
    pwm_freq: int = 20_000  # PWM frequency in Hz (20kHz works well for Nidec)
    invert_duty_cycle: bool = True  # Nidec: 0=max speed, 1=stop
    pulse_change_pin: bool = False  # Set True to pulse CHANGE_PIN for timing

    # Saleae MSO settings
    # NOTE: All digital channels on the same port share the same threshold voltage
    # If different thresholds needed, channels must be on different ports
    encoder_channels: list[int] = field(default_factory=lambda: [1, 2])  # Digital channels for encoder (CHA, CHB)
    threshold_volts: float = 1.65  # Digital threshold voltage (1.65V for 3.3V logic)
    minimum_pulse_width_samples: Optional[int] = 3  # Glitch filter (in samples, not seconds)
    digital_port: int = 0  # Digital probe port for the encoder channels

    # Encoder settings
    pulses_per_revolution: int = 1  # Number of encoder pulses per motor revolution
    edge_type: str = 'rising'  # Count 'rising', 'falling', or 'both' edges
    rpm_method: str = 'windowed'  # 'windowed' (robust) or 'edge_to_edge' (debug only)
    window_edges: int = 200  # Number of edges per window for windowed RPM
    channel_b_index: Optional[int] = 1  # Optional index into encoder_channels for quadrature
    signed: bool = False  # Signed RPM (quadrature only)
    invert_direction: bool = False  # Flip rotation direction (quadrature only)
    auto_fallback: bool = True  # Auto-fallback to single-channel if CHB unhealthy

    # Analog motor voltage capture (Saleae analog channel)
    motor_voltage_channel: Optional[int] = 0
    motor_voltage_name: str = "motor_voltage"
    analog_voltage_range: float = 20.0
    analog_probe_attenuation: mso_api.ProbeAttenuation = mso_api.ProbeAttenuation.PROBE_10X

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("./results"))
    motor_name: str = "nidec-24h"

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.duty_cycle_start <= 1.0:
            raise ValueError("duty_cycle_start must be between 0.0 and 1.0")
        if not 0.0 <= self.duty_cycle_end <= 1.0:
            raise ValueError("duty_cycle_end must be between 0.0 and 1.0")
        if self.duty_cycle_steps < 1:
            raise ValueError("duty_cycle_steps must be at least 1")
        if self.duty_cycle_steps == 1:
            if self.duty_cycle_start != self.duty_cycle_end:
                raise ValueError("For a single-point run, duty_cycle_start must equal duty_cycle_end")
        elif self.duty_cycle_start >= self.duty_cycle_end:
            raise ValueError("duty_cycle_start must be less than duty_cycle_end")

        if not self.encoder_channels:
            raise ValueError("encoder_channels must include at least one channel")

        if self.channel_b_index is not None:
            if not 0 <= self.channel_b_index < len(self.encoder_channels):
                raise ValueError("channel_b_index must reference encoder_channels")
            if self.channel_b_index == 0:
                raise ValueError("channel_b_index cannot be 0 (channel 0 is used as channel A)")

        if self.motor_voltage_channel is not None and self.motor_voltage_channel < 0:
            raise ValueError("motor_voltage_channel must be >= 0 or None")

        # Ensure output directory exists
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_duty_cycles(self) -> list[float]:
        """Generate list of duty cycle values for the sweep."""
        import numpy as np
        return np.linspace(
            self.duty_cycle_start,
            self.duty_cycle_end,
            self.duty_cycle_steps
        ).tolist()
