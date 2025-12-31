"""CLI entry point for motor benchmarks."""

import sys
from pathlib import Path

from .config import SweepConfig
from .sweep import MotorSweep


def main():
    """Run a motor parameter sweep with default or custom configuration."""

    # You can customize the configuration here or load from a file
    config = SweepConfig(
        # Sweep parameters
        duty_cycle_start=0.2,
        duty_cycle_end=0.9,
        duty_cycle_steps=8,

        # Timing
        acquisition_duration=5.0,
        settle_time=2.0,

        # Serial settings - UPDATE THESE FOR YOUR SETUP
        serial_port="/dev/ttyACM0",  # Change to your Jumperless port
        serial_baudrate=115200,

        # Nidec motor PWM settings
        pwm_pin=1,  # Jumperless GPIO_1
        pwm_freq=20_000,  # 20kHz PWM (good balance for Nidec)
        invert_duty_cycle=True,  # Nidec: 0.0=max speed, 1.0=stop
        pulse_change_pin=False,  # Enable for timing measurements

        # Saleae MSO settings
        encoder_channels=[0],  # Digital channel(s) for encoder - typically just one channel needed
        threshold_volts=1.65,  # Digital threshold (1.65V for 3.3V logic, 2.5V for 5V logic)

        # Encoder settings
        pulses_per_revolution=1,  # Adjust based on your encoder (e.g., 360 for 360 PPR encoder)
        edge_type='rising',  # Count 'rising', 'falling', or 'both' edges

        # Output
        output_dir=Path("./results"),
        motor_name="nidec-24h"
    )

    print("Motor Benchmark Sweep")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Duty cycle range: {config.duty_cycle_start:.1%} - {config.duty_cycle_end:.1%}")
    print(f"  Steps: {config.duty_cycle_steps}")
    print(f"  Acquisition: {config.acquisition_duration}s (settle: {config.settle_time}s)")
    print(f"  Serial port: {config.serial_port}")
    print(f"  Output: {config.output_dir}")
    print("=" * 60)

    # Run sweep
    sweep = MotorSweep(config)

    try:
        results_dir = sweep.run()
        print(f"\nSuccess! Results available at: {results_dir}")
        return 0
    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during sweep: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
