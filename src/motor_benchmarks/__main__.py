"""CLI entry point for motor benchmarks."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

from .bench import PwmCommand, rpm_stats
from .config import SweepConfig
from .nidec_controller import NidecConfig, NidecMotorController
from .saleae_capture import SaleaeTachConfig, SaleaeTachometer
from .sweep import MotorSweep


def _parse_int_list(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers")
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers") from exc


def _add_motor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--serial-port",
        default="/dev/cu.usbmodemJLV5port5",
        help="Jumperless V5 MicroPython serial port",
    )
    parser.add_argument("--serial-baudrate", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--pwm-freq", type=int, default=20_000, help="PWM frequency in Hz")
    parser.add_argument("--no-invert", action="store_true", help="Disable duty-cycle inversion")


def _add_tach_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--encoder-channels",
        type=_parse_int_list,
        default=[1, 2],
        help="Comma-separated digital channel list (e.g. 1,2 for CHA/CHB)",
    )
    parser.add_argument("--threshold-volts", type=float, default=1.65, help="Digital threshold voltage")
    parser.add_argument(
        "--minimum-pulse-width-samples",
        type=int,
        default=3,
        help="Glitch filter in samples (0 to disable)",
    )
    parser.add_argument("--digital-port", type=int, default=0, help="Saleae digital port")
    parser.add_argument("--pulses-per-revolution", type=int, default=1, help="Encoder PPR")
    parser.add_argument("--edge-type", choices=["rising", "falling", "both"], default="rising")
    parser.add_argument("--rpm-method", choices=["windowed", "edge_to_edge"], default="windowed")
    parser.add_argument("--window-edges", type=int, default=200, help="Edges per window for RPM")
    parser.add_argument("--channel-b-index", type=int, default=1, help="Index into encoder-channels for CHB")
    parser.add_argument("--signed", action="store_true", help="Return signed RPM (quadrature)")
    parser.add_argument("--invert-direction", action="store_true", help="Invert quadrature direction")
    parser.add_argument("--no-auto-fallback", action="store_true", help="Disable quadrature auto-fallback")
    parser.add_argument("--motor-voltage-channel", type=int, default=0, help="Analog channel for motor voltage")
    parser.add_argument("--no-motor-voltage", action="store_true", help="Disable motor voltage capture")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--settle-s", type=float, default=2.0, help="Settle time before capture")
    parser.add_argument("--capture-s", type=float, default=10.0, help="Capture duration per point")
    parser.add_argument("--output-dir", type=Path, default=Path("./results"), help="Results directory")
    parser.add_argument("--motor-name", default="nidec-24h", help="Motor name label")


def _build_sweep_config(
    args: argparse.Namespace,
    *,
    duty_start: float,
    duty_end: float,
    duty_steps: int,
) -> SweepConfig:
    motor_voltage_channel = None if args.no_motor_voltage else args.motor_voltage_channel
    min_pulse = args.minimum_pulse_width_samples
    if min_pulse == 0:
        min_pulse = None
    return SweepConfig(
        duty_cycle_start=duty_start,
        duty_cycle_end=duty_end,
        duty_cycle_steps=duty_steps,
        acquisition_duration=args.capture_s,
        settle_time=args.settle_s,
        serial_port=args.serial_port,
        serial_baudrate=args.serial_baudrate,
        pwm_freq=args.pwm_freq,
        invert_duty_cycle=not args.no_invert,
        encoder_channels=args.encoder_channels,
        threshold_volts=args.threshold_volts,
        minimum_pulse_width_samples=min_pulse,
        digital_port=args.digital_port,
        pulses_per_revolution=args.pulses_per_revolution,
        edge_type=args.edge_type,
        rpm_method=args.rpm_method,
        window_edges=args.window_edges,
        channel_b_index=args.channel_b_index,
        signed=args.signed,
        invert_direction=args.invert_direction,
        auto_fallback=not args.no_auto_fallback,
        motor_voltage_channel=motor_voltage_channel,
        output_dir=args.output_dir,
        motor_name=args.motor_name,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Motor benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    point_parser = subparsers.add_parser("point", help="Measure a single datapoint")
    _add_motor_args(point_parser)
    _add_tach_args(point_parser)
    _add_run_args(point_parser)
    point_parser.add_argument("--duty", type=float, default=0.5, help="Duty cycle 0..1")

    sweep_parser = subparsers.add_parser("sweep", help="Run a duty-cycle sweep")
    _add_motor_args(sweep_parser)
    _add_tach_args(sweep_parser)
    _add_run_args(sweep_parser)
    sweep_parser.add_argument("--duty-start", type=float, default=0.2, help="Starting duty cycle")
    sweep_parser.add_argument("--duty-end", type=float, default=0.9, help="Ending duty cycle")
    sweep_parser.add_argument("--duty-steps", type=int, default=8, help="Number of steps in sweep")

    motor_parser = subparsers.add_parser("motor-test", help="Smoke-test motor driver only")
    _add_motor_args(motor_parser)
    motor_parser.add_argument("--duty", type=float, default=0.3, help="Duty cycle 0..1")
    motor_parser.add_argument("--duration-s", type=float, default=2.0, help="Run duration in seconds")

    tach_parser = subparsers.add_parser("tach-test", help="Smoke-test tachometer only")
    _add_tach_args(tach_parser)
    tach_parser.add_argument("--capture-s", type=float, default=2.0, help="Capture duration in seconds")
    tach_parser.add_argument("--output-dir", type=Path, default=Path("./results"), help="Results directory")

    args = parser.parse_args()

    try:
        if args.command == "point":
            config = _build_sweep_config(
                args,
                duty_start=args.duty,
                duty_end=args.duty,
                duty_steps=1,
            )
            sweep = MotorSweep(config)
            results_dir = sweep.run()
            print(f"\nSuccess! Results available at: {results_dir}")
            return 0

        if args.command == "sweep":
            config = _build_sweep_config(
                args,
                duty_start=args.duty_start,
                duty_end=args.duty_end,
                duty_steps=args.duty_steps,
            )
            sweep = MotorSweep(config)
            results_dir = sweep.run()
            print(f"\nSuccess! Results available at: {results_dir}")
            return 0

        if args.command == "motor-test":
            config = NidecConfig(
                pwm_freq=args.pwm_freq,
                invert_duty_cycle=not args.no_invert,
            )
            motor = NidecMotorController(
                port=args.serial_port,
                baudrate=args.serial_baudrate,
                config=config,
            )
            cmd = PwmCommand(duty=args.duty, freq_hz=args.pwm_freq)
            try:
                motor.connect()
                print("Motor health:", motor.healthcheck())
                motor.apply(cmd)
                time.sleep(args.duration_s)
                motor.stop()
            finally:
                motor.close()
            return 0

        if args.command == "tach-test":
            min_pulse = args.minimum_pulse_width_samples
            if min_pulse == 0:
                min_pulse = None
            motor_voltage_channel = None if args.no_motor_voltage else args.motor_voltage_channel
            tach_config = SaleaeTachConfig(
                encoder_channels=args.encoder_channels,
                threshold_volts=args.threshold_volts,
                minimum_pulse_width_samples=min_pulse,
                pulses_per_revolution=args.pulses_per_revolution,
                edge_type=args.edge_type,
                rpm_method=args.rpm_method,
                window_edges=args.window_edges,
                digital_port=args.digital_port,
                channel_b_index=args.channel_b_index,
                signed=args.signed,
                invert_direction=args.invert_direction,
                auto_fallback=not args.no_auto_fallback,
                motor_voltage_channel=motor_voltage_channel,
            )
            tach = SaleaeTachometer(tach_config)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = args.output_dir / f"tach_test_{timestamp}"
            capture_dir = run_dir / "capture"
            rpm_file = run_dir / "rpm.csv"

            try:
                tach.connect()
                print("Tach health:", tach.healthcheck())
                measurement = tach.measure(
                    duration_s=args.capture_s,
                    capture_dir=capture_dir,
                    rpm_file=rpm_file,
                )
                stats = rpm_stats(measurement.rpm_df)
                print(
                    "Tach stats:",
                    f"mean={stats['mean_rpm']:.1f}",
                    f"std={stats['std_rpm']:.1f}",
                    f"n={stats['n_rpm']}",
                )
                if measurement.motor_voltage_mean is not None:
                    print(f"Motor voltage mean: {measurement.motor_voltage_mean:.3f} V")
                print(f"Capture saved to: {measurement.capture_dir}")
                print(f"RPM CSV saved to: {measurement.rpm_file}")
            finally:
                tach.close()
            return 0

        parser.error(f"Unhandled command: {args.command}")
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 1
    except Exception as exc:
        print(f"\nError: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
