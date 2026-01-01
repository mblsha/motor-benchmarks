"""Main sweep orchestration for motor parameter testing."""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd

from .bench import Motor, PwmCommand, PointResult, Tachometer, measure_point
from .config import SweepConfig
from .nidec_controller import NidecMotorController, NidecConfig
from .saleae_capture import SaleaeTachConfig, SaleaeTachometer
from .analysis import MotorAnalyzer


class MotorSweep:
    """Orchestrate a complete motor parameter sweep."""

    def __init__(
        self,
        config: SweepConfig,
        motor: Motor | None = None,
        tach: Tachometer | None = None,
    ):
        """Initialize sweep with configuration.

        Args:
            config: Sweep configuration
        """
        self.config = config

        if motor is None:
            nidec_config = NidecConfig(
                pwm_freq=config.pwm_freq,
                invert_duty_cycle=config.invert_duty_cycle,
            )
            motor = NidecMotorController(
                port=config.serial_port,
                baudrate=config.serial_baudrate,
                config=nidec_config,
            )

        if tach is None:
            tach_config = SaleaeTachConfig(
                encoder_channels=config.encoder_channels,
                threshold_volts=config.threshold_volts,
                minimum_pulse_width_samples=config.minimum_pulse_width_samples,
                pulses_per_revolution=config.pulses_per_revolution,
                edge_type=config.edge_type,
                rpm_method=config.rpm_method,
                window_edges=config.window_edges,
                digital_port=config.digital_port,
                channel_b_index=config.channel_b_index,
                signed=config.signed,
                invert_direction=config.invert_direction,
                auto_fallback=config.auto_fallback,
                motor_voltage_channel=config.motor_voltage_channel,
                motor_voltage_name=config.motor_voltage_name,
                analog_voltage_range=config.analog_voltage_range,
                analog_probe_attenuation=config.analog_probe_attenuation,
            )
            tach = SaleaeTachometer(tach_config)

        self.motor = motor
        self.tach = tach
        self.results: list[PointResult] = []

    def run(self) -> Path:
        """Execute the full parameter sweep.

        Returns:
            Path to the results directory
        """
        commands = [
            PwmCommand(duty=duty, freq_hz=self.config.pwm_freq)
            for duty in self.config.get_duty_cycles()
        ]
        return self.run_commands(commands)

    def run_commands(self, commands: list[PwmCommand]) -> Path:
        """Execute a run for the provided commands (sweep or single-point)."""
        # Create timestamped results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.output_dir / f"{self.config.motor_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting sweep: {run_dir}")
        print(f"Motor: {self.config.motor_name}")

        try:
            # Connect to devices
            print("Connecting to motor...")
            self.motor.connect()

            print("Connecting to tachometer...")
            self.tach.connect()

            self.results = []

            for i, cmd in enumerate(commands, 1):
                print(f"\n[{i}/{len(commands)}] Testing duty cycle: {cmd.duty:.2%}")

                result = self._run_single_point(cmd, run_dir)
                self.results.append(result)

                # Save intermediate results
                self._save_summary(run_dir)

            # Generate final analysis
            print("\nGenerating analysis...")
            self._generate_analysis(run_dir)

            print(f"\nSweep complete! Results saved to: {run_dir}")

        finally:
            # Cleanup
            print("\nDisconnecting devices...")
            try:
                self.motor.close()
            finally:
                self.tach.close()

        return run_dir

    def _run_single_point(self, cmd: PwmCommand, run_dir: Path) -> PointResult:
        """Run a single sweep point measurement.

        Args:
            cmd: PWM command to test
            run_dir: Directory to save results

        Returns:
            PointResult with measurement results
        """
        print(f"  Setting motor speed to {cmd.duty:.2%}...")
        print(f"  Settling for {self.config.settle_time}s...")

        capture_dir = run_dir / f"capture_{cmd.duty:.3f}"
        rpm_file = run_dir / f"rpm_{cmd.duty:.3f}.csv"

        result = measure_point(
            motor=self.motor,
            tach=self.tach,
            cmd=cmd,
            settle_s=self.config.settle_time,
            capture_s=self.config.acquisition_duration,
            capture_dir=capture_dir,
            rpm_file=rpm_file,
        )

        print(f"  Mean RPM: {result.mean_rpm:.1f} ± {result.std_rpm:.1f}")
        if result.motor_voltage_mean is not None:
            print(f"  Mean motor voltage: {result.motor_voltage_mean:.3f} V")
        return result

    def _save_summary(self, run_dir: Path) -> None:
        """Save sweep summary and metadata.

        Args:
            run_dir: Directory to save results
        """
        # Save summary CSV
        summary_df = pd.DataFrame(
            [result.summary_dict(run_dir) for result in self.results]
        )
        summary_df.to_csv(run_dir / "summary.csv", index=False)

        # Save metadata
        metadata = {
            'motor_name': self.config.motor_name,
            'timestamp': datetime.now().isoformat(),
            'duty_cycle_start': self.config.duty_cycle_start,
            'duty_cycle_end': self.config.duty_cycle_end,
            'duty_cycle_steps': self.config.duty_cycle_steps,
            'acquisition_duration': self.config.acquisition_duration,
            'settle_time': self.config.settle_time,
            'encoder_channels': self.config.encoder_channels,
            'threshold_volts': self.config.threshold_volts,
            'pulses_per_revolution': self.config.pulses_per_revolution,
            'edge_type': self.config.edge_type,
            'rpm_method': self.config.rpm_method,
            'window_edges': self.config.window_edges,
            'digital_port': self.config.digital_port,
            'channel_b_index': self.config.channel_b_index,
            'signed': self.config.signed,
            'invert_direction': self.config.invert_direction,
            'auto_fallback': self.config.auto_fallback,
            'motor_voltage_channel': self.config.motor_voltage_channel,
            'motor_voltage_name': self.config.motor_voltage_name,
            'analog_voltage_range': self.config.analog_voltage_range,
            'analog_probe_attenuation': str(self.config.analog_probe_attenuation),
        }

        with open(run_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def _generate_analysis(self, run_dir: Path) -> None:
        """Generate plots and reports.

        Args:
            run_dir: Directory containing results
        """
        analyzer = MotorAnalyzer(run_dir)
        summary_df, _ = analyzer.load_sweep_data()

        # Generate efficiency curve
        analyzer.plot_efficiency_curve(
            summary_df,
            run_dir / "efficiency_curve.png"
        )

        # Generate time series plots for each point
        for _, row in summary_df.iterrows():
            duty_cycle = row['duty_cycle']
            rpm_file = run_dir / row['rpm_file']

            if rpm_file.exists():
                rpm_df = pd.read_csv(rpm_file)
                analyzer.plot_time_series(
                    duty_cycle,
                    rpm_df,
                    run_dir / f"timeseries_{duty_cycle:.3f}.png"
                )

        # Generate text report
        report = analyzer.generate_report()

        with open(run_dir / "report.txt", 'w') as f:
            f.write(report)

        print("\n" + report)
