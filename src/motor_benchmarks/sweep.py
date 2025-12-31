"""Main sweep orchestration for motor parameter testing."""

import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

from .config import SweepConfig
from .nidec_controller import NidecMotorController, NidecConfig
from .saleae_capture import SaleaeCapture
from .analysis import MotorAnalyzer


class MotorSweep:
    """Orchestrate a complete motor parameter sweep."""

    def __init__(self, config: SweepConfig):
        """Initialize sweep with configuration.

        Args:
            config: Sweep configuration
        """
        self.config = config

        # Create Nidec configuration from sweep config
        nidec_config = NidecConfig(
            pwm_freq=config.pwm_freq,
            invert_duty_cycle=config.invert_duty_cycle
        )

        self.motor = NidecMotorController(
            port=config.serial_port,
            baudrate=config.serial_baudrate,
            config=nidec_config
        )
        self.saleae = SaleaeCapture()
        self.results: list[dict] = []

    def run(self) -> Path:
        """Execute the full parameter sweep.

        Returns:
            Path to the results directory
        """
        # Create timestamped results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.output_dir / f"{self.config.motor_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting sweep: {run_dir}")
        print(f"Motor: {self.config.motor_name}")

        # Connect to devices
        print("Connecting to Nidec motor via Jumperless...")
        self.motor.connect()

        print("Connecting to Saleae MSO...")
        self.saleae.connect()

        try:
            # Run sweep
            duty_cycles = self.config.get_duty_cycles()

            for i, duty_cycle in enumerate(duty_cycles, 1):
                print(f"\n[{i}/{len(duty_cycles)}] Testing duty cycle: {duty_cycle:.2%}")

                result = self._run_single_point(duty_cycle, run_dir)
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
            self.motor.disconnect()

        return run_dir

    def _run_single_point(self, duty_cycle: float, run_dir: Path) -> dict:
        """Run a single sweep point measurement.

        Args:
            duty_cycle: PWM duty cycle to test
            run_dir: Directory to save results

        Returns:
            Dictionary with measurement results
        """
        # Set motor speed (duty_cycle is the speed in our API)
        print(f"  Setting motor speed to {duty_cycle:.2%}...")
        self.motor.set_speed(duty_cycle)

        # Wait for motor to stabilize
        print(f"  Settling for {self.config.settle_time}s...")
        time.sleep(self.config.settle_time)

        # Capture encoder data using MSO API
        capture_dir = run_dir / f"capture_{duty_cycle:.3f}"

        capture = self.saleae.capture_encoder_data(
            duration=self.config.acquisition_duration,
            channels=self.config.encoder_channels,
            save_dir=capture_dir,
            channel_names=[f"encoder_{i}" for i in range(len(self.config.encoder_channels))],
            threshold_volts=self.config.threshold_volts,
            port=0  # All digital channels on port 0
        )

        # Calculate and save RPM data
        print("  Analyzing data...")
        rpm_file = run_dir / f"rpm_{duty_cycle:.3f}.csv"

        rpm_df = self.saleae.save_rpm_data(
            capture=capture,
            output_file=rpm_file,
            channel_name="encoder_0",  # Use first encoder channel
            pulses_per_revolution=self.config.pulses_per_revolution,
            edge_type=self.config.edge_type
        )

        # Calculate statistics
        stats = self.saleae.get_statistics(rpm_df)

        print(f"  Mean RPM: {stats['mean']:.1f} ± {stats['std']:.1f}")

        return {
            'duty_cycle': duty_cycle,
            'capture_dir': str(capture_dir.relative_to(run_dir)),
            'rpm_file': str(rpm_file.relative_to(run_dir)),
            **{f'{k}_rpm': v for k, v in stats.items()}
        }

    def _save_summary(self, run_dir: Path) -> None:
        """Save sweep summary and metadata.

        Args:
            run_dir: Directory to save results
        """
        # Save summary CSV
        summary_df = pd.DataFrame(self.results)
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
