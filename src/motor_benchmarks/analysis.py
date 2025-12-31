"""Analysis tools for motor benchmark data."""

import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class MotorAnalyzer:
    """Analyze motor benchmark sweep results."""

    def __init__(self, results_dir: Path):
        """Initialize analyzer with results directory.

        Args:
            results_dir: Directory containing sweep results
        """
        self.results_dir = Path(results_dir)

    def load_sweep_data(self) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Load all sweep data and metadata.

        Returns:
            Tuple of (summary DataFrame, metadata dict)
        """
        summary_file = self.results_dir / "summary.csv"
        metadata_file = self.results_dir / "metadata.json"

        if not summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_file}")

        summary_df = pd.read_csv(summary_file)

        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        return summary_df, metadata

    def calculate_rpm_statistics(self, rpm_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistical metrics for RPM data.

        Args:
            rpm_data: DataFrame with 'rpm' column

        Returns:
            Dictionary of statistics
        """
        if rpm_data.empty or 'rpm' not in rpm_data.columns:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'cv': 0.0  # Coefficient of variation
            }

        rpm = rpm_data['rpm'].dropna()

        mean_rpm = rpm.mean()
        std_rpm = rpm.std()

        return {
            'mean': float(mean_rpm),
            'std': float(std_rpm),
            'min': float(rpm.min()),
            'max': float(rpm.max()),
            'median': float(rpm.median()),
            'cv': float(std_rpm / mean_rpm * 100) if mean_rpm > 0 else 0.0  # % variation
        }

    def plot_time_series(
        self,
        duty_cycle: float,
        rpm_data: pd.DataFrame,
        output_file: Path
    ) -> None:
        """Plot RPM over time for a single sweep point.

        Args:
            duty_cycle: PWM duty cycle for this measurement
            rpm_data: DataFrame with 'time' and 'rpm' columns
            output_file: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if not rpm_data.empty and 'time' in rpm_data.columns and 'rpm' in rpm_data.columns:
            ax.plot(rpm_data['time'], rpm_data['rpm'], 'b-', linewidth=0.5, alpha=0.7)

            # Add mean line
            mean_rpm = rpm_data['rpm'].mean()
            ax.axhline(mean_rpm, color='r', linestyle='--', label=f'Mean: {mean_rpm:.1f} RPM')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('RPM')
        ax.set_title(f'Motor RPM vs Time (Duty Cycle: {duty_cycle:.2%})')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()

    def plot_efficiency_curve(
        self,
        summary_df: pd.DataFrame,
        output_file: Path
    ) -> None:
        """Plot RPM vs duty cycle efficiency curve.

        Args:
            summary_df: DataFrame with sweep summary data
            output_file: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot mean RPM with error bars
        ax1.errorbar(
            summary_df['duty_cycle'] * 100,
            summary_df['mean_rpm'],
            yerr=summary_df['std_rpm'],
            fmt='o-',
            capsize=5,
            label='Mean ± Std'
        )
        ax1.fill_between(
            summary_df['duty_cycle'] * 100,
            summary_df['min_rpm'],
            summary_df['max_rpm'],
            alpha=0.2,
            label='Min-Max Range'
        )

        ax1.set_xlabel('Duty Cycle (%)')
        ax1.set_ylabel('RPM')
        ax1.set_title('Motor Speed vs PWM Duty Cycle')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot coefficient of variation
        ax2.plot(
            summary_df['duty_cycle'] * 100,
            summary_df['cv_rpm'],
            'o-',
            color='orange'
        )
        ax2.set_xlabel('Duty Cycle (%)')
        ax2.set_ylabel('Coefficient of Variation (%)')
        ax2.set_title('Speed Stability vs PWM Duty Cycle')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()

    def generate_report(self) -> str:
        """Generate a text summary report of the sweep results.

        Returns:
            Formatted report string
        """
        summary_df, metadata = self.load_sweep_data()

        report = []
        report.append("=" * 60)
        report.append("MOTOR BENCHMARK SWEEP REPORT")
        report.append("=" * 60)
        report.append("")

        if metadata:
            report.append("Configuration:")
            report.append(f"  Motor: {metadata.get('motor_name', 'Unknown')}")
            report.append(f"  Sweep: {metadata.get('duty_cycle_start', 0):.1%} to "
                         f"{metadata.get('duty_cycle_end', 1):.1%} "
                         f"({metadata.get('duty_cycle_steps', 0)} steps)")
            report.append(f"  Acquisition: {metadata.get('acquisition_duration', 0):.1f}s "
                         f"(settle: {metadata.get('settle_time', 0):.1f}s)")
            report.append("")

        report.append("Results Summary:")
        report.append(f"  {'Duty Cycle':<12} {'Mean RPM':<12} {'Std RPM':<12} {'CV %':<12}")
        report.append("  " + "-" * 50)

        for _, row in summary_df.iterrows():
            report.append(f"  {row['duty_cycle']:>11.1%} "
                         f"{row['mean_rpm']:>11.1f} "
                         f"{row['std_rpm']:>11.1f} "
                         f"{row['cv_rpm']:>11.2f}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)
