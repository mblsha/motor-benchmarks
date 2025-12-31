"""Saleae MSO API interface for encoder data capture."""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from saleae import mso_api


class SaleaeCapture:
    """Interface to Saleae Logic MSO for capturing encoder signals using MSO API."""

    def __init__(self, serial_number: Optional[str] = None):
        """Initialize connection to Saleae Logic MSO device.

        Args:
            serial_number: Optional serial number of specific MSO device to connect to.
                          If None, connects to the first available device.
        """
        self.serial_number = serial_number
        self._mso: Optional[mso_api.MSO] = None

    def connect(self) -> None:
        """Connect to Saleae Logic MSO device.

        Raises:
            RuntimeError: If no MSO device is found
        """
        self._mso = mso_api.MSO(serial_number=self.serial_number)
        print("Connected to Saleae MSO")

    def capture_encoder_data(
        self,
        duration: float,
        channels: list[int],
        save_dir: Path,
        channel_names: Optional[list[str]] = None,
        threshold_volts: float = 1.65,
        port: int = 0
    ) -> mso_api.Capture:
        """Capture encoder data from specified digital channels.

        Args:
            duration: Capture duration in seconds
            channels: List of digital channel numbers to capture
            save_dir: Directory to save the capture data
            channel_names: Optional list of names for the channels (defaults to "encoder_0", "encoder_1", etc.)
            threshold_volts: Digital threshold voltage (default 1.65V for 3.3V logic)
            port: Digital probe port number (default 0)

        Returns:
            Capture object with digital data

        Raises:
            RuntimeError: If not connected to MSO device
        """
        if not self._mso:
            raise RuntimeError("Not connected to Saleae MSO. Call connect() first.")

        # Generate default channel names if not provided
        if channel_names is None:
            channel_names = [f"encoder_{i}" for i in range(len(channels))]
        elif len(channel_names) != len(channels):
            raise ValueError("Number of channel names must match number of channels")

        # Configure digital channels
        enabled_channels = [
            mso_api.DigitalChannel(
                channel=ch,
                name=name,
                port=port,
                threshold_volts=threshold_volts
            )
            for ch, name in zip(channels, channel_names)
        ]

        # Create capture configuration
        capture_config = mso_api.CaptureConfig(
            enabled_channels=enabled_channels,
            capture_settings=mso_api.TimedCapture(capture_length_seconds=duration)
        )

        # Execute capture
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Starting capture ({duration}s)...")
        capture = self._mso.capture(capture_config, save_dir=save_dir)
        print(f"  Capture complete. Data saved to: {save_dir}")

        return capture

    def calculate_rpm_from_encoder(
        self,
        digital_data: mso_api.capture.DigitalData,
        pulses_per_revolution: int = 1,
        edge_type: str = 'rising'
    ) -> pd.DataFrame:
        """Calculate RPM from encoder transition times.

        Args:
            digital_data: DigitalData object from a capture
            pulses_per_revolution: Number of encoder pulses per motor revolution
            edge_type: 'rising', 'falling', or 'both' to count transitions

        Returns:
            DataFrame with time and RPM columns
        """
        # Get transition times from the digital data
        transition_times = digital_data.transition_times

        if len(transition_times) < 2:
            return pd.DataFrame({'time': [], 'rpm': []})

        # Filter for rising or falling edges if specified
        if edge_type == 'rising':
            # Rising edges are at even indices (0, 2, 4...) if initial_state is False
            # or odd indices (1, 3, 5...) if initial_state is True
            if digital_data.initial_state:
                # First transition is falling, so rising edges are at odd indices
                edge_times = transition_times[1::2]
            else:
                # First transition is rising, so rising edges are at even indices
                edge_times = transition_times[0::2]
        elif edge_type == 'falling':
            # Opposite of rising
            if digital_data.initial_state:
                # First transition is falling, so falling edges are at even indices
                edge_times = transition_times[0::2]
            else:
                # First transition is rising, so falling edges are at odd indices
                edge_times = transition_times[1::2]
        else:  # 'both'
            edge_times = transition_times

        if len(edge_times) < 2:
            return pd.DataFrame({'time': [], 'rpm': []})

        # Calculate time between consecutive edges
        time_diffs = np.diff(edge_times)

        # Calculate RPM from period
        # RPM = (60 seconds/minute) / (time_per_pulse * pulses_per_revolution)
        rpm_values = 60.0 / (time_diffs * pulses_per_revolution)

        # Return DataFrame with edge times (excluding first) and corresponding RPM
        return pd.DataFrame({
            'time': edge_times[1:],
            'rpm': rpm_values
        })

    def save_rpm_data(
        self,
        capture: mso_api.Capture,
        output_file: Path,
        channel_name: str = "encoder_0",
        pulses_per_revolution: int = 1,
        edge_type: str = 'rising'
    ) -> pd.DataFrame:
        """Extract RPM data from capture and save to CSV.

        Args:
            capture: Capture object from capture_encoder_data
            output_file: Path to save RPM CSV file
            channel_name: Name of the encoder channel to analyze
            pulses_per_revolution: Number of encoder pulses per motor revolution
            edge_type: 'rising', 'falling', or 'both' to count transitions

        Returns:
            DataFrame with time and RPM columns
        """
        # Get digital data for the specified channel
        if channel_name not in capture.digital_data:
            available = list(capture.digital_data.keys())
            raise ValueError(
                f"Channel '{channel_name}' not found in capture. "
                f"Available channels: {available}"
            )

        digital_data = capture.digital_data[channel_name]

        # Calculate RPM
        rpm_df = self.calculate_rpm_from_encoder(
            digital_data,
            pulses_per_revolution=pulses_per_revolution,
            edge_type=edge_type
        )

        # Save to CSV
        output_file = Path(output_file)
        rpm_df.to_csv(output_file, index=False)
        print(f"  RPM data saved to: {output_file}")

        return rpm_df

    def get_statistics(self, rpm_df: pd.DataFrame) -> dict:
        """Calculate statistical metrics for RPM data.

        Args:
            rpm_df: DataFrame with 'rpm' column

        Returns:
            Dictionary of statistics
        """
        if rpm_df.empty or 'rpm' not in rpm_df.columns:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'cv': 0.0
            }

        rpm = rpm_df['rpm'].dropna()
        mean_rpm = rpm.mean()
        std_rpm = rpm.std()

        return {
            'mean': float(mean_rpm),
            'std': float(std_rpm),
            'min': float(rpm.min()),
            'max': float(rpm.max()),
            'median': float(rpm.median()),
            'cv': float(std_rpm / mean_rpm * 100) if mean_rpm > 0 else 0.0
        }

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # MSO API handles cleanup automatically
        pass
