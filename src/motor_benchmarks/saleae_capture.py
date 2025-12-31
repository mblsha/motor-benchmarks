"""Saleae MSO API interface for encoder data capture."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal, Tuple
import numpy as np
import pandas as pd
from saleae import mso_api

EdgeType = Literal["rising", "falling", "both"]
DecodeMode = Literal["x1", "x2", "x4"]


def _edge_times(digital_data, edge_type: EdgeType) -> np.ndarray:
    """
    Return transition times for the requested edge type.

    Saleae digital data is an initial state + monotonically increasing transition times.
    We infer edge polarity from the state before each transition.
    """
    tt = np.asarray(digital_data.transition_times, dtype=float)
    if tt.size == 0:
        return tt

    # State *before* each transition index i is initial_state toggled i times.
    idx = np.arange(tt.size, dtype=np.int64)
    before = np.logical_xor(bool(digital_data.initial_state), (idx & 1).astype(bool))

    # If state-before is 0, the transition is rising (0->1). If 1, it's falling (1->0).
    if edge_type == "rising":
        return tt[~before]
    if edge_type == "falling":
        return tt[before]
    return tt


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

    def capture_mixed_data(
        self,
        duration: float,
        save_dir: Path,
        digital_channels: Optional[list[int]] = None,
        digital_names: Optional[list[str]] = None,
        digital_port: int = 0,
        threshold_volts: float = 1.65,  # Default for 3.3V logic (measure your encoder voltage!)
        analog_channels: Optional[list[int]] = None,
        analog_names: Optional[list[str]] = None,
        analog_voltage_range: float = 20.0,
        analog_probe_attenuation: mso_api.ProbeAttenuation = mso_api.ProbeAttenuation.PROBE_10X
    ) -> mso_api.Capture:
        """Capture mixed analog and digital data.

        Args:
            duration: Capture duration in seconds
            save_dir: Directory to save the capture data
            digital_channels: Optional list of digital channel numbers to capture
            digital_names: Optional list of names for digital channels
            digital_port: Digital probe port number (default 0)
            threshold_volts: Digital threshold voltage (default 1.65V for 3.3V logic)
            analog_channels: Optional list of analog channel numbers to capture
            analog_names: Optional list of names for analog channels
            analog_voltage_range: Voltage range for analog channels (default 20.0V for 10X probe)
            analog_probe_attenuation: Probe attenuation setting (default PROBE_10X)

        Returns:
            Capture object with analog and/or digital data

        Raises:
            RuntimeError: If not connected to MSO device
        """
        if not self._mso:
            raise RuntimeError("Not connected to Saleae MSO. Call connect() first.")

        enabled_channels = []

        # Configure digital channels
        if digital_channels:
            if digital_names is None:
                digital_names = [f"digital_{i}" for i in range(len(digital_channels))]
            elif len(digital_names) != len(digital_channels):
                raise ValueError("Number of digital names must match number of digital channels")

            for ch, name in zip(digital_channels, digital_names):
                enabled_channels.append(
                    mso_api.DigitalChannel(
                        channel=ch,
                        name=name,
                        port=digital_port,
                        threshold_volts=threshold_volts
                    )
                )

        # Configure analog channels
        if analog_channels:
            if analog_names is None:
                analog_names = [f"analog_{i}" for i in range(len(analog_channels))]
            elif len(analog_names) != len(analog_channels):
                raise ValueError("Number of analog names must match number of analog channels")

            for ch, name in zip(analog_channels, analog_names):
                enabled_channels.append(
                    mso_api.AnalogChannel(
                        channel=ch,
                        name=name,
                        voltage_range=analog_voltage_range,
                        probe_attenuation=analog_probe_attenuation
                    )
                )

        if not enabled_channels:
            raise ValueError("Must specify at least one digital or analog channel")

        # Create capture configuration
        capture_config = mso_api.CaptureConfig(
            enabled_channels=enabled_channels,
            analog_settings=mso_api.AnalogSettings(sample_rate=1e6) if analog_channels else None,
            capture_settings=mso_api.TimedCapture(capture_length_seconds=duration)
        )

        # Execute capture
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Starting capture ({duration}s)...")
        capture = self._mso.capture(capture_config, save_dir=save_dir)
        print(f"  Capture complete. Data saved to: {save_dir}")

        return capture

    def capture_encoder_data(
        self,
        duration: float,
        channels: list[int],
        save_dir: Path,
        channel_names: Optional[list[str]] = None,
        threshold_volts: float = 1.65,  # Default for 3.3V logic
        port: int = 0
    ) -> mso_api.Capture:
        """Capture encoder data from digital channels (legacy method).

        Args:
            duration: Capture duration in seconds
            channels: List of digital channel numbers to capture
            save_dir: Directory to save the capture data
            channel_names: Optional list of names for the channels
            threshold_volts: Digital threshold voltage (default 2.5V for 5V logic)
            port: Digital probe port number (default 0)

        Returns:
            Capture object with digital data
        """
        return self.capture_mixed_data(
            duration=duration,
            save_dir=save_dir,
            digital_channels=channels,
            digital_names=channel_names,
            digital_port=port,
            threshold_volts=threshold_volts
        )

    def calculate_rpm_from_encoder(
        self,
        digital_data,
        pulses_per_revolution: int = 100,   # Nidec 24H is 100 PPR
        edge_type: EdgeType = "rising",
    ) -> pd.DataFrame:
        """
        Calculate RPM from a single encoder channel.

        IMPORTANT:
          - pulses_per_revolution is PPR (cycles per rev on one channel).
          - If edge_type == "both", we count 2 edges per pulse, so effective counts/rev doubles.
        """
        edge_times = _edge_times(digital_data, edge_type=edge_type)
        if edge_times.size < 2:
            return pd.DataFrame({"time": [], "rpm": []})

        dt = np.diff(edge_times)
        dt = dt[dt > 0]  # safety against any weird duplicates

        if dt.size == 0:
            return pd.DataFrame({"time": [], "rpm": []})

        edges_per_rev = pulses_per_revolution * (2 if edge_type == "both" else 1)
        rpm = 60.0 / (dt * edges_per_rev)

        # time aligns with the second edge in each dt interval
        return pd.DataFrame({"time": edge_times[1:1 + rpm.size], "rpm": rpm})

    def calculate_rpm_from_quadrature(
        self,
        cha_data,
        chb_data,
        pulses_per_revolution: int = 100,   # 100 PPR
        signed: bool = False,
        invert_direction: bool = False,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Quadrature X4 decode using CHA+CHB transition times.

        Returns:
          (rpm_df, diagnostics)

        rpm_df columns:
          - time: seconds
          - rpm: signed or unsigned RPM
          - count: accumulated quadrature counts (X4)
          - step: per-edge increment (-1, +1)
        """
        ta = np.asarray(cha_data.transition_times, dtype=float)
        tb = np.asarray(chb_data.transition_times, dtype=float)

        if ta.size == 0 or tb.size == 0:
            empty = pd.DataFrame({"time": [], "rpm": [], "count": [], "step": []})
            return empty, {"invalid_transitions": 0, "used_edges": 0}

        # Merge events: 0 for A, 1 for B
        t = np.concatenate([ta, tb])
        ch = np.concatenate([np.zeros_like(ta, dtype=np.uint8), np.ones_like(tb, dtype=np.uint8)])
        order = np.argsort(t, kind="mergesort")  # stable
        t = t[order]
        ch = ch[order]

        # Quadrature decode lookup for state=(A<<1)|B, forward sequence 00->01->11->10->00 gives +1
        # If your wiring defines opposite direction, use invert_direction=True.
        lut = np.zeros(16, dtype=np.int8)
        lut[(0 << 2) | 1] = +1
        lut[(1 << 2) | 3] = +1
        lut[(3 << 2) | 2] = +1
        lut[(2 << 2) | 0] = +1

        lut[(0 << 2) | 2] = -1
        lut[(2 << 2) | 3] = -1
        lut[(3 << 2) | 1] = -1
        lut[(1 << 2) | 0] = -1

        a = bool(cha_data.initial_state)
        b = bool(chb_data.initial_state)
        state = (int(a) << 1) | int(b)

        direction_sign = -1 if invert_direction else 1

        times_out = []
        counts_out = []
        steps_out = []
        invalid = 0
        count = 0

        # Optional: skip exact-duplicate timestamps (rare, but can happen with quantization)
        last_time = None

        for ti, chi in zip(t, ch):
            if last_time is not None and ti == last_time:
                # Two edges at the exact same time -> ambiguous (treat as invalid)
                invalid += 1
                continue
            last_time = ti

            prev_state = state

            if chi == 0:
                a = not a
            else:
                b = not b

            state = (int(a) << 1) | int(b)
            step = int(lut[(prev_state << 2) | state]) * direction_sign

            if step == 0:
                # invalid/non-Gray transition (glitch or dropped edge)
                invalid += 1
                continue

            count += step
            times_out.append(ti)
            counts_out.append(count)
            steps_out.append(step)

        if len(times_out) < 2:
            empty = pd.DataFrame({"time": [], "rpm": [], "count": [], "step": []})
            return empty, {"invalid_transitions": invalid, "used_edges": len(times_out)}

        times_out = np.asarray(times_out, dtype=float)
        counts_out = np.asarray(counts_out, dtype=np.int64)
        steps_out = np.asarray(steps_out, dtype=np.int8)

        dt = np.diff(times_out)
        dc = np.diff(counts_out)  # typically +/-1
        good = dt > 0

        dt = dt[good]
        dc = dc[good]
        t_rpm = times_out[1:][good]
        step_rpm = steps_out[1:][good]
        count_rpm = counts_out[1:][good]

        counts_per_rev = 4 * pulses_per_revolution  # X4 decode
        rpm = (dc / dt) * (60.0 / counts_per_rev)
        if not signed:
            rpm = np.abs(rpm)

        df = pd.DataFrame({"time": t_rpm, "rpm": rpm, "count": count_rpm, "step": step_rpm})
        return df, {"invalid_transitions": invalid, "used_edges": int(len(times_out))}

    def save_rpm_data(
        self,
        capture: mso_api.Capture,
        output_file: Path,
        channel_name: str,
        pulses_per_revolution: int = 100,
        edge_type: EdgeType = "rising",
        channel_b_name: Optional[str] = None,
        signed: bool = False,
        invert_direction: bool = False,
    ) -> pd.DataFrame:
        """
        If channel_b_name is provided, do quadrature X4 decode on (channel_name, channel_b_name).
        Otherwise, do single-channel edge timing on channel_name.

        Nidec 24H encoder is 100 PPR quadrature.
        """
        if channel_name not in capture.digital_data:
            raise ValueError(f"Channel '{channel_name}' not in capture.digital_data. Available: {list(capture.digital_data)}")

        output_file = Path(output_file)

        if channel_b_name is None:
            dd = capture.digital_data[channel_name]
            rpm_df = self.calculate_rpm_from_encoder(
                dd, pulses_per_revolution=pulses_per_revolution, edge_type=edge_type
            )
        else:
            if channel_b_name not in capture.digital_data:
                raise ValueError(f"Channel '{channel_b_name}' not in capture.digital_data. Available: {list(capture.digital_data)}")

            cha = capture.digital_data[channel_name]
            chb = capture.digital_data[channel_b_name]
            rpm_df, diag = self.calculate_rpm_from_quadrature(
                cha, chb,
                pulses_per_revolution=pulses_per_revolution,
                signed=signed,
                invert_direction=invert_direction,
            )
            # Optional: log quadrature health
            print(f"  Quadrature decode used_edges={diag['used_edges']} invalid_transitions={diag['invalid_transitions']}")

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
