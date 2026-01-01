"""Saleae MSO API interface for encoder data capture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Literal, Tuple
import numpy as np
import pandas as pd
from saleae import mso_api

from .bench import TachMeasurement
EdgeType = Literal["rising", "falling", "both"]
RpmMethod = Literal["windowed", "edge_to_edge"]


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


def _check_quadrature_health(
    cha_data,
    chb_data,
    ratio_bounds: Tuple[float, float] = (0.7, 1.3),
) -> Tuple[bool, float, str]:
    """
    Check if CHB signal is healthy enough for quadrature decoding.

    Args:
        cha_data: Channel A digital data
        chb_data: Channel B digital data
        ratio_bounds: Acceptable CHB/CHA transition ratio range (default: 0.7-1.3)

    Returns:
        (is_healthy, ratio, message)
    """
    cha_count = len(cha_data.transition_times)
    chb_count = len(chb_data.transition_times)

    if cha_count == 0:
        return False, 0.0, "CHA has no transitions"

    ratio = chb_count / cha_count

    if ratio < ratio_bounds[0]:
        return False, ratio, f"CHB has only {ratio:.1%} of CHA transitions (expected ~100%)"
    if ratio > ratio_bounds[1]:
        return False, ratio, f"CHB has {ratio:.1%} of CHA transitions (expected ~100%)"

    return True, ratio, f"CHB/CHA ratio {ratio:.3f} is healthy"


def rpm_from_edges(
    edge_times: np.ndarray,
    pulses_per_revolution: int,
    edge_type: EdgeType,
    method: RpmMethod = "windowed",
    window_edges: int = 200,
    trim_quantiles: Optional[Tuple[float, float]] = (0.001, 0.999),
) -> pd.DataFrame:
    """
    Calculate RPM from edge timestamps.

    Args:
        edge_times: Array of edge timestamps (seconds)
        pulses_per_revolution: Encoder PPR (cycles per revolution on one channel)
        edge_type: Which edges were counted ("rising", "falling", "both")
        method: "windowed" (robust, recommended) or "edge_to_edge" (debug only)
        window_edges: Number of edges per window for windowed method (default 200)
        trim_quantiles: For edge_to_edge only, trim outliers at these quantiles

    Returns:
        DataFrame with columns: time, rpm

    Note:
        - "windowed" method is robust to glitches and outliers (recommended)
        - "edge_to_edge" method is sensitive to noise and produces extreme outliers
          Use edge_to_edge only for debugging with trim_quantiles enabled
    """
    edge_times = np.asarray(edge_times, dtype=float)
    if edge_times.size < 2:
        return pd.DataFrame({"time": [], "rpm": []})

    edges_per_rev = pulses_per_revolution * (2 if edge_type == "both" else 1)

    if method == "edge_to_edge":
        dt = np.diff(edge_times)
        good = dt > 0
        dt = dt[good]
        t = edge_times[1:][good]
        if dt.size == 0:
            return pd.DataFrame({"time": [], "rpm": []})

        rpm = 60.0 / (dt * edges_per_rev)

        # Optional trimming to prevent insane spikes dominating stats/plots
        if trim_quantiles is not None and rpm.size > 10:
            lo, hi = np.quantile(rpm, trim_quantiles)
            keep = (rpm >= lo) & (rpm <= hi)
            rpm = rpm[keep]
            t = t[keep]

        return pd.DataFrame({"time": t, "rpm": rpm})

    # method == "windowed"
    if edge_times.size <= window_edges:
        return pd.DataFrame({"time": [], "rpm": []})

    dt = edge_times[window_edges:] - edge_times[:-window_edges]
    good = dt > 0
    dt = dt[good]
    t = edge_times[window_edges:][good]
    if dt.size == 0:
        return pd.DataFrame({"time": [], "rpm": []})

    rpm = 60.0 * window_edges / (dt * edges_per_rev)
    return pd.DataFrame({"time": t, "rpm": rpm})


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
        minimum_pulse_width_samples: Optional[int] = None,  # Glitch filter: reject pulses shorter than N samples
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
            minimum_pulse_width_samples: Optional glitch filter (rejects pulses shorter than N samples)
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
                dc_kwargs = {
                    "channel": ch,
                    "name": name,
                    "port": digital_port,
                    "threshold_volts": threshold_volts
                }
                if minimum_pulse_width_samples is not None:
                    dc_kwargs["minimum_pulse_width_samples"] = minimum_pulse_width_samples
                enabled_channels.append(mso_api.DigitalChannel(**dc_kwargs))

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
        minimum_pulse_width_samples: Optional[int] = None,
        port: int = 0
    ) -> mso_api.Capture:
        """Capture encoder data from digital channels (legacy method).

        Args:
            duration: Capture duration in seconds
            channels: List of digital channel numbers to capture
            save_dir: Directory to save the capture data
            channel_names: Optional list of names for the channels
            threshold_volts: Digital threshold voltage (default 2.5V for 5V logic)
            minimum_pulse_width_samples: Optional glitch filter (rejects pulses shorter than N samples)
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
            threshold_volts=threshold_volts,
            minimum_pulse_width_samples=minimum_pulse_width_samples
        )

    def calculate_rpm_from_encoder(
        self,
        digital_data,
        pulses_per_revolution: int = 100,   # Nidec 24H is 100 PPR
        edge_type: EdgeType = "rising",
        method: RpmMethod = "windowed",
        window_edges: int = 200,
    ) -> pd.DataFrame:
        """
        Calculate RPM from a single encoder channel.

        Args:
            digital_data: Saleae digital channel data object
            pulses_per_revolution: PPR (cycles per revolution on one channel)
            edge_type: "rising", "falling", or "both"
            method: "windowed" (robust, default) or "edge_to_edge" (debug only)
            window_edges: Number of edges per window (default 200)

        Returns:
            DataFrame with columns: time, rpm

        Note:
            Windowed method is strongly recommended for production use.
            Edge-to-edge method can produce extreme outliers from noise.
        """
        edges = _edge_times(digital_data, edge_type=edge_type)
        return rpm_from_edges(
            edges,
            pulses_per_revolution=pulses_per_revolution,
            edge_type=edge_type,
            method=method,
            window_edges=window_edges,
        )

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
        method: RpmMethod = "windowed",
        window_edges: int = 200,
        auto_fallback: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate and save RPM data from encoder capture.

        Args:
            capture: Saleae capture object
            output_file: Path to save CSV output
            channel_name: Name of encoder channel A
            pulses_per_revolution: Encoder PPR (100 for Nidec 24H)
            edge_type: "rising", "falling", or "both"
            channel_b_name: Optional channel B for quadrature decode
            signed: Return signed RPM (requires quadrature)
            invert_direction: Invert rotation direction
            method: "windowed" (robust, default) or "edge_to_edge" (debug)
            window_edges: Number of edges per window (default 200)
            auto_fallback: Automatically fall back to single-channel if CHB unhealthy

        Returns:
            DataFrame with RPM data

        Note:
            - Quadrature decoding requires healthy CHB signal (ratio 0.7-1.3 to CHA)
            - If auto_fallback=True and CHB is unhealthy, uses single-channel CHA only
            - Windowed method strongly recommended for production use
        """
        if channel_name not in capture.digital_data:
            raise ValueError(f"Channel '{channel_name}' not in capture.digital_data. Available: {list(capture.digital_data)}")

        output_file = Path(output_file)

        # Attempt quadrature decode if channel_b_name provided
        if channel_b_name is not None:
            if channel_b_name not in capture.digital_data:
                raise ValueError(f"Channel '{channel_b_name}' not in capture.digital_data. Available: {list(capture.digital_data)}")

            cha = capture.digital_data[channel_name]
            chb = capture.digital_data[channel_b_name]

            # Check CHB health
            is_healthy, ratio, msg = _check_quadrature_health(cha, chb)

            if not is_healthy:
                warning = f"  ⚠️  {msg}"
                print(warning)
                if auto_fallback:
                    print(f"  → Falling back to single-channel {channel_name} (auto_fallback=True)")
                    channel_b_name = None  # Fall through to single-channel path
                else:
                    print(f"  → Attempting quadrature decode anyway (auto_fallback=False)")

            if channel_b_name is not None:  # Still doing quadrature after health check
                print(f"  ✓ {msg} - using quadrature X4 decode")
                rpm_df, diag = self.calculate_rpm_from_quadrature(
                    cha, chb,
                    pulses_per_revolution=pulses_per_revolution,
                    signed=signed,
                    invert_direction=invert_direction,
                )
                print(f"  Quadrature decode: used_edges={diag['used_edges']} invalid_transitions={diag['invalid_transitions']}")

                rpm_df.to_csv(output_file, index=False)
                print(f"  RPM data saved to: {output_file}")
                return rpm_df

        # Single-channel decode path
        dd = capture.digital_data[channel_name]
        rpm_df = self.calculate_rpm_from_encoder(
            dd,
            pulses_per_revolution=pulses_per_revolution,
            edge_type=edge_type,
            method=method,
            window_edges=window_edges,
        )

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

    def close(self) -> None:
        """Close device connection if supported by the MSO API."""
        if self._mso is None:
            return
        close_fn = getattr(self._mso, "close", None)
        if callable(close_fn):
            close_fn()
        self._mso = None

    def healthcheck(self) -> dict[str, Any]:
        return {
            "connected": self._mso is not None,
            "serial_number": self.serial_number,
        }

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


@dataclass(frozen=True)
class SaleaeTachConfig:
    encoder_channels: list[int] | None = None
    threshold_volts: float = 1.65
    minimum_pulse_width_samples: Optional[int] = None
    pulses_per_revolution: int = 1
    edge_type: EdgeType = "rising"
    rpm_method: RpmMethod = "windowed"
    window_edges: int = 200
    digital_port: int = 0
    channel_b_index: Optional[int] = 1
    signed: bool = False
    invert_direction: bool = False
    auto_fallback: bool = True
    motor_voltage_channel: Optional[int] = 0
    motor_voltage_name: str = "motor_voltage"
    analog_voltage_range: float = 20.0
    analog_probe_attenuation: mso_api.ProbeAttenuation = mso_api.ProbeAttenuation.PROBE_10X

    def __post_init__(self) -> None:
        if self.encoder_channels is None:
            object.__setattr__(self, "encoder_channels", [1, 2])


def _analog_mean(data) -> float | None:
    if hasattr(data, "voltages"):
        voltages = np.asarray(data.voltages, dtype=float)
    elif hasattr(data, "traces"):
        traces = [np.asarray(trace.voltages, dtype=float) for trace in data.traces]
        voltages = np.concatenate(traces) if traces else np.array([], dtype=float)
    else:
        return None
    if voltages.size == 0:
        return None
    return float(voltages.mean())


class SaleaeTachometer:
    """Tachometer adapter using SaleaeCapture + RPM processing."""

    def __init__(self, config: SaleaeTachConfig, serial_number: Optional[str] = None):
        self.config = config
        self._capture = SaleaeCapture(serial_number=serial_number)

    def connect(self) -> None:
        self._capture.connect()

    def close(self) -> None:
        self._capture.close()

    def healthcheck(self) -> dict[str, Any]:
        info = self._capture.healthcheck()
        info.update(
            {
                "encoder_channels": self.config.encoder_channels,
                "threshold_volts": self.config.threshold_volts,
                "motor_voltage_channel": self.config.motor_voltage_channel,
            }
        )
        return info

    def measure(
        self,
        *,
        duration_s: float,
        capture_dir: Path,
        rpm_file: Path,
    ) -> TachMeasurement:
        capture_dir = Path(capture_dir)
        rpm_file = Path(rpm_file)
        rpm_file.parent.mkdir(parents=True, exist_ok=True)

        channel_names = [f"encoder_{i}" for i in range(len(self.config.encoder_channels))]
        analog_channels = None
        analog_names = None
        if self.config.motor_voltage_channel is not None:
            analog_channels = [self.config.motor_voltage_channel]
            analog_names = [self.config.motor_voltage_name]

        capture = self._capture.capture_mixed_data(
            duration=duration_s,
            save_dir=capture_dir,
            digital_channels=self.config.encoder_channels,
            digital_names=channel_names,
            digital_port=self.config.digital_port,
            threshold_volts=self.config.threshold_volts,
            minimum_pulse_width_samples=self.config.minimum_pulse_width_samples,
            analog_channels=analog_channels,
            analog_names=analog_names,
            analog_voltage_range=self.config.analog_voltage_range,
            analog_probe_attenuation=self.config.analog_probe_attenuation,
        )

        if not channel_names:
            raise ValueError("No encoder channels configured for tachometer")

        channel_a_name = channel_names[0]
        channel_b_name = None
        if self.config.channel_b_index is not None:
            if not 0 <= self.config.channel_b_index < len(channel_names):
                raise ValueError(
                    f"channel_b_index {self.config.channel_b_index} out of range for "
                    f"{len(channel_names)} encoder channels"
                )
            channel_b_name = channel_names[self.config.channel_b_index]

        rpm_df = self._capture.save_rpm_data(
            capture=capture,
            output_file=rpm_file,
            channel_name=channel_a_name,
            pulses_per_revolution=self.config.pulses_per_revolution,
            edge_type=self.config.edge_type,
            channel_b_name=channel_b_name,
            signed=self.config.signed,
            invert_direction=self.config.invert_direction,
            method=self.config.rpm_method,
            window_edges=self.config.window_edges,
            auto_fallback=self.config.auto_fallback,
        )

        motor_voltage_mean = None
        if analog_names:
            analog_name = analog_names[0]
            analog_data = capture.analog_data.get(analog_name)
            if analog_data is not None:
                motor_voltage_mean = _analog_mean(analog_data)

        diag = {
            "channel_a_name": channel_a_name,
            "channel_b_name": channel_b_name,
            "encoder_channels": self.config.encoder_channels,
            "pulses_per_revolution": self.config.pulses_per_revolution,
        }

        return TachMeasurement(
            rpm_df=rpm_df,
            capture_dir=capture_dir,
            rpm_file=rpm_file,
            diag=diag,
            motor_voltage_mean=motor_voltage_mean,
        )
