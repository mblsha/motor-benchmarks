# Motor Benchmarks

Parameter sweep system for motor characterization using Jumperless V5 (MicroPython PWM control) and Saleae Logic Analyzer (encoder data capture).

## Features

- **PWM Control**: Control motor via Jumperless V5 MicroPython serial interface
- **Data Acquisition**: Capture encoder signals using Saleae MSO API (v0.5.4+)
- **Motor Voltage**: Capture analog motor voltage (A0) and store average per point
- **Parameter Sweeps**: Automated duty cycle sweeps with configurable parameters
- **Analysis**: RPM statistics, time-series plots, and efficiency curves
- **Type-safe Configuration**: Python dataclass-based configuration
- **Direct Hardware Access**: Uses Saleae MSO API for direct USB device control (no Logic 2 software required during capture)

## Installation

```bash
# Install dependencies
uv sync

# Or install as a CLI tool
uv pip install -e .
```

## Quick Start

### 1. Configure Your Setup

Use CLI flags to match your hardware (no code edits required):

```bash
# Single datapoint
motor-benchmarks point \
  --serial-port /dev/cu.usbmodemJLV5port5 \
  --duty 0.35 \
  --encoder-channels 1,2 \
  --pulses-per-revolution 1

# Sweep
motor-benchmarks sweep \
  --serial-port /dev/cu.usbmodemJLV5port5 \
  --duty-start 0.2 \
  --duty-end 0.9 \
  --duty-steps 8 \
  --encoder-channels 1,2 \
  --pulses-per-revolution 1
```

### 2. Connect Saleae MSO Device

Connect your Saleae Logic MSO device via USB. The saleae-mso-api communicates directly with the hardware - no Logic 2 software needs to be running during capture (though you should have Logic 2 installed for drivers).

### 3. Run a Sweep

```bash
# Using uv
uv run motor-benchmarks sweep --serial-port /dev/cu.usbmodemJLV5port5

# Or if installed
motor-benchmarks sweep --serial-port /dev/cu.usbmodemJLV5port5
```

### 4. Smoke-Test Submodules (Real Hardware)

```bash
# Motor driver only
motor-benchmarks motor-test --serial-port /dev/cu.usbmodemJLV5port5 --duty 0.3 --duration-s 2

# Tachometer only
motor-benchmarks tach-test --encoder-channels 1,2 --capture-s 2
```

### 4. View Results

Results are saved in timestamped directories under `./results/`:

```
results/
└── nidec-24h_20250131_182230/
    ├── summary.csv              # Aggregate results (includes motor_voltage_mean)
    ├── metadata.json            # Sweep configuration
    ├── report.txt               # Text summary
    ├── efficiency_curve.png     # RPM vs duty cycle plot
    ├── capture_*.csv            # Raw Saleae captures
    ├── rpm_*.csv                # Processed RPM data
    └── timeseries_*.png         # RPM vs time plots
```

## Project Structure

```
src/motor_benchmarks/
├── __init__.py          # Package initialization
├── __main__.py          # CLI entry point
├── config.py            # Configuration dataclass
├── bench.py             # Single-point measurement primitives
├── saleae_capture.py    # Saleae data acquisition
├── analysis.py          # Data analysis and plotting
└── sweep.py             # Sweep orchestration
```

## Configuration Details

### SweepConfig Parameters

- **duty_cycle_start/end**: PWM duty cycle range (0.0 to 1.0)
- **duty_cycle_steps**: Number of measurement points in the sweep (use 1 for a single datapoint)
- **acquisition_duration**: Data capture duration at each point (seconds)
- **settle_time**: Wait time before capture for motor stabilization (seconds)
- **serial_port**: Jumperless serial port (find with `ls /dev/tty*`)
- **serial_baudrate**: Serial baud rate (default: 115200)
- **encoder_channels**: List of Saleae MSO digital channel numbers
- **threshold_volts**: Digital logic threshold voltage (1.65V for 3.3V, 2.5V for 5V)
- **pulses_per_revolution**: Number of encoder pulses per motor revolution
- **edge_type**: Which edges to count: 'rising', 'falling', or 'both'
- **motor_voltage_channel**: Analog channel for motor voltage (set None to disable)
- **output_dir**: Results directory
- **motor_name**: Motor identifier for labeling

## Hardware Setup

1. **Jumperless V5**: Connect via USB, PWM output to motor driver
2. **Motor & Driver**: Power supply and encoder connections
3. **Saleae Logic Analyzer**: Connect to encoder outputs (channels 1 & 2; reserve channel 0 for PWM)
4. **Saleae Analog A0**: Connect to motor voltage (default motor_voltage_channel=0)

## Customization

### Using as a Library

```python
from motor_benchmarks import SweepConfig, MotorSweep

config = SweepConfig(
    duty_cycle_start=0.3,
    duty_cycle_end=1.0,
    duty_cycle_steps=15,
    serial_port="/dev/ttyUSB0"
)

sweep = MotorSweep(config)
results_dir = sweep.run()
```

For a single datapoint, set start=end and steps=1:

```python
config = SweepConfig(
    duty_cycle_start=0.35,
    duty_cycle_end=0.35,
    duty_cycle_steps=1,
    serial_port="/dev/ttyUSB0"
)
sweep = MotorSweep(config)
results_dir = sweep.run()
```

### Analyzing Existing Data

```python
from motor_benchmarks import MotorAnalyzer

analyzer = MotorAnalyzer("results/nidec-24h_20250131_182230")
summary_df, metadata = analyzer.load_sweep_data()
report = analyzer.generate_report()
print(report)
```

## Troubleshooting

- **Serial Port Not Found**: Check `ls /dev/tty*` for available ports
- **Saleae MSO Not Found**:
  - Ensure device is connected via USB
  - On Linux, run `uv run python -m saleae.mso_api.utils.install_udev` to install udev rules
  - Check that Logic 2 software is installed (provides drivers)
- **Import Errors**: Make sure you've run `uv sync` to install dependencies
- **PWM Not Working**: Verify Jumperless pin configuration and MicroPython setup
- **No Transitions Detected**: Check encoder connections, adjust `threshold_volts`, verify motor is spinning

## Dependencies

- `pyserial` - Serial communication with Jumperless
- `saleae-mso-api` - Saleae Logic MSO API (v0.5.4+)
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Plotting

## Requirements

- **Python 3.12+**
- **Saleae Logic MSO device** (Logic MSO 4x50, 4x200, etc.)
- **Logic 2 software** v2.4.20+ installed (for drivers and udev rules)
- **Jumperless V5** with MicroPython firmware

## License

MIT
