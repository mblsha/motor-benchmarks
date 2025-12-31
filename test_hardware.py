#!/usr/bin/env python3
"""Quick hardware test to verify Jumperless and Saleae MSO connections."""

from pathlib import Path
from saleae import mso_api
from motor_benchmarks.jumperless import JumperlessController


def test_saleae_connection():
    """Test Saleae MSO device connection."""
    print("\n=== Testing Saleae MSO Connection ===")
    try:
        mso = mso_api.MSO()
        print(f"✓ Connected to Saleae MSO")
        print(f"  Serial Number: {mso.serial_number}")
        return True
    except Exception as e:
        print(f"✗ Failed to connect to Saleae MSO")
        print(f"  Error: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure Saleae MSO is connected via USB")
        print("  - On Linux: Run 'uv run python -m saleae.mso_api.utils.install_udev'")
        print("  - Check that Logic 2 v2.4.20+ is installed")
        return False


def test_jumperless_connection(port: str = "/dev/ttyACM0"):
    """Test Jumperless serial connection."""
    print(f"\n=== Testing Jumperless Connection ({port}) ===")
    try:
        controller = JumperlessController(port=port, baudrate=115200)
        controller.connect()
        print(f"✓ Connected to Jumperless on {port}")

        # Test basic command
        response = controller.send_command("print('Hello from Jumperless!')")
        print(f"✓ MicroPython REPL responding")

        controller.disconnect()
        return True
    except Exception as e:
        print(f"✗ Failed to connect to Jumperless")
        print(f"  Error: {e}")
        print("\nTroubleshooting:")
        print(f"  - Check available ports: ls /dev/tty*")
        print(f"  - Verify Jumperless is connected and powered")
        print(f"  - Ensure MicroPython firmware is installed")
        return False


def test_encoder_capture(
    encoder_channel: int = 0,
    duration: float = 2.0,
    threshold_volts: float = 1.65
):
    """Test encoder signal capture."""
    print(f"\n=== Testing Encoder Capture (Channel {encoder_channel}) ===")
    print(f"Make sure your encoder is connected to channel {encoder_channel}")
    print(f"and the motor is spinning...")

    try:
        mso = mso_api.MSO()

        # Configure capture
        capture_config = mso_api.CaptureConfig(
            enabled_channels=[
                mso_api.DigitalChannel(
                    channel=encoder_channel,
                    name=f"encoder_{encoder_channel}",
                    threshold_volts=threshold_volts
                )
            ],
            capture_settings=mso_api.TimedCapture(capture_length_seconds=duration)
        )

        # Capture data
        print(f"  Capturing {duration}s of data...")
        save_dir = Path("./test_capture")
        capture = mso.capture(capture_config, save_dir=save_dir)

        # Check for transitions
        channel_name = f"encoder_{encoder_channel}"
        if channel_name in capture.digital_data:
            digital_data = capture.digital_data[channel_name]
            num_transitions = len(digital_data.transition_times)

            print(f"✓ Capture complete")
            print(f"  Transitions detected: {num_transitions}")

            if num_transitions > 0:
                print(f"  First transition at: {digital_data.transition_times[0]:.6f}s")
                print(f"  Last transition at: {digital_data.transition_times[-1]:.6f}s")

                # Estimate frequency
                if num_transitions >= 2:
                    duration_actual = digital_data.transition_times[-1] - digital_data.transition_times[0]
                    freq = num_transitions / duration_actual
                    print(f"  Estimated frequency: {freq:.2f} Hz")
            else:
                print("  ⚠ No transitions detected!")
                print("  - Verify encoder is connected and motor is running")
                print(f"  - Try adjusting threshold_volts (current: {threshold_volts}V)")
                print("  - Check encoder signal levels with an oscilloscope")

            # Cleanup
            import shutil
            shutil.rmtree(save_dir, ignore_errors=True)

            return num_transitions > 0
        else:
            print(f"✗ No data captured for channel {encoder_channel}")
            return False

    except Exception as e:
        print(f"✗ Capture failed")
        print(f"  Error: {e}")
        return False


def main():
    """Run all hardware tests."""
    print("=" * 60)
    print("Motor Benchmark Hardware Test")
    print("=" * 60)

    results = []

    # Test Saleae
    results.append(("Saleae MSO", test_saleae_connection()))

    # Test Jumperless
    results.append(("Jumperless", test_jumperless_connection()))

    # Test encoder capture (optional)
    print("\n" + "=" * 60)
    test_encoder = input("Test encoder capture? (y/n): ").lower().strip() == 'y'
    if test_encoder:
        encoder_ch = int(input("Encoder channel number (default 0): ") or "0")
        threshold = float(input("Threshold voltage (default 1.65V): ") or "1.65")
        results.append(("Encoder Capture", test_encoder_capture(encoder_ch, threshold_volts=threshold)))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name:.<40} {status}")

    all_passed = all(success for _, success in results)
    print("=" * 60)

    if all_passed:
        print("✓ All tests passed! Ready to run sweeps.")
        return 0
    else:
        print("✗ Some tests failed. Please fix issues before running sweeps.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
