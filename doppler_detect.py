#!/usr/bin/env python3
"""
CQRobot 10.525GHz Doppler Microwave Motion Sensor (CQRSENWB01)
Detection script for Jetson Orin Nano Super.

Wiring:
  Red   -> 3.3V (Pin 1 or 17) -- set DIP switch to 3.3V!
  Black -> GND  (Pin 6, 9, 14, 20, 25, 30, 34, or 39)
  Green -> Pin 29 (GPIO01 / PQ.05 / gpiochip0 line 105)

Output is ACTIVE LOW:
  0 = motion detected
  1 = no motion
"""

import subprocess
import time
import signal
import sys

CHIP = "gpiochip0"
LINE = 105  # GPIO01 = PQ.05 = physical pin 29

running = True

def signal_handler(sig, frame):
    global running
    running = False
    print("\nStopping...")

signal.signal(signal.SIGINT, signal_handler)

def read_gpio():
    r = subprocess.run(["gpioget", "-B", "pull-up", CHIP, str(LINE)],
                       capture_output=True, text=True)
    return r.stdout.strip()

def check_sensor_present():
    """Check if the sensor is connected by reading with pull-up vs pull-down."""
    print("Checking if sensor is connected...\n")

    # With pull-up: unconnected pin reads 1, sensor may pull to 0
    r_up = subprocess.run(["gpioget", "-B", "pull-up", CHIP, str(LINE)],
                          capture_output=True, text=True)
    val_up = r_up.stdout.strip()

    # With pull-down: unconnected pin reads 0, sensor may push to 1
    r_down = subprocess.run(["gpioget", "-B", "pull-down", CHIP, str(LINE)],
                            capture_output=True, text=True)
    val_down = r_down.stdout.strip()

    # No bias (floating)
    r_float = subprocess.run(["gpioget", CHIP, str(LINE)],
                             capture_output=True, text=True)
    val_float = r_float.stdout.strip()

    print(f"  Pull-up read   : {val_up}  {'(sensor driving LOW = motion!)' if val_up == '0' else '(HIGH = idle or unconnected)'}")
    print(f"  Pull-down read : {val_down}  {'(sensor driving HIGH = no motion)' if val_down == '1' else '(LOW = motion or unconnected)'}")
    print(f"  Floating read  : {val_float}")

    # If pull-up and pull-down give different values, pin is floating (no sensor)
    # If both give same value, sensor is actively driving the pin
    if val_up == val_down:
        print(f"\n  -> Sensor IS driving the line to {val_up}")
        return True
    else:
        print(f"\n  -> Pin appears floating (pull-up={val_up}, pull-down={val_down})")
        print("     Sensor may not be connected or not powered.")
        return False

def monitor_motion(duration=30):
    """Monitor for motion with active-low logic."""
    global running
    print(f"\n--- Monitoring motion for {duration}s (Ctrl+C to stop) ---")
    print("    Active LOW: 0=MOTION  1=idle\n")

    motion_count = 0
    last_state = None
    start = time.time()

    while running and (time.time() - start) < duration:
        val = read_gpio()
        motion = (val == "0")
        elapsed = time.time() - start

        if motion != last_state:
            if motion:
                motion_count += 1
                print(f"  [{elapsed:6.1f}s] ** MOTION DETECTED ** (count: {motion_count})")
            else:
                print(f"  [{elapsed:6.1f}s]    motion stopped")
            last_state = motion

        time.sleep(0.01)

    elapsed = time.time() - start
    print(f"\n--- Done: {motion_count} motion events in {elapsed:.1f}s ---")
    return motion_count > 0

def scan_all_pins():
    """Quick scan of all 40-pin header GPIOs with pull-up bias."""
    pins = {
        144: ("Pin  7", "GPIO09"),  112: ("Pin 11", "UART1_RTS"),
         50: ("Pin 12", "I2S0_SCLK"), 122: ("Pin 13", "SPI1_SCK"),
         85: ("Pin 15", "GPIO12"),  126: ("Pin 16", "SPI1_CS1"),
        125: ("Pin 18", "SPI1_CS0"), 135: ("Pin 19", "SPI0_MOSI"),
        134: ("Pin 21", "SPI0_MISO"), 123: ("Pin 22", "SPI1_MISO"),
        133: ("Pin 23", "SPI0_SCK"), 136: ("Pin 24", "SPI0_CS0"),
        137: ("Pin 26", "SPI0_CS1"), 105: ("Pin 29", "GPIO01"),
        106: ("Pin 31", "GPIO11"),   41: ("Pin 32", "GPIO07"),
         43: ("Pin 33", "GPIO13"),   53: ("Pin 35", "I2S0_FS"),
        113: ("Pin 36", "UART1_CTS"), 124: ("Pin 37", "SPI1_MOSI"),
         52: ("Pin 38", "I2S0_SDIN"), 51: ("Pin 40", "I2S0_SDOUT"),
    }

    print("\n--- Scanning all header pins (with pull-up bias) ---")
    print("    Pins driven LOW by sensor will show as 0\n")
    driven = []
    for line_num, (pin, func) in sorted(pins.items()):
        try:
            r = subprocess.run(["gpioget", "-B", "pull-up", CHIP, str(line_num)],
                               capture_output=True, text=True, timeout=2)
            val = r.stdout.strip()
            marker = " <-- DRIVEN LOW" if val == "0" else ""
            if val == "0":
                driven.append((pin, func, line_num))
            print(f"  {pin:8s} {func:12s} (line {line_num:3d}): {val}{marker}")
        except Exception as e:
            print(f"  {pin:8s} {func:12s} (line {line_num:3d}): ERROR")

    if driven:
        print(f"\n  Found {len(driven)} pin(s) driven LOW:")
        for p, f, l in driven:
            print(f"    {p} ({f}) - could be sensor output")
    return driven

if __name__ == "__main__":
    print("=" * 58)
    print("  CQRobot 10.525GHz Doppler Motion Sensor")
    print("  Jetson Orin Nano Super - GPIO01 (Pin 29, line 105)")
    print("=" * 58)

    # Step 1: Check if sensor is driving the pin
    present = check_sensor_present()

    if not present:
        # Scan all pins to find where sensor might be
        driven = scan_all_pins()

        if not driven:
            print("\n" + "=" * 58)
            print("  NO SENSOR SIGNAL FOUND")
            print()
            print("  Checklist:")
            print("  [1] Set DIP switch to match voltage (3.3V recommended)")
            print("  [2] Check power LED on sensor board is ON (red)")
            print("  [3] Green wire -> Physical Pin 29 on 40-pin header")
            print("  [4] Red wire   -> 3.3V (Pin 1 or Pin 17)")
            print("  [5] Black wire -> GND  (Pin 6, 9, 14, etc.)")
            print("  [6] Try 5V if 3.3V doesn't work (but set DIP to 5V)")
            print("=" * 58)
            sys.exit(1)

    # Step 2: Live monitoring
    print()
    monitor_motion(duration=30)
