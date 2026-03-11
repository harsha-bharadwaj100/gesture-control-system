import nidaqmx
from nidaqmx.constants import TerminalConfiguration
import csv
import time

# --- Configuration ---
# Match this to what you see in NI MAX (e.g., "Dev1/ai0")
CHANNEL = "Dev1/ai0"
OUTPUT_FILE = "hackathon_gesture_data.csv"


def run_live_meter():
    try:
        # Initialize the DAQ task
        with nidaqmx.Task() as task:

            # Configure AI 0 for Differential mode
            task.ai_channels.add_ai_voltage_chan(
                CHANNEL,
                terminalconfig=TerminalConfiguration.DIFF,  # <-- The fix is right here!
                min_val=-5.0,
                max_val=5.0,
            )

            print(f"✅ Successfully connected to {CHANNEL}!")
            print("🚀 Recording and live-monitoring... Flex your arm!")
            print("🛑 Press Ctrl+C to stop.\n")

            # Open CSV for logging
            with open(OUTPUT_FILE, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Voltage"])

                # Read continuously in a loop
                while True:
                    # Read a single raw voltage sample
                    voltage = task.read()

                    # Log the data
                    writer.writerow([time.time(), voltage])

                    # Create the live terminal visualizer
                    amplitude = abs(voltage)
                    # Adjust the '50' multiplier if the bar is too sensitive or not sensitive enough
                    bar_length = int(amplitude * 50)

                    bar = "|" * bar_length

                    print(
                        f"EMG Signal: {voltage:>7.4f} V  {bar:<50}",
                        end="\r",
                        flush=True,
                    )
                    time.sleep(0.01)

    except nidaqmx.errors.DaqError as e:
        print(f"\n\n❌ NI-DAQmx Hardware Error: {e}")
        print("Check if NI MAX recognizes your device, and verify the 'Dev1' name.")
    except KeyboardInterrupt:
        print(f"\n\n📁 Recording stopped. Your data is saved to {OUTPUT_FILE}.")


if __name__ == "__main__":
    run_live_meter()
