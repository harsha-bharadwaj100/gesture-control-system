import tkinter as tk
import threading
import time
import csv
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType

# --- DAQ & Logging Configuration ---
CHANNEL = "Dev1/ai0"
SAMPLE_RATE = 1000
BUFFER_SIZE = 100
OUTPUT_FILE = "hackathon_balanced_dataset.csv"

# --- Protocol Generation ---
CYCLES = 10
ACTION_DURATION = 3.0
REST_DURATION = 3.0

# 1. Start with a solid baseline
PROTOCOL = [("Rest (Baseline)", 5.0)]

# 2. Fist Training Block (10 Reps)
for _ in range(CYCLES):
    PROTOCOL.append(("Fist", ACTION_DURATION))
    PROTOCOL.append(("Rest", REST_DURATION))

# 3. Transition Period (So the user knows to switch gestures)
PROTOCOL.append(("Get Ready for Open Hand...", 3.0))

# 4. Open Hand Training Block (10 Reps)
for _ in range(CYCLES):
    PROTOCOL.append(("Open Hand", ACTION_DURATION))
    PROTOCOL.append(("Rest", REST_DURATION))


class DatasetCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nadicare Hackathon - Focused EMG Collector")
        self.root.geometry("800x500")

        self.current_gesture = "Waiting..."
        self.is_recording = False
        self.protocol_index = 0

        self.instruction_label = tk.Label(
            root, text="Press Start to Begin Protocol", font=("Helvetica", 36, "bold")
        )
        self.instruction_label.pack(expand=True)

        # Added a progress counter so the user knows how many reps are left
        self.progress_label = tk.Label(root, text="", font=("Helvetica", 16))
        self.progress_label.pack(pady=10)

        self.btn_start = tk.Button(
            root,
            text="Start Recording",
            command=self.start_session,
            font=("Helvetica", 18),
            bg="#4CAF50",
            fg="white",
        )
        self.btn_start.pack(pady=30)

    def start_session(self):
        self.btn_start.config(state=tk.DISABLED, text="Recording in Progress...")
        self.is_recording = True

        self.daq_thread = threading.Thread(target=self.daq_loop, daemon=True)
        self.daq_thread.start()

        self.next_phase()

    def next_phase(self):
        if self.protocol_index < len(PROTOCOL):
            gesture, duration = PROTOCOL[self.protocol_index]
            self.current_gesture = gesture

            # Update Progress Label
            self.progress_label.config(
                text=f"Phase {self.protocol_index + 1} of {len(PROTOCOL)}"
            )

            # Visual feedback colors
            if "Rest" in gesture:
                color = "lightblue"
            elif "Ready" in gesture:
                color = "yellow"
            else:
                color = "salmon"

            self.root.configure(bg=color)
            self.instruction_label.configure(text=gesture, bg=color)

            self.protocol_index += 1
            self.root.after(int(duration * 1000), self.next_phase)
        else:
            self.finish_session()

    def finish_session(self):
        self.is_recording = False
        self.current_gesture = "Done"

        self.root.configure(bg="white")
        self.instruction_label.configure(
            text="Dataset Secured!", bg="white", fg="green"
        )
        self.progress_label.config(text="")
        self.btn_start.config(text="Finished")

    def daq_loop(self):
        try:
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan(
                    CHANNEL,
                    terminal_config=TerminalConfiguration.DIFF,
                    min_val=-5.0,
                    max_val=5.0,
                )

                task.timing.cfg_samp_clk_timing(
                    rate=SAMPLE_RATE,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=BUFFER_SIZE,
                )

                print(f"Hardware locked onto {CHANNEL}. Awaiting data stream...")

                with open(OUTPUT_FILE, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Timestamp", "Gesture_Label", "Voltage"])

                    task.start()

                    while self.is_recording:
                        data = task.read(number_of_samples_per_channel=BUFFER_SIZE)

                        current_time = time.time()
                        start_time = current_time - (BUFFER_SIZE / SAMPLE_RATE)
                        active_label = self.current_gesture

                        # Only write to CSV if it's an actual training label
                        if active_label not in [
                            "Waiting...",
                            "Get Ready for Open Hand...",
                        ]:
                            for i, val in enumerate(data):
                                t = start_time + (i * (1.0 / SAMPLE_RATE))
                                writer.writerow([t, active_label, val])

            print(f"File successfully written to {OUTPUT_FILE}")

        except Exception as e:
            print(f"\nCRITICAL DAQ ERROR:\n{e}")
            self.is_recording = False


if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetCollectorApp(root)
    root.mainloop()
