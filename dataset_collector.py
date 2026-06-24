import tkinter as tk
from tkinter import messagebox
import threading
import time
import csv
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolStep:
    label: str
    duration_seconds: float
    repetition_index: int | None = None
    is_recordable: bool = True


def build_study_protocol(
    gestures=("Fist", "Open palm", "Wrist up", "Wrist down", "Pinch"),
    repetitions_per_gesture: int = 30,
    action_duration_seconds: float = 4.0,
    rest_duration_seconds: float = 4.0,
    baseline_duration_seconds: float = 5.0,
    transition_duration_seconds: float = 3.0,
):
    protocol = [
        ProtocolStep("Rest (Baseline)", baseline_duration_seconds, is_recordable=False)
    ]

    for gesture_index, gesture_name in enumerate(gestures):
        for repetition_index in range(1, repetitions_per_gesture + 1):
            protocol.append(
                ProtocolStep(
                    gesture_name,
                    action_duration_seconds,
                    repetition_index=repetition_index,
                )
            )
            protocol.append(
                ProtocolStep(
                    "Rest",
                    rest_duration_seconds,
                    repetition_index=repetition_index,
                    is_recordable=False,
                )
            )

        if gesture_index < len(gestures) - 1:
            protocol.append(
                ProtocolStep(
                    f"Get Ready for {gestures[gesture_index + 1]}...",
                    transition_duration_seconds,
                    is_recordable=False,
                )
            )

    return protocol

# --- DAQ & Logging Configuration ---
CHANNEL = "Dev1/ai0"
SAMPLE_RATE = 1000
BUFFER_SIZE = 100
REPETITIONS_PER_GESTURE = 30
PROTOCOL = build_study_protocol(repetitions_per_gesture=REPETITIONS_PER_GESTURE)


def sanitize_participant_name(raw_name: str) -> str:
    cleaned = raw_name.strip().replace(" ", "_")
    allowed = [character for character in cleaned if character.isalnum() or character in ("_", "-")]
    return "".join(allowed)


class DatasetCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nadicare Hackathon - Focused EMG Collector")
        self.root.geometry("800x500")

        self.participant_var = tk.StringVar(value="")
        self.current_gesture = "Waiting..."
        self.is_recording = False
        self.protocol_index = 0
        self.current_step = None

        self.name_label = tk.Label(
            root, text="Enter participant name", font=("Helvetica", 16)
        )
        self.name_label.pack(pady=(20, 5))

        self.participant_entry = tk.Entry(
            root, textvariable=self.participant_var, font=("Helvetica", 18)
        )
        self.participant_entry.pack(pady=(0, 20))
        self.participant_entry.focus_set()

        self.instruction_label = tk.Label(
            root,
            text="Enter a name, then press Start to Begin",
            font=("Helvetica", 30, "bold"),
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
        participant_name = sanitize_participant_name(self.participant_var.get())
        if not participant_name:
            messagebox.showwarning(
                "Missing Name", "Please enter a participant name before starting."
            )
            return

        self.participant_id = participant_name
        self.output_file = f"{self.participant_id}5gesture_dataset.csv"
        self.participant_entry.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED, text="Recording in Progress...")
        self.is_recording = True

        self.daq_thread = threading.Thread(target=self.daq_loop, daemon=True)
        self.daq_thread.start()

        self.next_phase()

    def next_phase(self):
        if self.protocol_index < len(PROTOCOL):
            self.current_step = PROTOCOL[self.protocol_index]
            self.current_gesture = self.current_step.label
            duration = self.current_step.duration_seconds

            # Update Progress Label
            self.progress_label.config(
                text=f"Phase {self.protocol_index + 1} of {len(PROTOCOL)}"
            )

            # Visual feedback colors
            if "Rest" in self.current_gesture:
                color = "lightblue"
            elif "Ready" in self.current_gesture:
                color = "yellow"
            else:
                color = "salmon"

            self.root.configure(bg=color)
            self.instruction_label.configure(text=self.current_gesture, bg=color)

            self.protocol_index += 1
            self.root.after(int(duration * 1000), self.next_phase)
        else:
            self.finish_session()

    def finish_session(self):
        self.is_recording = False
        self.current_gesture = "Done"
        self.current_step = None

        self.root.configure(bg="white")
        self.instruction_label.configure(
            text="Dataset Secured!", bg="white", fg="green"
        )
        self.progress_label.config(text="")
        self.btn_start.config(text="Finished")
        self.participant_entry.config(state=tk.DISABLED)

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

                with open(self.output_file, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        ["Participant_ID", "Timestamp", "Gesture_Label", "Repetition", "Voltage"]
                    )

                    task.start()

                    while self.is_recording:
                        data = task.read(number_of_samples_per_channel=BUFFER_SIZE)

                        current_time = time.time()
                        start_time = current_time - (BUFFER_SIZE / SAMPLE_RATE)
                        active_label = self.current_gesture

                        if self.current_step and self.current_step.is_recordable:
                            for i, val in enumerate(data):
                                t = start_time + (i * (1.0 / SAMPLE_RATE))
                                writer.writerow(
                                    [
                                        self.participant_id,
                                        t,
                                        active_label,
                                        self.current_step.repetition_index,
                                        val,
                                    ]
                                )

            print(f"File successfully written to {self.output_file}")

        except Exception as e:
            print(f"\nCRITICAL DAQ ERROR:\n{e}")
            self.is_recording = False


if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetCollectorApp(root)
    root.mainloop()
