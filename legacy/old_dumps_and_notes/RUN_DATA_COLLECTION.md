Data collection run-sheet — first 3 participants

Purpose
- Capture full 5-gesture dataset for first three participants (5 gestures × 50 repetitions).

Gestures
- Fist
- Open palm
- Wrist up
- Wrist down
- Pinch

Filename convention
- participant_XX_5gesture_dataset.csv (XX = 01, 02, 03)
- CSV columns: Participant_ID, Timestamp, Gesture_Label, Repetition, Voltage

Per-participant steps
1. Power on DAQ and amplifier; verify channel connection.
2. Open an elevated PowerShell and activate the venv.

   powershell
   (& .venv\Scripts\Activate.ps1)

3. Launch collector GUI:

   powershell
   python dataset_collector.py

4. In the GUI: set Participant ID to two digits (e.g., 01), confirm protocol (5 gestures, 50 reps).
5. Run the session. For each repetition: follow the on-screen prompts.
6. When finished, verify CSV saved in project root and named correctly.

Quick verification (after session)
- Open the CSV; verify it contains ~5 * 50 * samples_per_rep rows and expected columns.
- Example filename: participant_01_5gesture_dataset.csv

Notes
- When switching hardware between people, allow 60–120s warm-up for amplifier stability.
- If a session fails, append a postfix _retryN to filename and restart that participant.

Next steps after collecting 3 participants
- Run the benchmark (pilot) on collected CSVs to validate pipeline and timing.

Benchmark command (pilot run):

powershell
python Model_Training.py --data-dir . --window-size 200 --step-size 100 --results-csv results_pilot.csv

Contact
- Ask the operator to notify when participant CSVs are saved; I will run the benchmark and report results.