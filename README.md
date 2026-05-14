# Nadicare EMG Gesture Recognition System

![Status: MVP / Hackathon Winner](https://img.shields.io/badge/Status-Hackathon%20Winner-brightgreen)
![Hardware: Anthriq (Nexstem)](https://img.shields.io/badge/Hardware-Anthriq%20%7C%20NI%20DAQ-blue)

A real-time, single-channel sEMG gesture recognition system and Brain-Computer Interface (BCI). Built for the Nadicare HealthTech Hackathon, this project uses enterprise-grade biosignal infrastructure to decode human intent and map it to digital actions. 

By leveraging advanced Digital Signal Processing (DSP) and Machine Learning (XGBoost), the system overcomes traditional AC-coupled hardware limitations to provide zero-latency, zero-shot calibration gesture recognition.

## 🌟 Key Innovations & USPs

*   **The "Anatomical Hack" (Single-Channel Spatial Resolution):** Bypassed the limitation of a single AC-coupled channel by placing sensors on horizontal flexors (Radial vs. Ulnar). This forces the thumb and index finger to produce opposite electrical polarities, which are classified using polarity-sensitive features (Kurtosis and Skewness).
*   **Zero-Shot Guided Onboarding (Dynamic Standardization):** Skin impedance and muscle density change per user. The live inference script runs a 10-second calibration to compute a dynamic `StandardScaler` baseline in real-time, matching the user's live microvolt range to the trained model's normalized space.
*   **Master Signal Processing Pipeline:** 
    *   **Digital DRL (50Hz IIR Notch Filter):** Mathematically scrubs powerline room hum.
    *   **Bandpass Filter:** 20Hz - 450Hz 4th-order Butterworth filter to isolate true motor unit action potentials from movement artifacts.
    *   **EMG Linear Envelope:** Full-wave rectification + 5Hz lowpass filter to extract the smooth contraction curve.
*   **Temporal Smoothing (Functional UI Accuracy):** Translates statistical accuracy to functional accuracy using a rolling memory buffer (majority voting) and a strict \>55-60% confidence threshold to debounce output glitches.
*   **Reaction Time Purge:** Training pipeline explicitly drops the first 600ms of gesture blocks to eliminate human reaction-time latency, ensuring the XGBoost model only learns peak-contraction signatures.

## 🛠️ Hardware Requirements

*   **Amplifier:** Anthriq (formerly Nexstem) Biosignal Amplifier
*   **DAQ:** National Instruments (NI) DAQ (USB-600X series)
*   **Electrodes:** Dry Gold-Plated Electrodes or Wet Gel Patches
*   **Wiring Scheme (Differential Input):**
    *   `AI 0`: Positive (Target Muscle)
    *   `AI 4`: Negative (Target Muscle)
    *   `AGND`: Analog Ground (e.g., Elbow bone)
    *   `DRL`: Driven Right Leg / Reference (e.g., Wrist bone)
    *   *Power:* True Dual-Rail Power Supply (+5V, GND, -5V)

## 🗂️ Project Architecture

*   **`dataset_collector.py` & `finger_data_collector.py`:** Tkinter GUI-based scripts that prompt the user to hold specific gestures and log synchronized EMG data to CSVs.
*   **`Model_Training.py`:** The offline pipeline to process data, apply the "Reaction Time Purge", extract up to 16 time-domain/frequency-domain features, and train the XGBoost classifiers.
*   **`live_xgboost_inference.py`:** The enterprise-grade live engine. It handles:
    *   Continuous DAQ hardware streaming via `nidaqmx`.
    *   Dual-Buffer Filtering (1000-sample background rolling buffer to prevent edge ringing, slicing 300 samples for feature extraction).
    *   Guided onboarding sequence for real-time `StandardScaler` calculation.
    *   Temporal smoothing and live predictions.
*   **`visual_demo3_inference.py` / `Updated_final_inference.py`:** Application-level scripts bridging the ML predictions with physical/UI outputs (like Servo control for a mechanical arm or PyAutoGUI).

## 🚀 Getting Started

### 1. System Requirements
- Windows OS (Required for native NI MAX and driver compatibility).
- Python 3.8+

### 2. Install Dependencies
```bash
pip install nidaqmx numpy pandas scipy xgboost scikit-learn
```

### 3. Hardware Setup
1. Verify the dual-rail power supply is providing **exactly +5.0V and -5.0V** to the Anthriq amplifier before powering on.
2. Connect the DAQ to the PC and verify it is recognized as `Dev1` using the **NI MAX** application.

### 4. Running Live Inference
Strap the electrodes to your forearm, ensuring proper placement (either general extensor/flexor for Gross Motor, or horizontal flexors for Fine Motor).

```bash
python live_xgboost_inference.py
```
*Follow the on-screen prompts for the 10-second Guided Onboarding calibration. Keep your arm completely relaxed when instructed!*

## 📚 Future Scope
- Integration with physical 3D-printed prosthetic limbs using Serial-to-Arduino communication ("Sensor Hijack" method).
- Publishing the system architecture and USPs in IEEE format for BCI research.
