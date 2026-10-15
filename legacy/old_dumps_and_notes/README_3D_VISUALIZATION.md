# NadiCare 3D Visualization Setup

This project now includes **3D Virtual Visualizations** for gesture recognition using the Ursina Engine.

## Requirements

You must install the `ursina` graphics engine to run the new GUI files.

```bash
pip install ursina
```

## How to Run

1.  **Gross Motor Gestures (Fist, Open Hand)**
    Run the following command:
    ```bash
    python gui_3d_demo3_inference.py
    ```
    *Follow the on-screen calibration instructions.*

2.  **Fine Motor Gestures (Fingers)**
    Run the following command:
    ```bash
    python gui_3d_fingers_inference.py
    ```

## Features
- **Real-time 3D Hand Model**: A robotic hand mimics your gestures in real-time.
- **Professional Dashboard**: Dark theme with high-contrast neon data displays.
- **Visual Feedback**: Confidence bars and particle effects for strong signals.
- **Threaded Architecture**: Non-blocking data acquisition ensures smooth 60 FPS graphics.
