from dataclasses import dataclass
from typing import List, Sequence, Tuple


DEFAULT_GESTURES: Tuple[str, ...] = (
    "Fist",
    "Open palm",
    "Wrist up",
    "Wrist down",
    "Pinch",
)


@dataclass(frozen=True)
class ProtocolStep:
    label: str
    duration_seconds: float
    repetition_index: int | None = None
    is_recordable: bool = True


def build_study_protocol(
    gestures: Sequence[str] = DEFAULT_GESTURES,
    repetitions_per_gesture: int = 50,
    action_duration_seconds: float = 3.0,
    rest_duration_seconds: float = 3.0,
    baseline_duration_seconds: float = 5.0,
    transition_duration_seconds: float = 3.0,
) -> List[ProtocolStep]:
    protocol: List[ProtocolStep] = [
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
            next_gesture = gestures[gesture_index + 1]
            protocol.append(
                ProtocolStep(
                    f"Get Ready for {next_gesture}...",
                    transition_duration_seconds,
                    is_recordable=False,
                )
            )

    return protocol
