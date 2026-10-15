from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tensorflow import keras
from tensorflow.keras import layers


WINDOW_SIZE = 200
WINDOW_STEP = 100
DEFAULT_DATA_GLOB = "*_dataset.csv"


@dataclass
class Segment:
	participant_id: str
	gesture_label: str
	repetition_index: int
	voltage: np.ndarray


def apply_filters(signal: np.ndarray, sample_rate: int = 1000) -> np.ndarray:
	if signal.size < 8:
		return signal.astype(float)

	nyquist = sample_rate / 2.0
	low = 20.0 / nyquist
	high = 450.0 / nyquist
	b_band, a_band = butter(4, [low, high], btype="band")
	b_notch, a_notch = butter(2, [49.0 / nyquist, 51.0 / nyquist], btype="bandstop")

	filtered = filtfilt(b_band, a_band, signal)
	filtered = filtfilt(b_notch, a_notch, filtered)
	return filtered.astype(float)


def extract_features(window: np.ndarray) -> np.ndarray:
	window = np.asarray(window, dtype=float)
	if window.size == 0:
		return np.zeros(4, dtype=float)

	mav = np.mean(np.abs(window))
	rms = np.sqrt(np.mean(window**2))
	wl = np.sum(np.abs(np.diff(window))) if window.size > 1 else 0.0
	zc = np.sum(np.diff(np.sign(window)) != 0) if window.size > 1 else 0.0
	return np.array([rms, mav, wl, zc], dtype=float)


def load_segments(data_paths: Sequence[str | Path]) -> list[Segment]:
	segments: list[Segment] = []

	for path in data_paths:
		frame = pd.read_csv(path)
		required_columns = {"Participant_ID", "Gesture_Label", "Repetition", "Voltage"}
		missing = required_columns.difference(frame.columns)
		if missing:
			raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

		frame = frame.dropna(subset=["Participant_ID", "Gesture_Label", "Repetition", "Voltage"])
		frame["Repetition"] = frame["Repetition"].astype(int)

		grouped = frame.groupby(["Participant_ID", "Gesture_Label", "Repetition"], sort=False)
		for (participant_id, gesture_label, repetition_index), group in grouped:
			voltage = group.sort_values("Timestamp")["Voltage"].to_numpy(dtype=float)
			if voltage.size == 0:
				continue
			segments.append(
				Segment(
					participant_id=str(participant_id),
					gesture_label=str(gesture_label),
					repetition_index=int(repetition_index),
					voltage=voltage,
				)
			)

	return segments


def segment_to_windows(signal: np.ndarray, window_size: int = WINDOW_SIZE, step: int = WINDOW_STEP) -> list[np.ndarray]:
	if signal.size <= window_size:
		return [signal]

	windows: list[np.ndarray] = []
	for start in range(0, signal.size - window_size + 1, step):
		windows.append(signal[start : start + window_size])
	return windows


def build_feature_table(segments: Sequence[Segment]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	feature_rows: list[np.ndarray] = []
	labels: list[str] = []
	participants: list[str] = []
	timing_rows: list[tuple[float, float, float]] = []

	for segment in segments:
		filtered = apply_filters(segment.voltage)
		windows = segment_to_windows(filtered)
		for window in windows:
			feature_start = time.perf_counter()
			features = extract_features(window)
			feature_elapsed = time.perf_counter() - feature_start
			feature_rows.append(features)
			labels.append(segment.gesture_label)
			participants.append(segment.participant_id)
			timing_rows.append((feature_elapsed, len(window) / 1000.0, float(len(window))))

	return (
		np.vstack(feature_rows) if feature_rows else np.empty((0, 4), dtype=float),
		np.asarray(labels),
		np.asarray(participants),
		np.asarray(timing_rows, dtype=float),
	)


def build_cnn_inputs(segments: Sequence[Segment]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	samples: list[np.ndarray] = []
	labels: list[str] = []
	participants: list[str] = []

	for segment in segments:
		filtered = apply_filters(segment.voltage)
		windows = segment_to_windows(filtered)
		for window in windows:
			if window.size < WINDOW_SIZE:
				window = np.pad(window, (0, WINDOW_SIZE - window.size), mode="edge")
			elif window.size > WINDOW_SIZE:
				window = window[:WINDOW_SIZE]
			samples.append(window.reshape(-1, 1))
			labels.append(segment.gesture_label)
			participants.append(segment.participant_id)

	return (
		np.asarray(samples, dtype=float),
		np.asarray(labels),
		np.asarray(participants),
	)


def make_cnn(input_shape: tuple[int, int], class_count: int) -> keras.Model:
	model = keras.Sequential(
		[
			layers.Input(shape=input_shape),
			layers.Conv1D(32, 5, activation="relu"),
			layers.MaxPooling1D(2),
			layers.Conv1D(64, 3, activation="relu"),
			layers.MaxPooling1D(2),
			layers.Flatten(),
			layers.Dense(128, activation="relu"),
			layers.Dropout(0.25),
			layers.Dense(class_count, activation="softmax"),
		]
	)
	model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
	return model


def evaluate_classifier(name: str, model, x_test: np.ndarray, y_test: np.ndarray) -> dict:
	predict_start = time.perf_counter()
	y_pred = model.predict(x_test)
	predict_elapsed = time.perf_counter() - predict_start

	if y_pred.ndim > 1:
		y_pred = np.argmax(y_pred, axis=1)

	accuracy = accuracy_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred, average="weighted")
	report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
	conf = confusion_matrix(y_test, y_pred)

	return {
		"model": name,
		"accuracy": float(accuracy),
		"f1_weighted": float(f1),
		"inference_latency_seconds": float(predict_elapsed / max(len(x_test), 1)),
		"confusion_matrix": conf.tolist(),
		"classification_report": report,
	}


def evaluate_stability(model, x_test: np.ndarray, y_test: np.ndarray) -> float:
	if len(x_test) < 2:
		return 1.0

	y_pred = model.predict(x_test)
	if y_pred.ndim > 1:
		y_pred = np.argmax(y_pred, axis=1)

	if len(y_pred) < 2:
		return 1.0

	agreement = np.mean(y_pred[1:] == y_pred[:-1])
	return float(agreement)


def leave_one_participant_out(groups: np.ndarray) -> Iterable[tuple[np.ndarray, np.ndarray]]:
	splitter = GroupKFold(n_splits=min(len(np.unique(groups)), max(2, len(np.unique(groups)))))
	dummy_x = np.zeros((len(groups), 1))
	for train_index, test_index in splitter.split(dummy_x, groups=groups, y=groups):
		yield train_index, test_index


def run_benchmark(data_paths: Sequence[str | Path]) -> dict:
	segments = load_segments(data_paths)
	if not segments:
		raise ValueError("No usable gesture segments found in the provided CSV files.")

	feature_x, labels, participants, timing_rows = build_feature_table(segments)
	cnn_x, cnn_labels, cnn_participants = build_cnn_inputs(segments)

	label_encoder = LabelEncoder()
	y = label_encoder.fit_transform(labels)
	cnn_y = label_encoder.transform(cnn_labels)

	scaler = StandardScaler()
	feature_x = scaler.fit_transform(feature_x)

	participant_ids = np.unique(participants)
	if len(participant_ids) >= 2:
		train_ids, test_ids = train_test_split(participant_ids, test_size=0.25, random_state=42)
		train_mask = np.isin(participants, train_ids)
		test_mask = np.isin(participants, test_ids)
	else:
		train_mask, test_mask = np.ones(len(feature_x), dtype=bool), np.ones(len(feature_x), dtype=bool)

	x_train, x_test = feature_x[train_mask], feature_x[test_mask]
	y_train, y_test = y[train_mask], y[test_mask]

	if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
		x_train, x_test, y_train, y_test = train_test_split(
			feature_x, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
		)

	classical_models = {
		"SVM": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
		"Random Forest": RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42),
		"KNN": KNeighborsClassifier(n_neighbors=min(7, max(1, len(x_train) // 10))),
	}

	results: list[dict] = []
	for model_name, model in classical_models.items():
		fit_start = time.perf_counter()
		model.fit(x_train, y_train)
		fit_elapsed = time.perf_counter() - fit_start
		metrics = evaluate_classifier(model_name, model, x_test, y_test)
		metrics["train_time_seconds"] = float(fit_elapsed)
		metrics["stability"] = evaluate_stability(model, x_test, y_test)
		results.append(metrics)

	cnn_train_mask, cnn_test_mask = train_test_split(
		np.arange(len(cnn_x)), test_size=0.25, random_state=42, stratify=cnn_y if len(np.unique(cnn_y)) > 1 else None
	)
	cnn_model = make_cnn((WINDOW_SIZE, 1), class_count=len(label_encoder.classes_))
	cnn_start = time.perf_counter()
	cnn_model.fit(cnn_x[cnn_train_mask], cnn_y[cnn_train_mask], epochs=10, batch_size=32, verbose=0)
	cnn_train_elapsed = time.perf_counter() - cnn_start
	cnn_metrics = evaluate_classifier("CNN", cnn_model, cnn_x[cnn_test_mask], cnn_y[cnn_test_mask])
	cnn_metrics["train_time_seconds"] = float(cnn_train_elapsed)
	cnn_metrics["stability"] = evaluate_stability(cnn_model, cnn_x[cnn_test_mask], cnn_y[cnn_test_mask])
	results.append(cnn_metrics)

	best_accuracy = max(results, key=lambda row: row["accuracy"])
	fastest = min(results, key=lambda row: row["inference_latency_seconds"])
	best_stability = max(results, key=lambda row: row["stability"])

	summary = {
		"label_map": list(label_encoder.classes_),
		"participant_count": int(len(participant_ids)),
		"sample_count": int(len(feature_x)),
		"results": results,
		"best_accuracy": best_accuracy,
		"fastest_model": fastest,
		"best_stability": best_stability,
		"feature_time_seconds_mean": float(np.mean(timing_rows[:, 0])) if len(timing_rows) else 0.0,
	}

	return summary


def main() -> None:
	data_paths = sorted(Path(".").glob(DEFAULT_DATA_GLOB))
	if not data_paths:
		raise SystemExit(f"No dataset files found matching {DEFAULT_DATA_GLOB}")

	summary = run_benchmark(data_paths)
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D
from tensorflow.keras.utils import to_categorical


FEATURE_COLUMNS = ["mav", "rms", "wl", "zc"]
DEFAULT_WINDOW_SIZE = 200
DEFAULT_STEP_SIZE = 100


@dataclass
class BenchmarkResult:
	model_name: str
	split_name: str
	accuracy: float
	macro_f1: float
	train_seconds: float
	predict_seconds: float
	predict_ms_per_window: float
	stability: float


def extract_features(window: np.ndarray) -> np.ndarray:
	mav = float(np.mean(np.abs(window)))
	rms = float(np.sqrt(np.mean(window**2)))
	wl = float(np.sum(np.abs(np.diff(window))))
	zc = float(np.sum(np.diff(np.sign(window)) != 0))
	return np.asarray([mav, rms, wl, zc], dtype=np.float32)


def load_csv_file(csv_path: Path) -> pd.DataFrame:
	frame = pd.read_csv(csv_path)
	if "Voltage" not in frame.columns or "Gesture_Label" not in frame.columns:
		raise ValueError(f"Missing required columns in {csv_path.name}")

	if "Participant_ID" not in frame.columns:
		frame["Participant_ID"] = csv_path.stem.split("_")[0]

	if "Repetition" not in frame.columns:
		frame["Repetition"] = frame.groupby(["Participant_ID", "Gesture_Label"]).cumcount() + 1

	if "Timestamp" not in frame.columns:
		frame["Timestamp"] = np.arange(len(frame), dtype=np.float32)

	frame["source_file"] = csv_path.name
	return frame


def load_dataset(data_dir: Path) -> pd.DataFrame:
	csv_files = sorted(data_dir.rglob("*.csv"))
	if not csv_files:
		raise FileNotFoundError(f"No CSV files found under {data_dir}")

	frames = []
	for csv_file in csv_files:
		try:
			frames.append(load_csv_file(csv_file))
		except Exception as exc:
			print(f"Skipping {csv_file.name}: {exc}")

	if not frames:
		raise ValueError("No usable CSV files were loaded")

	combined = pd.concat(frames, ignore_index=True)
	combined = combined.sort_values(["Participant_ID", "Gesture_Label", "Repetition", "Timestamp"]) \
		.reset_index(drop=True)
	return combined


def build_windows(
	data: pd.DataFrame,
	window_size: int = DEFAULT_WINDOW_SIZE,
	step_size: int = DEFAULT_STEP_SIZE,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
	feature_rows: List[np.ndarray] = []
	raw_windows: List[np.ndarray] = []
	metadata_rows: List[Dict[str, object]] = []

	group_columns = ["Participant_ID", "Gesture_Label", "Repetition"]
	for (participant_id, gesture_label, repetition), group in data.groupby(group_columns, sort=True):
		signal = group.sort_values("Timestamp")["Voltage"].astype(np.float32).to_numpy()
		if signal.size < window_size:
			continue

		for window_index, start in enumerate(range(0, signal.size - window_size + 1, step_size)):
			window = signal[start:start + window_size]
			feature_rows.append(extract_features(window))
			raw_windows.append(window.reshape(window_size, 1))
			metadata_rows.append(
				{
					"Participant_ID": participant_id,
					"Gesture_Label": gesture_label,
					"Repetition": int(repetition) if pd.notna(repetition) else None,
					"Window_Index": window_index,
					"Window_Start": start,
				}
			)

	if not feature_rows:
		raise ValueError(
			f"No windows could be built. Check that each gesture block has at least {window_size} samples."
		)

	feature_matrix = np.asarray(feature_rows, dtype=np.float32)
	raw_tensor = np.asarray(raw_windows, dtype=np.float32)
	metadata = pd.DataFrame(metadata_rows)
	return feature_matrix, raw_tensor, metadata


def make_classical_models() -> Dict[str, object]:
	return {
		"SVM": make_pipeline(
			StandardScaler(),
			SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced"),
		),
		"Random Forest": RandomForestClassifier(
			n_estimators=250,
			max_depth=15,
			random_state=42,
			class_weight="balanced_subsample",
		),
		"KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
	}


def build_cnn(input_shape: Tuple[int, int], num_classes: int) -> Sequential:
	model = Sequential(
		[
			Input(shape=input_shape),
			Conv1D(32, kernel_size=5, activation="relu"),
			BatchNormalization(),
			MaxPooling1D(pool_size=2),
			Conv1D(64, kernel_size=3, activation="relu"),
			BatchNormalization(),
			MaxPooling1D(pool_size=2),
			Flatten(),
			Dense(128, activation="relu"),
			Dropout(0.3),
			Dense(num_classes, activation="softmax"),
		]
	)
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
	return model


def predict_with_timing(model: object, features: np.ndarray, is_keras: bool = False) -> Tuple[np.ndarray, float]:
	start = time.perf_counter()
	if is_keras:
		probabilities = model.predict(features, verbose=0)
		predictions = np.argmax(probabilities, axis=1)
	elif hasattr(model, "predict_proba"):
		probabilities = model.predict_proba(features)
		predictions = np.argmax(probabilities, axis=1)
	else:
		predictions = model.predict(features)
	elapsed = time.perf_counter() - start
	return np.asarray(predictions), elapsed


def stability_score(predictions: np.ndarray) -> float:
	if predictions.size <= 1:
		return 1.0

	switches = np.sum(predictions[1:] != predictions[:-1])
	return float(max(0.0, 1.0 - (switches / (predictions.size - 1))))


def train_and_evaluate_split(
	model_name: str,
	model: object,
	X_train: np.ndarray,
	X_test: np.ndarray,
	y_train: np.ndarray,
	y_test: np.ndarray,
	split_name: str,
	is_cnn: bool = False,
) -> Tuple[BenchmarkResult, object, np.ndarray]:
	train_start = time.perf_counter()
	if is_cnn:
		y_train_cat = to_categorical(y_train)
		y_test_cat = to_categorical(y_test, num_classes=y_train_cat.shape[1])
		early_stop = EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)
		model.fit(
			X_train,
			y_train_cat,
			validation_split=0.2,
			epochs=40,
			batch_size=32,
			verbose=0,
			callbacks=[early_stop],
		)
		train_seconds = time.perf_counter() - train_start
		predictions, predict_seconds = predict_with_timing(model, X_test, is_keras=True)
		probabilities = model.predict(X_test, verbose=0)
		_ = y_test_cat  # keep the shape normalization explicit for CNN runs
	else:
		model.fit(X_train, y_train)
		train_seconds = time.perf_counter() - train_start
		predictions, predict_seconds = predict_with_timing(model, X_test)
		probabilities = None

	accuracy = accuracy_score(y_test, predictions)
	macro_f1 = f1_score(y_test, predictions, average="macro")
	result = BenchmarkResult(
		model_name=model_name,
		split_name=split_name,
		accuracy=float(accuracy),
		macro_f1=float(macro_f1),
		train_seconds=float(train_seconds),
		predict_seconds=float(predict_seconds),
		predict_ms_per_window=float((predict_seconds / max(len(X_test), 1)) * 1000.0),
		stability=float(stability_score(predictions)),
	)
	return result, model, predictions


def evaluate_random_split(
	feature_matrix: np.ndarray,
	raw_tensor: np.ndarray,
	labels: np.ndarray,
	metadata: pd.DataFrame,
	label_encoder: LabelEncoder,
) -> Tuple[List[BenchmarkResult], Dict[str, object]]:
	X_train, X_test, y_train, y_test, raw_train, raw_test, meta_train, meta_test = train_test_split(
		feature_matrix,
		labels,
		raw_tensor,
		metadata,
		test_size=0.2,
		random_state=42,
		stratify=labels,
	)

	results: List[BenchmarkResult] = []
	trained_models: Dict[str, object] = {}

	for model_name, model in make_classical_models().items():
		result, fitted_model, _ = train_and_evaluate_split(
			model_name,
			model,
			X_train,
			X_test,
			y_train,
			y_test,
			split_name="random_split",
		)
		results.append(result)
		trained_models[model_name] = fitted_model

	cnn_model = build_cnn((raw_tensor.shape[1], raw_tensor.shape[2]), len(label_encoder.classes_))
	cnn_result, fitted_cnn, _ = train_and_evaluate_split(
		"CNN",
		cnn_model,
		raw_train,
		raw_test,
		y_train,
		y_test,
		split_name="random_split",
		is_cnn=True,
	)
	results.append(cnn_result)
	trained_models["CNN"] = fitted_cnn

	return results, trained_models


def evaluate_leave_one_participant_out(
	feature_matrix: np.ndarray,
	raw_tensor: np.ndarray,
	labels: np.ndarray,
	metadata: pd.DataFrame,
	label_encoder: LabelEncoder,
) -> List[BenchmarkResult]:
	results: List[BenchmarkResult] = []
	participants = metadata["Participant_ID"].astype(str).to_numpy()
	unique_participants = sorted(pd.unique(participants))

	if len(unique_participants) < 2:
		return results

	for held_out_participant in unique_participants:
		test_mask = participants == held_out_participant
		train_mask = ~test_mask

		if train_mask.sum() == 0 or test_mask.sum() == 0:
			continue

		X_train = feature_matrix[train_mask]
		X_test = feature_matrix[test_mask]
		y_train = labels[train_mask]
		y_test = labels[test_mask]
		raw_train = raw_tensor[train_mask]
		raw_test = raw_tensor[test_mask]

		for model_name, model in make_classical_models().items():
			result, _, _ = train_and_evaluate_split(
				model_name,
				model,
				X_train,
				X_test,
				y_train,
				y_test,
				split_name=f"lo_po_{held_out_participant}",
			)
			results.append(result)

		cnn_model = build_cnn((raw_tensor.shape[1], raw_tensor.shape[2]), len(label_encoder.classes_))
		cnn_result, _, _ = train_and_evaluate_split(
			"CNN",
			cnn_model,
			raw_train,
			raw_test,
			y_train,
			y_test,
			split_name=f"lo_po_{held_out_participant}",
			is_cnn=True,
		)
		results.append(cnn_result)

	return results


def summarize_results(results: List[BenchmarkResult]) -> pd.DataFrame:
	frame = pd.DataFrame([asdict(result) for result in results])
	summary = (
		frame.groupby("model_name", as_index=False)
		.agg(
			accuracy=("accuracy", "mean"),
			macro_f1=("macro_f1", "mean"),
			train_seconds=("train_seconds", "mean"),
			predict_seconds=("predict_seconds", "mean"),
			predict_ms_per_window=("predict_ms_per_window", "mean"),
			stability=("stability", "mean"),
		)
		.sort_values(["accuracy", "stability", "predict_ms_per_window"], ascending=[False, False, True])
		.reset_index(drop=True)
	)
	return summary


def main() -> None:
	parser = argparse.ArgumentParser(description="Train and benchmark EMG gesture models")
	parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing EMG CSV files")
	parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
	parser.add_argument("--step-size", type=int, default=DEFAULT_STEP_SIZE)
	parser.add_argument("--results-csv", type=Path, default=Path("model_benchmark_results.csv"))
	parser.add_argument("--results-json", type=Path, default=Path("model_benchmark_results.json"))
	args = parser.parse_args()

	data = load_dataset(args.data_dir)
	feature_matrix, raw_tensor, metadata = build_windows(data, window_size=args.window_size, step_size=args.step_size)

	label_encoder = LabelEncoder()
	labels = label_encoder.fit_transform(metadata["Gesture_Label"].astype(str))

	random_results, _ = evaluate_random_split(feature_matrix, raw_tensor, labels, metadata, label_encoder)
	lopo_results = evaluate_leave_one_participant_out(feature_matrix, raw_tensor, labels, metadata, label_encoder)

	all_results = random_results + lopo_results
	summary = summarize_results(all_results)

	detail_frame = pd.DataFrame([asdict(result) for result in all_results])
	detail_frame.to_csv(args.results_csv, index=False)
	summary.to_json(args.results_json, orient="records", indent=2)

	print("\nRandom split and leave-one-participant-out benchmark complete.\n")
	print(summary.to_string(index=False))

	best_accuracy = summary.iloc[0]
	fastest = summary.sort_values("predict_ms_per_window", ascending=True).iloc[0]
	best_generalization = summary.sort_values(["stability", "macro_f1"], ascending=[False, False]).iloc[0]

	print("\nBest overall accuracy:")
	print(best_accuracy[["model_name", "accuracy", "macro_f1"]].to_string())
	print("\nFastest model:")
	print(fastest[["model_name", "predict_ms_per_window", "train_seconds"]].to_string())
	print("\nBest generalization proxy:")
	print(best_generalization[["model_name", "stability", "macro_f1"]].to_string())


if __name__ == "__main__":
	main()

