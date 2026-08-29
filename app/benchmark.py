from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier

from .config import SimulationSettings
from .features import extract_input_baseline_features, extract_reservoir_features
from .schemas import SimulatorOverrides
from .simulator import DualBM1373Reservoir


CLASS_NAMES = ["smooth-periodic", "switching-load", "drifting-anomaly"]


def _make_signal(class_id: int, length: int, rng: np.random.Generator) -> list[float]:
    t = np.linspace(0.0, 1.0, length, endpoint=False)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    noise = rng.normal(0.0, 0.07, size=length)

    if class_id == 0:
        frequency = rng.uniform(2.0, 4.5)
        signal = np.sin(2.0 * math.pi * frequency * t + phase) + noise
    elif class_id == 1:
        frequency = rng.uniform(3.0, 6.0)
        signal = np.sign(np.sin(2.0 * math.pi * frequency * t + phase))
        signal = signal + noise
    else:
        frequency = rng.uniform(1.5, 3.0)
        signal = 0.55 * np.sin(2.0 * math.pi * frequency * t + phase) + noise
        anomaly_count = int(rng.integers(2, 6))
        for _ in range(anomaly_count):
            start = int(rng.integers(4, length - 5))
            width = int(rng.integers(2, min(10, length - start)))
            signal[start : start + width] += rng.uniform(0.8, 1.7)
        signal += np.linspace(0.0, rng.uniform(0.25, 0.8), length)

    amplitude = rng.uniform(0.75, 1.25)
    return (signal * amplitude).astype(float).tolist()


def run_benchmark(
    settings: SimulationSettings,
    samples_per_class: int,
    sequence_length: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    baseline_rows: list[np.ndarray] = []
    reservoir_rows: list[np.ndarray] = []
    labels: list[int] = []

    for class_id in range(len(CLASS_NAMES)):
        for sample_index in range(samples_per_class):
            signal = _make_signal(class_id, sequence_length, rng)
            simulator = DualBM1373Reservoir(
                settings=settings,
                overrides=SimulatorOverrides(noise_scale=0.01),
                seed=seed + class_id * 100_000 + sample_index,
            )
            telemetry = simulator.run(signal)
            baseline_rows.append(extract_input_baseline_features(signal))
            reservoir_rows.append(extract_reservoir_features(telemetry, settings.chip_count))
            labels.append(class_id)

    baseline_matrix = np.vstack(baseline_rows)
    reservoir_matrix = np.vstack(reservoir_rows)
    y = np.asarray(labels, dtype=int)
    indices = np.arange(y.size)
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )

    def make_model() -> Pipeline:
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("classify", RidgeClassifier(alpha=1.0)),
            ]
        )

    baseline_model = make_model()
    reservoir_model = make_model()
    baseline_model.fit(baseline_matrix[train_indices], y[train_indices])
    reservoir_model.fit(reservoir_matrix[train_indices], y[train_indices])

    baseline_predictions = baseline_model.predict(baseline_matrix[test_indices])
    reservoir_predictions = reservoir_model.predict(reservoir_matrix[test_indices])
    baseline_accuracy = float(accuracy_score(y[test_indices], baseline_predictions))
    reservoir_accuracy = float(accuracy_score(y[test_indices], reservoir_predictions))

    return {
        "classes": CLASS_NAMES,
        "train_samples": int(train_indices.size),
        "test_samples": int(test_indices.size),
        "baseline_accuracy": baseline_accuracy,
        "reservoir_accuracy": reservoir_accuracy,
        "reservoir_minus_baseline": reservoir_accuracy - baseline_accuracy,
        "feature_count": int(reservoir_matrix.shape[1]),
        "confusion_matrix": confusion_matrix(
            y[test_indices], reservoir_predictions, labels=list(range(len(CLASS_NAMES)))
        ).tolist(),
    }
