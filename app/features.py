from __future__ import annotations

from collections import defaultdict

import numpy as np


METRIC_NAMES = (
    "hashrate_ths",
    "voltage_v",
    "frequency_mhz",
    "temperature_c",
    "power_w",
    "hardware_error_rate",
    "nonce_latency_us",
    "accepted_share_rate",
    "latent_energy",
)


def _series_features(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0] * 8
    x = np.arange(values.size, dtype=float)
    trend = float(np.polyfit(x, values, 1)[0]) if values.size >= 2 else 0.0
    centred = values - float(np.mean(values))
    autocorrelation = 0.0
    if values.size >= 3:
        denominator = float(np.dot(centred, centred))
        if denominator > 1e-12:
            autocorrelation = float(np.dot(centred[:-1], centred[1:]) / denominator)
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
        trend,
        autocorrelation,
    ]


def extract_reservoir_features(telemetry: list[dict[str, float | int]], chip_count: int) -> np.ndarray:
    by_chip: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for point in telemetry:
        by_chip[int(point["chip_id"])].append(point)

    features: list[float] = []
    for chip_id in range(chip_count):
        points = by_chip[chip_id]
        for metric_name in METRIC_NAMES:
            values = np.asarray([float(point[metric_name]) for point in points], dtype=float)
            features.extend(_series_features(values))

    if chip_count >= 2:
        left = by_chip[0]
        right = by_chip[1]
        for metric_name in METRIC_NAMES:
            count = min(len(left), len(right))
            difference = np.asarray(
                [float(left[index][metric_name]) - float(right[index][metric_name]) for index in range(count)],
                dtype=float,
            )
            features.extend(_series_features(difference))

    return np.asarray(features, dtype=float)


def extract_input_baseline_features(signal: list[float]) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    derivative = np.diff(values, prepend=values[0])
    fft = np.abs(np.fft.rfft(values))
    spectral = fft[1:9] if fft.size > 1 else np.zeros(8)
    spectral = np.pad(spectral, (0, max(0, 8 - spectral.size)))[:8]
    return np.asarray(_series_features(values) + _series_features(derivative) + spectral.tolist(), dtype=float)


def telemetry_summary(telemetry: list[dict[str, float | int]]) -> dict[str, float]:
    if not telemetry:
        return {}
    return {
        "average_total_hashrate_ths": float(
            sum(float(point["hashrate_ths"]) for point in telemetry) / max(1, len(telemetry))
        ),
        "maximum_temperature_c": max(float(point["temperature_c"]) for point in telemetry),
        "average_power_per_chip_w": float(
            sum(float(point["power_w"]) for point in telemetry) / max(1, len(telemetry))
        ),
        "average_hardware_error_rate": float(
            sum(float(point["hardware_error_rate"]) for point in telemetry) / max(1, len(telemetry))
        ),
    }
