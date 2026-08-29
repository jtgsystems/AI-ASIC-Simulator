from __future__ import annotations

import math

from app.config import SimulationSettings
from app.features import extract_reservoir_features, telemetry_summary
from app.schemas import SimulatorOverrides
from app.simulator import DualBM1373Reservoir


def test_dual_chip_simulation_produces_finite_telemetry() -> None:
    settings = SimulationSettings(chip_count=2, target_total_ths=10.0)
    signal = [math.sin(index / 5.0) for index in range(64)]
    simulator = DualBM1373Reservoir(settings, SimulatorOverrides(), seed=42)
    telemetry = simulator.run(signal)

    assert len(telemetry) == len(signal) * 2
    assert {point["chip_id"] for point in telemetry} == {0, 1}
    assert all(math.isfinite(float(point["hashrate_ths"])) for point in telemetry)
    assert all(float(point["hashrate_ths"]) > 0 for point in telemetry)


def test_feature_vector_is_stable_and_nonempty() -> None:
    settings = SimulationSettings(chip_count=2, target_total_ths=10.0)
    signal = [float(index % 7) for index in range(48)]
    simulator = DualBM1373Reservoir(settings, SimulatorOverrides(), seed=9)
    telemetry = simulator.run(signal)
    features = extract_reservoir_features(telemetry, settings.chip_count)
    summary = telemetry_summary(telemetry)

    assert features.size == 216
    assert math.isfinite(float(features.sum()))
    assert summary["maximum_temperature_c"] > settings.ambient_temp_c
