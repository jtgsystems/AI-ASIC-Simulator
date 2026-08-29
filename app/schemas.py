from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SimulatorOverrides(BaseModel):
    ambient_temp_c: float | None = Field(default=None, ge=-20, le=100)
    base_voltage_v: float | None = Field(default=None, ge=0.5, le=2.0)
    base_frequency_mhz: float | None = Field(default=None, ge=50, le=1500)
    noise_scale: float = Field(default=0.012, ge=0.0, le=0.25)
    coupling_strength: float = Field(default=0.18, ge=0.0, le=1.0)


class SimulateRequest(BaseModel):
    signal: list[float] = Field(min_length=8, max_length=4096)
    seed: int = 1373
    overrides: SimulatorOverrides = Field(default_factory=SimulatorOverrides)
    include_telemetry: bool = True

    @field_validator("signal")
    @classmethod
    def reject_non_finite(cls, values: list[float]) -> list[float]:
        if any(value != value or value in (float("inf"), float("-inf")) for value in values):
            raise ValueError("signal must contain only finite values")
        return values


class BenchmarkRequest(BaseModel):
    samples_per_class: int = Field(default=80, ge=20, le=500)
    sequence_length: int = Field(default=96, ge=32, le=512)
    seed: int = 1373


class TelemetryPoint(BaseModel):
    step: int
    chip_id: int
    input_drive: float
    hashrate_ths: float
    voltage_v: float
    frequency_mhz: float
    temperature_c: float
    power_w: float
    hardware_error_rate: float
    nonce_latency_us: float
    accepted_share_rate: float
    latent_energy: float


class SimulateResponse(BaseModel):
    disclaimer: str
    chip_count: int
    target_total_ths: float
    feature_count: int
    features: list[float]
    summary: dict[str, float]
    telemetry: list[TelemetryPoint] | None


class BenchmarkResponse(BaseModel):
    disclaimer: str
    classes: list[str]
    train_samples: int
    test_samples: int
    baseline_accuracy: float
    reservoir_accuracy: float
    reservoir_minus_baseline: float
    feature_count: int
    confusion_matrix: list[list[int]]
