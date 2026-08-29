from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .config import SimulationSettings
from .schemas import SimulatorOverrides


@dataclass(slots=True)
class ChipState:
    temperature_c: float
    latent: np.ndarray
    previous_drive: float = 0.0


class BM1373ChipSimulator:
    """Behavioural simulator, not a transistor-level or protocol-accurate emulator."""

    def __init__(
        self,
        chip_id: int,
        settings: SimulationSettings,
        overrides: SimulatorOverrides,
        seed: int,
    ) -> None:
        self.chip_id = chip_id
        self.settings = settings
        self.overrides = overrides
        self.rng = np.random.default_rng(seed + chip_id * 1009)
        self.ambient_temp_c = overrides.ambient_temp_c or settings.ambient_temp_c
        self.base_voltage_v = overrides.base_voltage_v or settings.base_voltage_v
        self.base_frequency_mhz = overrides.base_frequency_mhz or settings.base_frequency_mhz
        self.base_hashrate_ths = settings.target_total_ths / settings.chip_count
        self.noise_scale = overrides.noise_scale
        self.coupling_strength = overrides.coupling_strength

        latent_size = settings.latent_size
        raw = self.rng.normal(0.0, 1.0, size=(latent_size, latent_size))
        spectral_radius = max(abs(np.linalg.eigvals(raw)))
        self.recurrent = raw * (0.88 / max(float(spectral_radius), 1e-9))
        self.input_weights = self.rng.normal(0.0, 0.6, size=latent_size)
        self.bias = self.rng.normal(0.0, 0.08, size=latent_size)
        self.state = ChipState(
            temperature_c=self.ambient_temp_c + 4.0 + chip_id,
            latent=np.zeros(latent_size, dtype=float),
        )

    def step(self, input_drive: float, neighbour_signal: float, step_index: int) -> dict[str, float | int]:
        drive = float(np.clip(input_drive, -1.0, 1.0))
        coupled_drive = drive + self.coupling_strength * neighbour_signal
        process_noise = self.rng.normal(0.0, self.noise_scale, size=self.state.latent.shape)
        self.state.latent = np.tanh(
            self.recurrent @ self.state.latent
            + self.input_weights * coupled_drive
            + self.bias
            + process_noise
        )

        latent_mean = float(np.mean(self.state.latent))
        latent_energy = float(np.mean(np.square(self.state.latent)))
        drive_delta = drive - self.state.previous_drive

        frequency_mhz = self.base_frequency_mhz * (
            1.0 + 0.055 * drive + 0.018 * latent_mean + 0.008 * math.sin(step_index / 9.0)
        )
        voltage_v = self.base_voltage_v + 0.018 * drive + 0.006 * latent_mean

        voltage_margin = voltage_v - (0.92 + frequency_mhz / 2600.0)
        thermal_pressure = max(0.0, self.state.temperature_c - 72.0) / 35.0
        instability = max(0.0, -voltage_margin) * 2.2 + thermal_pressure + abs(drive_delta) * 0.015
        hardware_error_rate = float(np.clip(0.00025 + instability**2 * 0.035, 0.0, 0.22))

        frequency_ratio = frequency_mhz / self.base_frequency_mhz
        thermal_efficiency = max(0.72, 1.0 - max(0.0, self.state.temperature_c - 78.0) * 0.006)
        hashrate_ths = self.base_hashrate_ths * frequency_ratio * thermal_efficiency * (1.0 - hardware_error_rate)

        dynamic_power = 20.0 * (voltage_v / self.base_voltage_v) ** 2 * frequency_ratio
        power_w = dynamic_power + 1.8 + latent_energy * 1.4
        target_temp = self.ambient_temp_c + power_w * 1.65
        self.state.temperature_c += (target_temp - self.state.temperature_c) * 0.045
        self.state.temperature_c += float(self.rng.normal(0.0, 0.025))

        nonce_latency_us = max(
            1.0,
            84.0 / max(hashrate_ths, 0.05)
            * (1.0 + hardware_error_rate * 8.0)
            * (1.0 + abs(latent_mean) * 0.09)
            + float(self.rng.normal(0.0, 0.25)),
        )
        accepted_share_rate = max(
            0.0,
            hashrate_ths * 0.31 * (1.0 - hardware_error_rate)
            + float(self.rng.normal(0.0, 0.025)),
        )

        self.state.previous_drive = drive
        return {
            "step": step_index,
            "chip_id": self.chip_id,
            "input_drive": drive,
            "hashrate_ths": float(hashrate_ths),
            "voltage_v": float(voltage_v),
            "frequency_mhz": float(frequency_mhz),
            "temperature_c": float(self.state.temperature_c),
            "power_w": float(power_w),
            "hardware_error_rate": float(hardware_error_rate),
            "nonce_latency_us": float(nonce_latency_us),
            "accepted_share_rate": float(accepted_share_rate),
            "latent_energy": latent_energy,
        }


class DualBM1373Reservoir:
    def __init__(
        self,
        settings: SimulationSettings,
        overrides: SimulatorOverrides | None = None,
        seed: int | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.overrides = overrides or SimulatorOverrides()
        resolved_seed = settings.random_seed if seed is None else seed
        self.chips = [
            BM1373ChipSimulator(index, settings, self.overrides, resolved_seed)
            for index in range(settings.chip_count)
        ]

    @staticmethod
    def _normalise_signal(signal: list[float]) -> np.ndarray:
        values = np.asarray(signal, dtype=float)
        centred = values - float(np.mean(values))
        scale = float(np.std(centred))
        if scale < 1e-9:
            return np.zeros_like(centred)
        return np.clip(centred / (3.0 * scale), -1.0, 1.0)

    @staticmethod
    def _hash_encode(value: float, step: int, chip_id: int) -> float:
        quantised = int(round((value + 1.0) * 1_000_000))
        payload = f"{quantised}:{step}:{chip_id}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        integer = int.from_bytes(digest[:8], "big", signed=False)
        return (integer / ((1 << 64) - 1)) * 2.0 - 1.0

    def run(self, signal: list[float]) -> list[dict[str, float | int]]:
        normalised = self._normalise_signal(signal)
        telemetry: list[dict[str, float | int]] = []
        prior_latent_signals = [0.0 for _ in self.chips]

        for step_index, value in enumerate(normalised):
            next_latent_signals: list[float] = []
            for chip_id, chip in enumerate(self.chips):
                hashed_component = self._hash_encode(float(value), step_index, chip_id)
                drive = 0.78 * float(value) + 0.22 * hashed_component
                neighbour = prior_latent_signals[(chip_id - 1) % len(self.chips)]
                point = chip.step(drive, neighbour, step_index)
                telemetry.append(point)
                next_latent_signals.append(float(point["latent_energy"]) - 0.35)
            prior_latent_signals = next_latent_signals

        return telemetry
