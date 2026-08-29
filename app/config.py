from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulationSettings:
    random_seed: int = int(os.getenv("SIM_RANDOM_SEED", "1373"))
    chip_count: int = int(os.getenv("SIM_CHIP_COUNT", "2"))
    target_total_ths: float = float(os.getenv("SIM_TARGET_TOTAL_THS", "10.0"))
    ambient_temp_c: float = float(os.getenv("SIM_AMBIENT_TEMP_C", "24.0"))
    base_voltage_v: float = float(os.getenv("SIM_BASE_VOLTAGE_V", "1.15"))
    base_frequency_mhz: float = float(os.getenv("SIM_BASE_FREQUENCY_MHZ", "525.0"))
    latent_size: int = int(os.getenv("SIM_LATENT_SIZE", "24"))

    def validate(self) -> None:
        if self.chip_count < 1:
            raise ValueError("chip_count must be at least 1")
        if self.target_total_ths <= 0:
            raise ValueError("target_total_ths must be greater than 0")
        if self.latent_size < 4:
            raise ValueError("latent_size must be at least 4")
