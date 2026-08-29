from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from .benchmark import run_benchmark
from .config import SimulationSettings
from .features import extract_reservoir_features, telemetry_summary
from .schemas import BenchmarkRequest, BenchmarkResponse, SimulateRequest, SimulateResponse
from .simulator import DualBM1373Reservoir


DISCLAIMER = (
    "Behavioural digital-twin approximation only. It does not emulate BM1373 gates, "
    "firmware timing, serial protocol, or real silicon noise. Calibrate against measured hardware telemetry."
)
SETTINGS = SimulationSettings()
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Dual BM1373 Hash Reservoir Simulator",
    version="0.1.0",
    description="Simulates a dual-chip SHA-256 physical-reservoir experiment and benchmarks its readout features.",
)


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/simulate", response_model=SimulateResponse)
def simulate(request: SimulateRequest) -> SimulateResponse:
    simulator = DualBM1373Reservoir(
        settings=SETTINGS,
        overrides=request.overrides,
        seed=request.seed,
    )
    telemetry = simulator.run(request.signal)
    features = extract_reservoir_features(telemetry, SETTINGS.chip_count)
    summary = telemetry_summary(telemetry)
    summary["average_total_hashrate_ths"] *= SETTINGS.chip_count

    return SimulateResponse(
        disclaimer=DISCLAIMER,
        chip_count=SETTINGS.chip_count,
        target_total_ths=SETTINGS.target_total_ths,
        feature_count=int(features.size),
        features=features.round(8).tolist(),
        summary=summary,
        telemetry=telemetry if request.include_telemetry else None,
    )


@app.post("/benchmark", response_model=BenchmarkResponse)
def benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    result = run_benchmark(
        settings=SETTINGS,
        samples_per_class=request.samples_per_class,
        sequence_length=request.sequence_length,
        seed=request.seed,
    )
    return BenchmarkResponse(disclaimer=DISCLAIMER, **result)
