# Dual BM1373 Hash Reservoir Simulator

A Dockerized behavioural simulation of two BM1373-like SHA-256 ASICs used as a proposed physical reservoir for narrow machine-learning experiments.

## What it simulates

- Two independently noisy virtual ASICs
- Approximately 10 TH/s combined nominal throughput
- Input-dependent frequency and voltage perturbations
- Thermal inertia, power, timing and hardware-error behaviour
- Weak cross-chip coupling
- SHA-256-derived input encoding
- Telemetry feature extraction
- A small Ridge Classifier readout
- Baseline-versus-reservoir synthetic benchmark

## What it does not simulate

This is **not** a transistor-level BM1373 emulator. It does not reproduce proprietary gates, exact serial commands, clock domains, work scheduling, nonce logic or measured silicon noise. Its numerical coefficients are hypotheses for software experiments. Replace or fit them using real telemetry before drawing hardware conclusions.

## Start it

```bash
git clone https://github.com/jtgsystems/AI-ASIC-Simulator.git
cd AI-ASIC-Simulator
docker compose up --build
```

Open:

- Dashboard: `http://localhost:8000`
- Interactive API: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

Stop it:

```bash
docker compose down
```

## Run tests

```bash
docker compose run --rm bm1373-simulator pytest -q
```

## Simulate a signal

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "signal": [0, 0.2, 0.7, 1, 0.4, -0.3, -0.8, -0.2, 0.1, 0.5],
    "seed": 1373,
    "include_telemetry": true,
    "overrides": {
      "ambient_temp_c": 24,
      "noise_scale": 0.012,
      "coupling_strength": 0.18
    }
  }'
```

## Run the synthetic benchmark

```bash
curl -X POST http://localhost:8000/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "samples_per_class": 80,
    "sequence_length": 96,
    "seed": 1373
  }'
```

The benchmark distinguishes three synthetic signal classes:

1. Smooth periodic input
2. Switching load
3. Drift with injected anomalies

It trains the same scaled Ridge Classifier on:

- Direct input statistics and spectrum
- Simulated BM1373 telemetry features

Do not treat a high synthetic score as proof that real hardware will work. The useful experiment is whether calibrated hardware telemetry improves held-out performance over strong software-only baselines.

## Calibration path for real hardware

1. Capture timestamped per-chip voltage, frequency, temperature, current, accepted work and hardware errors.
2. Replay known labelled input sequences at several voltage-frequency operating points.
3. Fit the simulator coefficients to the measured transient response.
4. Freeze the simulator parameters.
5. Compare software baseline, simulated reservoir and physical reservoir on the same held-out test set.
6. Require repeated improvements across seeds and operating days before claiming value.

## API overview

### `POST /simulate`

Accepts a numeric time series and returns telemetry, 216 engineered features and summary metrics.

### `POST /benchmark`

Generates a labelled synthetic dataset, trains baseline and reservoir readouts and returns accuracy plus a confusion matrix.

---

## 🏆 Created by JTG Systems

<div align="center">

<a href="https://jtgsystems.com">
  <img src="https://raw.githubusercontent.com/jtgsystems/.github/main/assets/jtgsystems-attribution-banner.jpg" alt="JTG Systems - Support & Sponsorship" width="750"/>
</a>

**Engineered with pride by [JTG Systems](https://jtgsystems.com)**  
*Enterprise Systems Architecture, Custom Workstations & AI Solutions*

🌐 **Website**: [jtgsystems.com](https://jtgsystems.com)  
📞 **Contact**: (905) 892-4555  
☕ **Tips & Sponsorship**: `jtgsystems@gmail.com`

</div>

