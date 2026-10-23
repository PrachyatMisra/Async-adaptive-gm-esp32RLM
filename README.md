# RLM ESP32 AAGM

An embedded recursive language-model research and verification project for ESP32-class hardware. The repository combines:

- A Phase 1 shared-weight recursive model study.
- A Phase 2 ESP32 AAGM runtime with adaptive gating, dual-core scheduling, and telemetry.
- Native C++ parity verification and Verilog control-path checks.
- A FastAPI backend and React control center with live telemetry, parity execution, local repository chat, and interactive 3-D pipeline exploration.

## Run the control center

Requirements: Python 3.11+ and Node.js 20+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd host/frontend
npm install
cd ../..
./launch_v2.sh
```

Open:

- Dashboard: <http://127.0.0.1:5173>
- Backend health: <http://127.0.0.1:8000/health>
- OpenAPI: <http://127.0.0.1:8000/docs>

The dashboard includes live five-field telemetry, progressive 12-vector parity verification, a Markdown local chat panel, an interactive 3-D force graph, and an expandable architecture blueprint.

## Verification

```bash
make -C firmware/test check
make -C verification/verilog check
curl http://127.0.0.1:8000/api/v1/verification/golden
```

The golden-vector endpoint executes the native harness for every request. Re-running the parity check in the dashboard requests a fresh result before animating the current samples.

## Repository layout

```text
app/                 Typed FastAPI services and routers
artifacts/           Verification JSON and analysis outputs
docs/                Research, engineering, and patent documentation
firmware/            ESP32 firmware and native parity harness
graphify-out/        Source graph index used by the 3-D dashboard
host/backend/        Canonical FastAPI entrypoint
host/frontend/       React control center
phase1/              Shared-weight recursive model study
tools/               Training, inference, and analysis utilities
verification/        Verilog control-path verification
```

## Hardware contract

The telemetry WebSocket payload remains:

```json
{
  "latency_ms": 0.0,
  "current_window": 0,
  "g_t": 0.0,
  "active_core": 0,
  "memory_free_bytes": 0
}
```

The firmware retains programmatic buffer filling and the 120 KB heap threshold used for adaptive token-window contraction.

## Research notes

- Phase 1 implementation details are in [phase1/README.md](phase1/README.md).
- Engineering findings are in [docs/ENGINEERING_AUDIT.md](docs/ENGINEERING_AUDIT.md).
- Verification coverage is in [docs/VERIFICATION_MATRIX.md](docs/VERIFICATION_MATRIX.md).
- The patent-oriented technical specification is in [docs/PATENT_SPECIFICATION.md](docs/PATENT_SPECIFICATION.md).

