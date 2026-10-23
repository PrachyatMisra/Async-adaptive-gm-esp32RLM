# Wokwi ESP32 Simulation

The repository includes a Wokwi simulation for the Arduino ESP32 firmware.
It uses an ESP32 DevKit-compatible board and a potentiometer connected to GPIO34
to exercise the firmware's simulated battery input.

## Build and launch

From the repository root:

```bash
./.venv/bin/pio run -d firmware/rlm_esp32 -e wokwi
```

Then open the repository in VS Code and use the Wokwi extension command
`Wokwi: Start Simulator`. The Wokwi serial monitor runs at 115200 baud.

## Serial commands

Enter these commands in the Wokwi serial monitor:

```text
PING
STAT
INFER the first act is clumsy but the final scene is magnificent
CHAT the movie was a brilliant masterpiece
MODE PERF
MODE BAL
MODE ECO
BATT 3600
BATT AUTO
BENCH 5
```

The potentiometer is connected to GPIO34, matching the firmware's 100k/100k
battery-divider input. `BATT AUTO` reads that simulated ADC path. The Wokwi
simulation validates firmware behavior, serial protocol, adaptive recursion,
and telemetry formatting; it does not reproduce physical current draw,
battery impedance, or the optional APB RTL coprocessor timing.

The `wokwi` environment sets `RLM_MAX_SEQ=20` so the classic ESP32 DevKit
memory model can link the large attention workspace. The production `esp32dev`
environment keeps the project's 96-token configuration.

## Troubleshooting

If the simulator says the firmware is missing, build the `wokwi` environment
again. The paths are declared in `wokwi.toml` and point to PlatformIO's `.pio`
output. If Wokwi asks for a token, sign in to Wokwi and configure the token in
the extension; the token is not stored in this repository.