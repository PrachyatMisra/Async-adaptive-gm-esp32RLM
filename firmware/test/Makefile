# Native build of the real firmware engine for parity testing.
#   make               -> builds host_harness
#   make check         -> builds + runs parity vs golden vectors
#   make firmware-check-> host-compiles the ESP32 sketch against a stub
#                         (a compile check, NOT a firmware build or a
#                         hardware test)
#   make sanitize      -> ASan/UBSan build + golden parity run
CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++14 -Wall -Wextra

SRC  := ../rlm_esp32/src
GOLD := ../../tools/out/golden_vectors.txt
STUB := arduino_stub
SKETCH := ../rlm_esp32/rlm_esp32.ino

host_harness: host_harness.cpp $(SRC)/rlm_engine.cpp $(SRC)/tokenizer.cpp \
              $(SRC)/rlm_engine.h $(SRC)/rlm_config.h $(SRC)/rlm_weights.h
	$(CXX) $(CXXFLAGS) -I$(SRC) host_harness.cpp $(SRC)/rlm_engine.cpp \
	       $(SRC)/tokenizer.cpp -lm -o $@

check: host_harness
	./host_harness $(GOLD) --bench 20

# Host-side compile check of the ESP32 sketch.
#
# This is NOT a firmware build: PlatformIO and the ESP-IDF toolchain are not
# required and no ESP32 binary is produced. It compiles the sketch against a
# minimal Arduino/FreeRTOS stub so that syntax errors, typos and API misuse
# are caught on a workstation. It does not validate the code against real
# hardware behaviour.
firmware-check:
	@echo "== ESP32 sketch compile check | execution_source=host_native | toolchain=host stub (not an ESP32 build)"
	$(CXX) -std=c++14 -Wall -Wextra -fsyntax-only \
	       -I$(STUB) -I$(SRC) -x c++ $(SKETCH)
	@echo "sketch compiles cleanly against the host stub"

# ASan + UBSan build of the engine, run against the golden vectors.
sanitize:
	$(CXX) -O1 -g -std=c++14 -fsanitize=address,undefined \
	       -fno-omit-frame-pointer -I$(SRC) \
	       host_harness.cpp $(SRC)/rlm_engine.cpp $(SRC)/tokenizer.cpp \
	       -lm -o host_harness_asan
	./host_harness_asan $(GOLD) --bench 3

clean:
	rm -f host_harness host_harness_asan

.PHONY: check firmware-check sanitize clean
