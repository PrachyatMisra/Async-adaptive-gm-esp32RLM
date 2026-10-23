/*
 * Arduino.h - minimal host-side stub of the Arduino/ESP32 API surface used
 * by firmware/rlm_esp32/rlm_esp32.ino.
 *
 * PURPOSE
 *   This is NOT an ESP32 firmware build and NOT a hardware test. It exists
 *   so that the sketch can be compiled on a workstation, which catches
 *   syntax errors, typos and API misuse that would otherwise only surface
 *   on a device that is not available in CI.
 *
 *   It does not emulate the hardware: the returned timings, heap sizes and
 *   ADC readings are fixed placeholders. Any result produced against this
 *   stub is a compile check, never a measurement.
 */
#ifndef ARDUINO_STUB_H
#define ARDUINO_STUB_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define ARDUINO_VARIANT "HOST_STUB"
#define PROGMEM

/* ---- Arduino core helpers ------------------------------------------- */
template <typename T, typename U>
static inline T constrain(T v, U lo, U hi) {
    return v < (T)lo ? (T)lo : (v > (T)hi ? (T)hi : v);
}
template <typename T>
static inline T min(T a, T b) { return a < b ? a : b; }
template <typename T>
static inline T max(T a, T b) { return a > b ? a : b; }

static inline uint32_t millis(void) { return 0; }
static inline uint32_t micros(void) { return 0; }
static inline void delay(uint32_t) {}
static inline void delayMicroseconds(uint32_t) {}

/* ---- Serial ---------------------------------------------------------- */
class StubSerial {
public:
    void begin(unsigned long) {}
    operator bool() const { return true; }
    int available() { return 0; }
    int read() { return -1; }
    void flush() {}
    size_t write(uint8_t) { return 1; }
    size_t print(const char* s) { return fputs(s, stdout), strlen(s); }
    size_t print(char c) { return fputc(c, stdout), 1; }
    size_t print(int v) { return printf("%d", v); }
    size_t print(unsigned v) { return printf("%u", v); }
    size_t print(long v) { return printf("%ld", v); }
    size_t print(unsigned long v) { return printf("%lu", v); }
    size_t print(double v) { return printf("%f", v); }
    size_t println() { return printf("\n"); }
    size_t println(const char* s) { return printf("%s\n", s); }
    size_t println(int v) { return printf("%d\n", v); }
    size_t println(unsigned v) { return printf("%u\n", v); }
    size_t println(unsigned long v) { return printf("%lu\n", v); }
    size_t println(double v) { return printf("%f\n", v); }
    int printf(const char*, ...) { return 0; }
};
extern StubSerial Serial;

/* ---- Analog / clock -------------------------------------------------- */
static inline uint32_t analogReadMilliVolts(uint8_t) { return 3300; }
static inline void setCpuFrequencyMhz(uint32_t) {}
static inline uint32_t getCpuFrequencyMhz(void) { return 240; }

/* ---- Chip information ------------------------------------------------ */
class StubEspClass {
public:
    uint32_t getFreeHeap() { return 200000; }
    uint32_t getMinFreeHeap() { return 180000; }
    uint32_t getSketchSize() { return 0; }
    uint32_t getFlashChipSize() { return 4 * 1024 * 1024; }
};
extern StubEspClass ESP;

/* ---- FreeRTOS (ESP-IDF flavoured) ------------------------------------- */
typedef void* TaskHandle_t;
typedef void* QueueHandle_t;
typedef void* SemaphoreHandle_t;
typedef unsigned int TickType_t;
typedef int BaseType_t;
typedef unsigned int UBaseType_t;

#define portMAX_DELAY ((TickType_t)0xffffffffUL)
#define pdTRUE  1
#define pdFALSE 0
#define pdPASS  1
#define pdMS_TO_TICKS(ms) ((TickType_t)(ms))

typedef struct { uint32_t owner; } portMUX_TYPE;
#define portMUX_INITIALIZER_UNLOCKED { (uint32_t)0 }

static inline void portENTER_CRITICAL(portMUX_TYPE* m) { (void)m; }
static inline void portEXIT_CRITICAL(portMUX_TYPE* m) { (void)m; }

static inline QueueHandle_t xQueueCreate(UBaseType_t, UBaseType_t) { return (QueueHandle_t)1; }
static inline BaseType_t xQueueSend(QueueHandle_t, const void*, TickType_t) { return pdTRUE; }
static inline BaseType_t xQueueReceive(QueueHandle_t, void*, TickType_t) { return pdTRUE; }
static inline void vQueueDelete(QueueHandle_t) {}

static inline SemaphoreHandle_t xSemaphoreCreateMutex(void) { return (SemaphoreHandle_t)1; }
static inline BaseType_t xSemaphoreTake(SemaphoreHandle_t, TickType_t) { return pdTRUE; }
static inline BaseType_t xSemaphoreGive(SemaphoreHandle_t) { return pdTRUE; }

static inline BaseType_t xTaskCreatePinnedToCore(void (*)(void*), const char*,
                                                 uint32_t, void*, unsigned,
                                                 TaskHandle_t*, BaseType_t) {
    return pdPASS;
}
static inline TaskHandle_t xTaskGetCurrentTaskHandle(void) { return (TaskHandle_t)1; }
static inline void vTaskDelay(TickType_t) {}
static inline void taskYIELD(void) {}
static inline uint32_t ulTaskNotifyTake(BaseType_t, TickType_t) { return 0; }
static inline void xTaskNotifyGive(TaskHandle_t) {}

#endif /* ARDUINO_STUB_H */
