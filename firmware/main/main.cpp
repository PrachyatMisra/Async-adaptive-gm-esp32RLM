#include "http_client.h"
#include <algorithm>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <esp_heap_caps.h>
#include <esp_log.h>

static const char *TAG = "RLM_MAIN";

void inference_task(void *) {
    while (true) {
        post_inference_request();
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void telemetry_task(void *) {
    while (true) {
        stream_telemetry();
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void monitor_memory(void *) {
    while (true) {
        const size_t free_heap = esp_get_free_heap_size();
        if (free_heap < 120 * 1024) {
            ESP_LOGW(TAG, "Free heap below 120KB, reducing max_tokens");
            set_max_tokens(48);
        } else {
            set_max_tokens(96);
        }
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

extern "C" void app_main() {
    xTaskCreatePinnedToCore(inference_task, "inference_task", 3072, nullptr, 5, nullptr, 1);
    xTaskCreatePinnedToCore(telemetry_task, "telemetry_task", 3072, nullptr, 5, nullptr, 0);
    xTaskCreatePinnedToCore(monitor_memory, "monitor_memory", 2048, nullptr, 5, nullptr, 0);
}
