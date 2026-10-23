#include "http_client.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <esp_heap_caps.h>

void inference_task(void *pvParameters) {
    while (true) {
        post_inference_request();
        vTaskDelay(pdMS_TO_TICKS(1000)); // Delay for 1 second
    }
}

void telemetry_task(void *pvParameters) {
    while (true) {
        stream_telemetry();
        vTaskDelay(pdMS_TO_TICKS(1000)); // Delay for 1 second
    }
}

void app_main() {
    xTaskCreatePinnedToCore(inference_task, "inference_task", 2048, NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(telemetry_task, "telemetry_task", 2048, NULL, 5, NULL, 0);
}
#include "http_client.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <esp_heap_caps.h>

void inference_task(void *pvParameters) {
    while (true) {
        post_inference_request();
        vTaskDelay(pdMS_TO_TICKS(1000)); // Delay for 1 second
    }
}

void telemetry_task(void *pvParameters) {
    while (true) {
        stream_telemetry();
        vTaskDelay(pdMS_TO_TICKS(1000)); // Delay for 1 second
    }
}

void app_main() {
    xTaskCreatePinnedToCore(inference_task, "inference_task", 2048, NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(telemetry_task, "telemetry_task", 2048, NULL, 5, NULL, 0);
}

void monitor_memory() {
    while (true) {
        size_t free_heap = esp_get_free_heap_size();
        if (free_heap < 120 * 1024) {
            ESP_LOGW(TAG, "Free heap below 120KB, reducing max_tokens");
            // Dynamically reduce max_tokens here
        }
        vTaskDelay(pdMS_TO_TICKS(1000)); // Delay for 1 second
    }
}

void app_main() {
    xTaskCreatePinnedToCore(inference_task, "inference_task", 2048, NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(telemetry_task, "telemetry_task", 2048, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(monitor_memory, "monitor_memory", 2048, NULL, 5, NULL, 0);
}
