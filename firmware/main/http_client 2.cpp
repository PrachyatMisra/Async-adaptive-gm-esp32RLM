#include "http_client.h"
#include <esp_http_client.h>
#include <esp_log.h>

static const char *TAG = "HTTP_CLIENT";

void post_inference_request() {
    esp_http_client_config_t config = {
        .url = "http://127.0.0.1:8000/api/v2/infer",
        .method = HTTP_METHOD_POST,
        .post_field = {
            .buf = "{\"tokens\": [101, 2053, 102], \"max_tokens\": 96}",
            .buf_len = strlen("{\"tokens\": [101, 2053, 102], \"max_tokens\": 96}")
        }
    };

    esp_http_client_handle_t client = esp_http_client_init(&config);
    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "HTTP POST request sent successfully");
    } else {
        ESP_LOGE(TAG, "HTTP POST request failed with error: %s", esp_err_to_name(err));
    }
    esp_http_client_cleanup(client);
}
void stream_telemetry() {
    esp_websocket_client_config_t config = {
        .uri = "ws://127.0.0.1:8000/ws/v2/telemetry"
    };

    esp_websocket_client_handle_t client = esp_websocket_client_init(&config);
    esp_err_t err = esp_websocket_client_start(client);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "WebSocket connection started successfully");
    } else {
        ESP_LOGE(TAG, "WebSocket connection failed with error: %s", esp_err_to_name(err));
        return;
    }

    while (true) {
        esp_websocket_event_msg_t msg;
        err = esp_websocket_client_read_message(client, &msg, portMAX_DELAY);
        if (err == ESP_OK) {
            if (msg.type == WEBSOCKET_EVENT_DATA) {
                ESP_LOGI(TAG, "Received message: %.*s", msg.len, (char *)msg.data);
            }
        } else {
            ESP_LOGE(TAG, "Failed to read message with error: %s", esp_err_to_name(err));
            break;
        }
    }

    esp_websocket_client_stop(client);
    esp_websocket_client_cleanup(client);
}
