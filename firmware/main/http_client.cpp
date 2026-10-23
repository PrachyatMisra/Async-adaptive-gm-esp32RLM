#include "http_client.h"
#include <cstdio>
#include <cstring>
#include <esp_http_client.h>
#include <esp_log.h>

static const char *TAG = "HTTP_CLIENT";
static volatile int max_tokens = 96;

void set_max_tokens(int value) {
    if (value < 24) value = 24;
    if (value > 96) value = 96;
    max_tokens = value;
}

void post_inference_request() {
    int tokens[96];
    std::fill_n(tokens, 96, 0);
    tokens[0] = 101;
    tokens[1] = 2053;
    tokens[2] = 102;

    char request_body[160];
    const int body_length = snprintf(
        request_body, sizeof(request_body),
        "{\"tokens\":[%d,%d,%d],\"max_tokens\":%d}",
        tokens[0], tokens[1], tokens[2], max_tokens
    );
    esp_http_client_config_t config = {};
    config.url = "http://127.0.0.1:8000/api/v2/infer";
    config.method = HTTP_METHOD_POST;
    config.post_field = request_body;
    config.post_field_len = body_length;

    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (client == nullptr) {
        ESP_LOGE(TAG, "HTTP client initialization failed");
        return;
    }
    const esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "HTTP POST request sent successfully");
    } else {
        ESP_LOGE(TAG, "HTTP POST request failed with error: %s", esp_err_to_name(err));
    }
    esp_http_client_cleanup(client);
}

void stream_telemetry() {
    esp_websocket_client_config_t config = {};
    config.uri = "ws://127.0.0.1:8000/ws/v2/telemetry";
    esp_websocket_client_handle_t client = esp_websocket_client_init(&config);
    if (client == nullptr || esp_websocket_client_start(client) != ESP_OK) {
        ESP_LOGE(TAG, "WebSocket connection failed");
        return;
    }
    while (true) {
        esp_websocket_event_msg_t message;
        const esp_err_t err = esp_websocket_client_read_message(client, &message, portMAX_DELAY);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Failed to read telemetry: %s", esp_err_to_name(err));
            break;
        }
        if (message.type == WEBSOCKET_EVENT_DATA) {
            ESP_LOGI(TAG, "Received message: %.*s", message.len, static_cast<char *>(message.data));
        }
    }
    esp_websocket_client_stop(client);
    esp_websocket_client_cleanup(client);
}
