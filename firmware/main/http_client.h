#ifndef HTTP_CLIENT_H
#define HTTP_CLIENT_H

#include <esp_http_client.h>
#include <esp_websocket_client.h>

void post_inference_request();
void stream_telemetry();
void set_max_tokens(int value);

#endif // HTTP_CLIENT_H
