import { useState, useEffect } from 'react';

export interface TelemetryPayload {
  latency_ms: number;
  current_window: number;
  g_t: number;
  active_core: number;
  memory_free_bytes: number;
}

export function useTelemetry() {
  const [data, setData] = useState<TelemetryPayload | null>(null);
  const [connected, setConnected] = useState(false);
  const [history, setHistory] = useState<TelemetryPayload[]>([]);

  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimeout: ReturnType<typeof setTimeout>;
    const historyBuffer: TelemetryPayload[] = [];

    const connect = () => {
      const apiBase = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000';
      ws = new WebSocket(apiBase.replace(/^http/, 'ws') + '/ws/v2/telemetry');
      ws.onopen = () => setConnected(true);
      ws.onmessage = (event) => {
        const newData = JSON.parse(event.data);
        setData(newData);
        historyBuffer.push(newData);
        if (historyBuffer.length > 20) {
          historyBuffer.shift();
        }
        setHistory([...historyBuffer]);
      };
      ws.onclose = () => {
        setConnected(false);
        reconnectTimeout = setTimeout(connect, 2000);
      };
    };

    connect();
    return () => {
      clearTimeout(reconnectTimeout);
      if (ws) ws.close();
    };
  }, []);

  return { data, connected, history };
}
