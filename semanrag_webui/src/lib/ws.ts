type EventName = "pipeline-status" | "chat-stream" | "graph-update";
type Listener = (data: unknown) => void;

const BASE_WS = import.meta.env.VITE_WS_BASE ?? `ws://${window.location.host}`;

class WsClient {
  private ws: WebSocket | null = null;
  private listeners = new Map<EventName, Set<Listener>>();
  private attempt = 0;
  private maxDelay = 30_000;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private url: string;

  constructor(path = "/ws") {
    this.url = `${BASE_WS}${path}`;
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      this.attempt = 0;
    };

    this.ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as { event: EventName; data: unknown };
        this.listeners.get(msg.event)?.forEach((fn) => fn(msg.data));
      } catch { /* ignore non-json */ }
    };

    this.ws.onclose = () => this.reconnect();
    this.ws.onerror = () => this.ws?.close();
  }

  private reconnect() {
    const delay = Math.min(1000 * 2 ** this.attempt, this.maxDelay);
    this.attempt++;
    this.timer = setTimeout(() => this.connect(), delay);
  }

  on(event: EventName, fn: Listener) {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event)!.add(fn);
    return () => this.listeners.get(event)?.delete(fn);
  }

  send(event: string, data: unknown) {
    this.ws?.send(JSON.stringify({ event, data }));
  }

  disconnect() {
    if (this.timer) clearTimeout(this.timer);
    this.ws?.close();
    this.ws = null;
  }
}

export const wsClient = new WsClient();
