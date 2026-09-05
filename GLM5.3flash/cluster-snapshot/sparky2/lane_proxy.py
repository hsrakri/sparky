#!/usr/bin/env python3
"""lane_proxy.py — QoS lane for a shared vLLM endpoint (2x DGX Spark).

One engine (TP=2) serves on :8888 with a single shared KV pool. Each lane is a
thin proxy that stamps per-lane defaults into /v1/chat/completions bodies:

  agent  lane (sparky1:8891): priority 0 (high), reasoning_effort=low default
  coding lane (sparky2:8892): priority 10 (lower), reasoning_effort=high default

Caller-provided values always win; the lane only fills gaps. `priority` is
honored once the engine runs with --scheduling-policy priority (staged in the
launcher for the next restart); until then it is carried but inert.

Usage: lane_proxy.py <listen_port> <lane_name> <priority> <reasoning_effort> [upstream]
"""
import json
import sys
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN_PORT = int(sys.argv[1])
LANE = sys.argv[2]
PRIORITY = int(sys.argv[3])
EFFORT = sys.argv[4]
UPSTREAM = sys.argv[5] if len(sys.argv) > 5 else "http://10.100.128.1:8888"

HOP_HEADERS = {"host", "content-length", "connection", "transfer-encoding",
               "keep-alive", "te", "trailer", "upgrade", "proxy-authorization"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet; one line per request
        sys.stderr.write("[%s] %s\n" % (LANE, fmt % args))

    def _proxy(self, body=None):
        url = UPSTREAM + self.path
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP_HEADERS}
        headers["X-Lane"] = LANE
        req = urllib.request.Request(url, data=body, headers=headers,
                                     method=self.command)
        try:
            resp = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as e:
            resp = e  # forward upstream error status + body as-is
        except Exception as e:
            self.send_response(502)
            msg = json.dumps({"error": {"message": "lane proxy: upstream unreachable: %s" % e,
                                        "type": "bad_gateway"}}).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        self.send_response(resp.status)
        is_chunked = False
        for k, v in resp.getheaders():
            if k.lower() in ("connection", "transfer-encoding"):
                is_chunked = is_chunked or (k.lower() == "transfer-encoding"
                                            and "chunked" in v.lower())
                continue
            self.send_header(k, v)
        if is_chunked:
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        # stream through (SSE-safe): small chunks, flush as they come
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            if is_chunked:
                self.wfile.write(("%x\r\n" % len(chunk)).encode())
                self.wfile.write(chunk + b"\r\n")
            else:
                self.wfile.write(chunk)
            self.wfile.flush()
        if is_chunked:
            self.wfile.write(b"0\r\n\r\n")

    def do_GET(self):
        self._proxy()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if self.path.startswith("/v1/chat/completions") and raw:
            try:
                d = json.loads(raw)
                d.setdefault("priority", PRIORITY)
                ctk = d.setdefault("chat_template_kwargs", {})
                ctk.setdefault("reasoning_effort", EFFORT)
                raw = json.dumps(d).encode()
            except Exception:
                pass  # not JSON we understand -> pass through untouched
        self._proxy(raw)


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), Handler)
    sys.stderr.write("[%s] lane on :%d -> %s (priority=%d effort=%s)\n"
                     % (LANE, LISTEN_PORT, UPSTREAM, PRIORITY, EFFORT))
    srv.serve_forever()
