#!/usr/bin/env python3
"""
llm-proxy: Multi-purpose Proxy for LLM API & Brave Search

DESIGN
------
This proxy runs on a server in your LAN. Clients (any PC in the same LAN)
simply set:

    export LLM_CLI_PROXY_URL=http://<proxy-server>:8080

No API keys or model names needed on clients! The proxy injects them server-side.

The proxy intelligently routes requests:

  1. /chat/completions  → Upstream LLM API (OpenAI-compatible)
  2. /web_search        → Brave Search API
  3. Any other path     → Forward proxy (HTTP/HTTPS) for general web requests

The proxy inspects the request body to determine the type when path-based
routing is ambiguous.

ENVIRONMENT VARIABLES (set on the proxy server):
  LLM_CLI_API_KEY              - Required. Upstream LLM API key.
  LLM_CLI_API_URL              - Actual LLM API base (default: http://localhost:11434/v1)
  LLM_CLI_MODEL                - Model to inject into client requests (e.g. gpt-4o).
                                 The proxy injects this server-side.
  BRAVE_API_KEY                - Brave Search API key (for /web_search routing).
  PROXY_PORT                   - Port to listen on (default: 8080)
  LOG_LEVEL                    - DEBUG, INFO, WARNING, ERROR (default: INFO)

CLIENT USAGE (any PC in LAN):
  export LLM_CLI_PROXY_URL=http://<proxy-ip>:8080
  # No LLM_CLI_API_KEY, LLM_CLI_MODEL, or BRAVE_API_KEY needed!
  # The proxy injects everything server-side.
  llm-cli-py
"""

import asyncio
import contextlib
import json
import logging
import os
import typing

import aiohttp
from aiohttp import web

# ── Configuration ──────────────────────────────────────────────────

DEFAULT_PORT = 8080
DEFAULT_LLM_API_URL = "http://localhost:11434/v1"
DEFAULT_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/llm/context"

LLM_API_KEY = os.environ.get("LLM_CLI_API_KEY", "").strip()
LLM_API_URL = os.environ.get("LLM_CLI_API_URL", DEFAULT_LLM_API_URL).rstrip("/")
LLM_MODEL = os.environ.get("LLM_CLI_MODEL", "").strip()
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "").strip()
PROXY_PORT = int(os.environ.get("PROXY_PORT", str(DEFAULT_PORT)))

# Maximum accepted request body size in bytes (aiohttp's client_max_size).
# LLM chat requests can grow large as conversation history accumulates
# (tool results, file contents, search snippets). The aiohttp default is
# only 1MB, which causes intermittent HTTP 413 "Content Too Large" errors.
# Default to 100MB; override with PROXY_MAX_BODY_SIZE (bytes).
DEFAULT_MAX_BODY_SIZE = 100 * 1024 * 1024  # 100 MiB
MAX_BODY_SIZE = int(os.environ.get("PROXY_MAX_BODY_SIZE", str(DEFAULT_MAX_BODY_SIZE)))


def _get_local_ips() -> list[str]:
    """Get all non-loopback IPv4 addresses of this machine (no DNS resolution).

    Strategy (fastest first):
      1. UDP connect trick — pure Python, no subprocess, no DNS, cross-platform.
      2. /proc/net/fib_trie — Linux only, no subprocess, no DNS.
      3. Platform-specific subprocess fallback (short timeout).
      4. If all else fails, return empty list (caller shows ``<IP>``).
    """
    import re
    import socket
    import subprocess
    import sys

    ips: set[str] = set()

    # ── 1. UDP connect trick ──────────────────────────────────────
    # Connect a UDP socket to a non-routable address; the kernel
    # picks the best source IP without actually sending a packet.
    for dst in ("10.255.255.255", "8.8.8.8"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.1)
            s.connect((dst, 1))
            ip = s.getsockname()[0]
            s.close()
            if ip and not ip.startswith("127."):
                ips.add(ip)
                break  # primary IP is enough for the banner
        except Exception:
            pass

    # ── 2. /proc/net/fib_trie (Linux) ────────────────────────────
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/net/fib_trie") as f:
                lines = f.read().split("\n")
            for i, line in enumerate(lines):
                if "/32 host LOCAL" in line and i > 0:
                    m = re.search(r"(\d+\.\d+\.\d+\.\d+)", lines[i - 1])
                    if m:
                        ip = m.group(1)
                        if not ip.startswith("127."):
                            ips.add(ip)
        except (OSError, FileNotFoundError):
            pass

    # ── 3. Platform-specific subprocess fallback ─────────────────
    if not ips:
        try:
            if sys.platform == "win32":
                result = subprocess.run(["ipconfig"], capture_output=True, text=True, timeout=2)
                for line in result.stdout.splitlines():
                    m = re.search(r"IPv4 Address[^:]*:\s*(\d+\.\d+\.\d+\.\d+)", line)
                    if m:
                        ip = m.group(1)
                        if not ip.startswith("127."):
                            ips.add(ip)
            elif sys.platform == "darwin":
                result = subprocess.run(["ifconfig"], capture_output=True, text=True, timeout=2)
                for line in result.stdout.splitlines():
                    m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", line)
                    if m:
                        ip = m.group(1)
                        if not ip.startswith("127."):
                            ips.add(ip)
            else:  # Linux fallback
                result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    for ip in result.stdout.strip().split():
                        ip = ip.strip()
                        if ip and not ip.startswith("127.") and ":" not in ip:
                            ips.add(ip)
                if not ips:
                    result = subprocess.run(
                        ["ip", "-4", "addr", "show", "scope", "global"],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    for line in result.stdout.splitlines():
                        parts = line.strip().split()
                        for i, part in enumerate(parts):
                            if part == "inet" and i + 1 < len(parts):
                                ip = parts[i + 1].split("/")[0]
                                if ip and not ip.startswith("127."):
                                    ips.add(ip)
        except Exception:
            pass

    return sorted(ips)


# ── Logging ────────────────────────────────────────────────────────

_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

# ── Raw Data Parser for CONNECT Tunnel ──────────────────────────


class _RawDataParser:
    """A minimal parser for aiohttp's set_parser() that buffers raw bytes.

    After a successful CONNECT, aiohttp's HTTP parser must be replaced
    so that subsequent data from the client is not interpreted as HTTP.
    This parser simply accumulates all data in a buffer for reading.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._eof = False
        self._waiter: asyncio.Future[None] | None = None

    def feed_data(self, data: bytes) -> tuple[bool, bytes]:
        """Called by aiohttp's data_received(). Returns (eof, tail)."""
        self._buffer.extend(data)
        if self._waiter is not None and not self._waiter.done():
            self._waiter.set_result(None)
            self._waiter = None
        return (False, b"")

    def feed_eof(self) -> None:
        self._eof = True
        if self._waiter is not None and not self._waiter.done():
            self._waiter.set_result(None)
            self._waiter = None

    async def read(self, n: int = 65536) -> bytes:
        """Read up to n bytes, waiting for data if necessary."""
        if self._buffer:
            chunk = bytes(self._buffer[:n])
            self._buffer = self._buffer[n:]
            return chunk
        while not self._eof:
            self._waiter = asyncio.get_event_loop().create_future()
            try:
                await asyncio.wait_for(self._waiter, timeout=300)
            except TimeoutError:
                break
            if self._buffer:
                chunk = bytes(self._buffer[:n])
                self._buffer = self._buffer[n:]
                return chunk
        return b""


logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("llm-proxy")

# ── Request Inspection ────────────────────────────────────────────


def _is_chat_request(body: bytes) -> bool:
    """Check if the request body looks like a chat completion request."""
    if not body:
        return False
    try:
        data = json.loads(body)
        return bool(data.get("messages"))
    except (json.JSONDecodeError, TypeError):
        return False


def _is_web_search_request(body: bytes) -> bool:
    """Check if the request body looks like a web search request."""
    if not body:
        return False
    try:
        data = json.loads(body)
        return bool(data.get("query")) and "model" not in data
    except (json.JSONDecodeError, TypeError):
        return False


def _inspect_request(_method: str, path: str, body: bytes) -> str:
    """Inspect the request and return the route type.

    Returns one of: 'chat', 'web_search', 'forward_proxy', 'unknown'
    """
    if path == "/chat/completions" or path.startswith("/chat/completions"):
        return "chat"
    if path == "/web_search" or path.startswith("/web_search"):
        return "web_search"

    if body and len(body) > 0:
        if _is_chat_request(body):
            log.info("Inspection: detected chat request by body analysis")
            return "chat"
        if _is_web_search_request(body):
            log.info("Inspection: detected web search request by body analysis")
            return "web_search"

    return "forward_proxy"


# ── Core Handlers ──────────────────────────────────────────────────


async def handle_chat(request: web.Request) -> web.StreamResponse:
    """Forward a chat completion request to the upstream LLM API.

    The proxy injects the model name from its own ``LLM_CLI_MODEL``
    environment variable into the request body. If the client already
    provided a model, the proxy's value takes precedence (overrides it).
    This allows clients to omit ``LLM_CLI_MODEL`` entirely — the proxy
    is the single source of truth for which model to use.
    """
    target_url = f"{LLM_API_URL}/chat/completions"
    log.info(f"Chat: {request.method} -> {target_url}")

    body = await request.read()
    headers = _build_upstream_headers(request)

    is_stream = False
    if LLM_MODEL:
        try:
            data = json.loads(body)
            data["model"] = LLM_MODEL
            is_stream = bool(data.get("stream", False))
            body = json.dumps(data).encode("utf-8")
            log.info(f"Injected model '{LLM_MODEL}' into chat request")
        except (json.JSONDecodeError, TypeError) as exc:
            log.warning(f"Could not parse request body to inject model: {exc}")

    if is_stream:
        # Pass through the SSE token stream live instead of buffering it.
        return await _forward_request_stream(
            request,
            method=request.method,
            url=target_url,
            headers=headers,
            body=body,
            params=dict(request.query),
            timeout=300,
            service="LLM Chat",
        )

    return await _forward_request(
        method=request.method,
        url=target_url,
        headers=headers,
        body=body,
        params=dict(request.query),
        timeout=300,
        service="LLM Chat",
    )


async def handle_web_search(request: web.Request) -> web.StreamResponse:
    """Forward a web search request to Brave Search API."""
    target_url = DEFAULT_BRAVE_SEARCH_URL
    log.info(f"Search: {request.method} -> {target_url}")

    body = await request.read()

    # Parse the query from the request body
    try:
        data = json.loads(body) if body else {}
        query = data.get("query", "")
    except (json.JSONDecodeError, TypeError):
        query = ""

    # Build Brave Search API headers
    brave_headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    }
    if BRAVE_API_KEY:
        brave_headers["X-Subscription-Token"] = BRAVE_API_KEY

    params = {"q": query}

    return await _forward_request(
        method="GET",
        url=target_url,
        headers=brave_headers,
        body=b"",
        params=params,
        timeout=60,
        service="Brave Search",
    )


async def handle_forward_proxy(request: web.Request) -> web.StreamResponse:
    """Handle forward proxy requests (absolute URI or CONNECT)."""
    url_str = str(request.url)
    if url_str.startswith(("http://", "https://")):
        return await _forward_http_request(request, url_str)

    log.warning(f"Forward proxy received non-absolute URI: {url_str}")
    return web.Response(
        status=400,
        text=json.dumps(
            {
                "error": "Forward proxy requires absolute URI (e.g., http://example.com/path). "
                "Send the full URL in the request line."
            }
        ),
        content_type="application/json",
    )


async def _forward_http_request(request: web.Request, target_url: str) -> web.Response:
    """Forward an HTTP request to the target URL."""
    log.info(f"Forward HTTP: {request.method} {target_url}")

    body = await request.read()
    headers = dict(request.headers)
    for h in ("Host", "Proxy-Connection", "Proxy-Authorization", "Transfer-Encoding"):
        headers.pop(h, None)

    return await _forward_request(
        method=request.method,
        url=target_url,
        headers=headers,
        body=body,
        params=dict(request.query),
        timeout=120,
        service="Forward Proxy",
    )


async def handle_connect(request: web.Request) -> web.StreamResponse:
    """Handle CONNECT tunneling for HTTPS forward proxy.

    NOTE: aiohttp's router does NOT match CONNECT requests (authority-form)
    to path patterns like /{tail:.*} because request.path is empty string.
    Therefore, this function is called from the middleware, not from the router.
    We use the Host header to determine the target host:port.

    After sending 200 Connection Established, we install a raw data parser
    and forward data bidirectionally at the transport level, bypassing
    aiohttp's HTTP layer entirely.
    """
    host_header = request.headers.get("Host", "")
    if ":" in host_header:
        host, port_str = host_header.split(":", 1)
    else:
        host = host_header
        port_str = "443"

    try:
        port = int(port_str)
    except ValueError:
        return web.Response(status=400, text="Invalid port")

    log.info(f"CONNECT tunnel: {host}:{port}")

    try:
        remote_reader, remote_writer = await asyncio.open_connection(host, port)
        log.info(f"CONNECT: connected to {host}:{port}")
    except Exception as e:
        log.error(f"CONNECT: failed to connect to {host}:{port}: {e}")
        return web.Response(
            status=502,
            text=json.dumps({"error": f"Failed to connect to {host}:{port}: {str(e)}"}),
            content_type="application/json",
        )

    transport = request.transport
    assert transport is not None, "transport must not be None for CONNECT"

    resp_headers = b"HTTP/1.1 200 Connection Established\r\nProxy-Agent: llm-proxy\r\n\r\n"
    transport.write(resp_headers)

    raw_parser = _RawDataParser()
    request.protocol.set_parser(raw_parser, data_received_cb=lambda: None)  # type: ignore[arg-type]
    request.protocol.keep_alive(False)

    log.info(f"CONNECT tunnel established: {host}:{port}, forwarding data...")

    async def _forward_to_remote() -> None:
        try:
            while True:
                data = await raw_parser.read(65536)
                if not data:
                    break
                remote_writer.write(data)
                await remote_writer.drain()
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            pass
        except Exception as e:
            log.debug(f"CONNECT tunnel [C->R] closed: {e}")
        finally:
            with contextlib.suppress(Exception):
                remote_writer.close()

    async def _forward_to_client() -> None:
        try:
            while True:
                data = await remote_reader.read(65536)
                if not data:
                    break
                transport.write(data)
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            pass
        except Exception as e:
            log.debug(f"CONNECT tunnel [R->C] closed: {e}")
        finally:
            with contextlib.suppress(Exception):
                transport.close()

    await asyncio.gather(
        _forward_to_remote(),
        _forward_to_client(),
    )

    return web.Response(status=200, text="OK")


# ── Shared Utilities ──────────────────────────────────────────────


def _build_upstream_headers(request: web.Request) -> dict[str, str]:
    """Build headers for upstream requests, injecting API key."""
    headers = {}
    for key, value in request.headers.items():
        key_lower = key.lower()
        if key_lower not in (
            "host",
            "authorization",
            "proxy-authorization",
            "transfer-encoding",
            "content-encoding",
            "content-length",
            "proxy-connection",
        ):
            headers[key] = value

    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    if "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"

    return headers


async def _forward_request_stream(
    request: web.Request,
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
    params: dict[str, str] | None = None,
    timeout: int = 300,
    service: str = "LLM Chat",
) -> web.StreamResponse:
    """Forward a request to the target URL and stream the SSE response back.

    Used for ``stream: true`` chat requests. The upstream ``text/event-stream``
    body is piped to the client chunk-by-chunk so reasoning / answer tokens
    appear live, rather than being buffered until the whole response completes.
    """
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=body,
                timeout=aiohttp.ClientTimeout(total=timeout, sock_read=timeout),
            ) as resp,
        ):
            resp_headers = {
                k: v
                for k, v in resp.headers.items()
                if k.lower()
                not in (
                    "transfer-encoding",
                    "content-encoding",
                    "content-length",
                    "alt-svc",
                )
            }
            # Stream back the body incrementally.
            stream_resp = web.StreamResponse(
                status=resp.status,
                headers=resp_headers,
            )
            await stream_resp.prepare(request)
            async for chunk in resp.content.iter_any():
                if chunk:
                    await stream_resp.write(chunk)
            await stream_resp.write_eof()
            log.info(f"{service}: streamed {method} {url} -> {resp.status}")
            return stream_resp
    except TimeoutError:
        log.error(f"{service}: Timeout after {timeout}s for {url}")
        return web.Response(
            status=504,
            text=json.dumps({"error": f"Upstream timeout after {timeout}s"}),
            content_type="application/json",
        )
    except aiohttp.ClientError as e:
        log.error(f"{service}: Connection error for {url}: {e}")
        return web.Response(
            status=502,
            text=json.dumps({"error": f"Upstream connection error: {str(e)}"}),
            content_type="application/json",
        )
    except Exception as e:
        log.error(f"{service}: Unexpected error for {url}: {e}")
        return web.Response(
            status=502,
            text=json.dumps({"error": f"Proxy error: {str(e)}"}),
            content_type="application/json",
        )


async def _forward_request(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
    params: dict[str, str] | None = None,
    timeout: int = 120,
    service: str = "Unknown",
) -> web.Response:
    """Forward a request to the target URL and return the response."""
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=body,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp,
        ):
            resp_body = await resp.read()
            resp_headers = {
                k: v
                for k, v in resp.headers.items()
                if k.lower()
                not in (
                    "transfer-encoding",
                    "content-encoding",
                    "content-length",
                    "alt-svc",
                )
            }
            log.info(f"{service}: {method} {url} -> {resp.status} ({len(resp_body)} bytes)")
            # Log the upstream error body for non-2xx responses so that
            # failures (e.g. HTTP 400) can be diagnosed from the log alone.
            if not (200 <= resp.status < 300):
                try:
                    err_text = resp_body.decode("utf-8", errors="replace")
                except Exception:
                    err_text = f"<{len(resp_body)} bytes, undecodable>"
                log.error(f"{service}: {method} {url} -> {resp.status} ERROR BODY: {err_text}")
            return web.Response(
                status=resp.status,
                body=resp_body,
                headers=resp_headers,
            )
    except TimeoutError:
        log.error(f"{service}: Timeout after {timeout}s for {url}")
        return web.Response(
            status=504,
            text=json.dumps({"error": f"Upstream timeout after {timeout}s"}),
            content_type="application/json",
        )
    except aiohttp.ClientError as e:
        log.error(f"{service}: Connection error for {url}: {e}")
        return web.Response(
            status=502,
            text=json.dumps({"error": f"Upstream connection error: {str(e)}"}),
            content_type="application/json",
        )
    except Exception as e:
        log.error(f"{service}: Unexpected error for {url}: {e}")
        return web.Response(
            status=502,
            text=json.dumps({"error": f"Proxy error: {str(e)}"}),
            content_type="application/json",
        )


# ── Middleware ─────────────────────────────────────────────────────


@web.middleware
async def connect_middleware(
    request: web.Request,
    handler: typing.Callable[[web.Request], typing.Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """Intercept CONNECT requests before routing."""
    if request.method == "CONNECT":
        return await handle_connect(request)
    return await handler(request)


# ── Main Dispatcher ───────────────────────────────────────────────


async def dispatcher(request: web.Request) -> web.StreamResponse:
    """Route requests based on path, method, and body inspection."""
    path = request.path
    method = request.method

    if path == "/":
        return await handle_root(request)

    if method == "CONNECT":
        return await handle_connect(request)

    body = await request.read()

    route_type = _inspect_request(method, path, body)

    if route_type == "chat":
        request._body = body
        return await handle_chat(request)

    if route_type == "web_search":
        request._body = body
        return await handle_web_search(request)

    url_str = str(request.url)
    if url_str.startswith(("http://", "https://")):
        request._body = body
        return await _forward_http_request(request, url_str)

    log.warning(f"Unknown route: {method} {path}")
    return web.Response(
        status=404,
        text=json.dumps(
            {
                "error": f"Unknown path: {path}. "
                "Use /chat/completions for LLM, /web_search for search, "
                "or send an absolute URI for forward proxy."
            }
        ),
        content_type="application/json",
    )


# ── Status Page ────────────────────────────────────────────────────


async def handle_root(_request: web.Request) -> web.Response:
    """Show proxy status and configuration."""
    ips = _get_local_ips()
    client_lines = (
        "\n".join(f"export LLM_CLI_PROXY_URL=http://{ip}:{PROXY_PORT}" for ip in ips)
        if ips
        else f"export LLM_CLI_PROXY_URL=http://<proxy-ip>:{PROXY_PORT}"
    )

    return web.Response(
        content_type="text/html",
        text=f"""<!DOCTYPE html>
<html lang="en">
<head>
  <title>llm-proxy</title>
  <meta charset="utf-8">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           max-width: 800px; margin: 40px auto; padding: 0 20px;
           background: #0d1117; color: #c9d1d9; }}
    h1 {{ color: #58a6ff; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
             padding: 20px; margin: 16px 0; }}
    .ok {{ color: #3fb950; }} .err {{ color: #f85149; }}
    code {{ background: #1f2937; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
    pre {{ background: #1f2937; padding: 12px; border-radius: 6px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td, th {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #30363d; }}
  </style>
</head>
<body>
  <h1>🔌 llm-proxy</h1>
  <p>Multi-purpose proxy for LLM API + Brave Search</p>

  <div class="card">
    <h2>Status</h2>
    <table>
      <tr><td>LLM API Key</td><td class="{"ok" if LLM_API_KEY else "err"}">
        {"✓ Configured" if LLM_API_KEY else "✗ NOT SET"}</td></tr>
      <tr><td>LLM Backend</td><td><code>{LLM_API_URL}</code></td></tr>
      <tr><td>LLM Model</td><td><code>{LLM_MODEL or "(client must provide)"}</code></td></tr>
      <tr><td>Brave Search Key</td><td class="{"ok" if BRAVE_API_KEY else "err"}">
        {"✓ Configured" if BRAVE_API_KEY else "✗ NOT SET"}</td></tr>
      <tr><td>Listening on</td><td><code>http://0.0.0.0:{PROXY_PORT}</code></td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Client Configuration</h2>
    <p>On any PC in the same LAN, set these environment variables:</p>
    <pre>{client_lines}
# No API keys or model needed!
# The proxy injects everything server-side.</pre>
  </div>

  <div class="card">
    <h2>Routes</h2>
    <table>
      <tr><th>Path</th><th>Destination</th><th>Description</th></tr>
      <tr><td><code>/chat/completions</code></td><td><code>{LLM_API_URL}/chat/completions</code></td>
          <td>LLM chat completions</td></tr>
      <tr><td><code>/web_search</code></td><td><code>Brave Search API</code></td>
          <td>Web search</td></tr>
      <tr><td><code>CONNECT host:port</code></td><td>Direct tunnel</td>
          <td>HTTPS forward proxy</td></tr>
      <tr><td><code>http://target/...</code></td><td>Target URL</td>
          <td>HTTP forward proxy (absolute URI)</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Request Inspection</h2>
    <p>The proxy inspects request bodies to detect API calls even
       when the path is non-standard. This enables transparent proxying
       for any HTTP client.</p>
  </div>
</body>
</html>""",
    )


def print_banner() -> None:
    """Print the startup banner."""
    ips = _get_local_ips()
    client_lines = (
        "\n".join(f"  export LLM_CLI_PROXY_URL=http://{ip}:{PROXY_PORT}" for ip in ips)
        if ips
        else f"  export LLM_CLI_PROXY_URL=http://<this-ip>:{PROXY_PORT}"
    )

    print(
        f"""
🔌 llm-proxy
Multi-purpose Proxy for LLM API + Brave Search
LLM Backend: {LLM_API_URL}
LLM Model:   {LLM_MODEL or "(client must provide)"}
LLM API Key: {"✓ SET" if LLM_API_KEY else "✗ NOT SET"}
Brave Search Key: {"✓ SET" if BRAVE_API_KEY else "✗ NOT SET"}
Listening on: http://0.0.0.0:{PROXY_PORT}
Client Setup (on any LAN PC):
{client_lines}
  # No API keys or model needed!
  # The proxy injects everything server-side.
Routes:
  POST /chat/completions  → LLM Chat API
  POST /web_search        → Brave Search API
  CONNECT host:port       → HTTPS Tunnel (forward proxy)
  http://target/...       → HTTP Forward Proxy
Request Inspection: ✓ Auto-detects API calls
""",
        flush=True,
    )


def main() -> None:
    """Start the proxy server."""
    if not LLM_API_KEY:
        log.warning("LLM_CLI_API_KEY is not set! LLM API calls will fail.")
    if not BRAVE_API_KEY:
        log.warning("BRAVE_API_KEY is not set! Brave Search calls will fail.")

    app = web.Application(middlewares=[connect_middleware], client_max_size=MAX_BODY_SIZE)
    app.router.add_route("*", "/{tail:.*}", dispatcher)

    print_banner()
    web.run_app(app, host="0.0.0.0", port=PROXY_PORT, print=lambda *_: None)


if __name__ == "__main__":
    main()
