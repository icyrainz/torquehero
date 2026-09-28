"""Companion HTTP server: live pages for the rig's extra screens (SPEC 2, 8.6).

Standard library only. The game calls `publish_song(info, chart)` when a song
starts and `publish(snapshot)` every frame. `publish` serializes at most
STREAM_HZ times a second, on the game thread, so no server thread ever reads a
live game object. One broadcaster thread frames the latest JSON once per tick
and every client thread sends the same bytes.

Routes: `/` (page list and URLs), `/dash`, `/top`, `/state`, `/song` and
`/results` (latest JSON, `null` before the first publish), `/info`, `/events`
(server-sent events: `song` first, with the note results so far, then `state`
at about 20 Hz).
"""
from __future__ import annotations

import errno
import ipaddress
import json
import socket
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

from ..config import config_dir
from ..state import Snapshot, _finite

SETTINGS_FILE = "companion.json"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
STREAM_HZ = 20.0
KEEPALIVE_S = 2.0
MAX_CLIENTS = 8
MAX_CONNECTIONS = 32  # every open connection, not only /events; over it the new one is closed
CLOSE_TIMEOUT_S = 0.5
RESTART_BACK_S = 1.0  # song clock going back this far without publish_song = a restart

PAGES = {
    "/": ("index.html", "Pages and the URLs to open on another device"),
    "/dash": ("dash.html", "Dash display by the wheel (800x480)"),
    "/top": ("top.html", "Monitor above the triples (1920x1080)"),
}
ASSETS = {"/common.js": "common.js", "/common.css": "common.css"}
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8"}


class CompanionError(RuntimeError):
    pass


def check_port(port) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise CompanionError(f"companion: port must be a whole number from 0 to 65535, not {port!r}")
    return port


def check_host(host) -> str:
    if not isinstance(host, str) or not host.strip():
        raise CompanionError(f"companion: host must be a non-empty string, not {host!r}")
    return host.strip()


@dataclass
class CompanionSettings:
    """Persisted in `config_dir()/companion.json`, not in Config. `allowed_hosts`
    adds names a browser may use besides localhost and this machine's own."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    enabled: bool = True
    allowed_hosts: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path | None = None) -> CompanionSettings:
        """Defaults when the file is missing; CompanionError when it is unusable."""
        path = Path(path) if path else config_dir() / SETTINGS_FILE
        if not path.exists():
            return cls()
        try:
            d = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            raise CompanionError(f"companion: cannot read {path}: {e}") from e
        if not isinstance(d, dict):
            raise CompanionError(f"companion: {path} must hold a JSON object")
        s = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        check_host(s.host)
        check_port(s.port)
        if not isinstance(s.enabled, bool):
            raise CompanionError(f"companion: {path}: enabled must be true or false")
        if not isinstance(s.allowed_hosts, list) or not all(isinstance(h, str) for h in s.allowed_hosts):
            raise CompanionError(f"companion: {path}: allowed_hosts must be a list of names")
        return s

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else config_dir() / SETTINGS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path


def static_file(name: str) -> bytes:
    return resources.files(__package__).joinpath("static", name).read_bytes()


def lan_addresses() -> list[str]:
    """Best-effort IPv4 addresses another device on the LAN can reach."""
    addrs: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # UDP connect sends nothing; it only picks a route
            addrs.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addrs.append(str(info[4][0]))
    except OSError:
        pass
    out = [a for a in dict.fromkeys(addrs) if not a.startswith("127.")]
    return out or ["127.0.0.1"]


def own_names() -> set[str]:
    """Host names that mean this machine: localhost and its host names."""
    names = {"localhost"}
    for n in (socket.gethostname(), socket.getfqdn()):
        n = n.lower().rstrip(".")
        if n:
            short = n.split(".")[0]
            names |= {n, short, short + ".local"}
    return names


def host_of(header: str) -> str:
    """Host name from a Host header, without the port."""
    h = header.strip().lower()
    if h.startswith("["):
        return h[1:h.find("]")] if "]" in h else h
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


def is_ip(name: str) -> bool:
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


def _encode(value) -> str:
    """Compact JSON with non-finite floats as null. A string is parsed and
    re-encoded, so nothing raw ever reaches an SSE frame."""
    if isinstance(value, Snapshot):
        return value.to_json()
    if isinstance(value, str):
        value = json.loads(value, parse_constant=lambda _: None)
    elif hasattr(value, "to_dict"):
        value = value.to_dict()
    return json.dumps(_finite(value), separators=(",", ":"), allow_nan=False)


def _views(value) -> list[tuple[int, str]]:
    """(note index, result) for every finished note in a snapshot. A done note
    with no result (listen gates) counts as "done"."""
    notes = value.notes if isinstance(value, Snapshot) else value.get("notes") if isinstance(value, dict) else None
    out = []
    for v in notes or ():
        d = v if isinstance(v, dict) else asdict(v)
        if d.get("done") and isinstance(d.get("index"), int):
            out.append((d["index"], d.get("result") or "done"))
    return out


def _frame(event: str, data: str) -> bytes:
    return f"event: {event}\ndata: {data}\n\n".encode()


class _Hub:
    """Latest song and state as JSON text, plus the broadcast frame."""

    def __init__(self, min_interval: float, max_clients: int) -> None:
        self.cond = threading.Condition()
        self.stop = threading.Event()
        self.min_interval = min_interval
        self.max_clients = max_clients
        self.clients = 0
        self._due = 0.0
        self._phase: object = None
        self._warned = False
        self._song_warned = False
        self._last_now: float | None = None
        self.state = "null"
        self.state_ver = 0
        self._song_head: str | None = None
        self.song_ver = 0
        self.results: dict[int, str] = {}
        self.frame = b""
        self.seq = 0

    # --- game thread ---

    def publish(self, snapshot) -> None:
        now = time.monotonic()
        phase = snapshot.phase if isinstance(snapshot, Snapshot) else \
            snapshot.get("phase") if isinstance(snapshot, dict) else None
        if now < self._due and phase == self._phase:
            return
        self._due, self._phase = now + self.min_interval, phase
        try:
            if isinstance(snapshot, str):
                snapshot = json.loads(snapshot, parse_constant=lambda _: None)
            text, views = _encode(snapshot), _views(snapshot)
        except (TypeError, ValueError) as e:
            if not self._warned:
                self._warned = True
                print(f"companion: dropped a snapshot that is not JSON: {e}", file=sys.stderr)
            return
        t = snapshot.now if isinstance(snapshot, Snapshot) else \
            snapshot.get("now") if isinstance(snapshot, dict) else None
        restarted = isinstance(t, int | float) and self._last_now is not None and t < self._last_now - RESTART_BACK_S
        if isinstance(t, int | float):
            self._last_now = t
        with self.cond:
            if restarted:
                self.results = {}
            self.results.update(views)
            self.state = text
            self.state_ver += 1

    def publish_song(self, info, chart) -> None:
        try:
            head = f'{{"info":{_encode(info)},"chart":{_encode(chart)}'
        except (TypeError, ValueError) as e:
            if not self._song_warned:
                self._song_warned = True
                print(f"companion: ignored a song that is not JSON: {e}", file=sys.stderr)
            return
        self._last_now = None
        with self.cond:
            self._song_head = head
            self.results = {}
            self.song_ver += 1

    # --- server threads (call with self.cond held) ---

    def song_payload(self) -> str:
        if self._song_head is None:
            return "null"
        return f'{self._song_head},"results":{self.results_json()}}}'

    def results_json(self) -> str:
        return json.dumps(self.results, separators=(",", ":"))

    def broadcast(self) -> None:
        """One thread: frame the new state once per tick and wake every client.
        Songs are not in the shared frame: a client that skips frames would lose
        one, so each client sends the song itself when its version is behind."""
        period = 1.0 / STREAM_HZ
        state_ver = 0
        last = time.monotonic()
        while not self.stop.wait(period):
            now = time.monotonic()
            with self.cond:
                parts = []
                if self.state_ver != state_ver:
                    state_ver = self.state_ver
                    parts.append(_frame("state", self.state))
                if not parts and now - last >= KEEPALIVE_S:
                    parts.append(b": ping\n\n")  # finds dead clients
                if parts:
                    last = now
                    self.frame = b"".join(parts)
                    self.seq += 1
                    self.cond.notify_all()


class _Handler(BaseHTTPRequestHandler):
    server: _Server  # type: ignore[assignment]
    timeout = 5

    def log_message(self, format, *args) -> None:  # noqa: A002 - base class signature
        pass

    def do_GET(self) -> None:  # noqa: N802
        host = self.headers.get("Host")
        if host is not None and not self.server.host_allowed(host_of(host)):
            self.send_error(HTTPStatus.FORBIDDEN, "unknown Host header")
            return
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        hub = self.server.hub
        if path in PAGES:
            self._file(PAGES[path][0])
        elif path in ASSETS:
            self._file(ASSETS[path])
        elif path in ("/state", "/song", "/results"):
            with hub.cond:
                body = hub.state if path == "/state" else hub.song_payload() if path == "/song" else hub.results_json()
            self._send(body.encode(), "application/json")
        elif path == "/info":
            self._send(json.dumps(self.server.info()).encode(), "application/json")
        elif path == "/events":
            self._events()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _file(self, name: str) -> None:
        self._send(static_file(name), TYPES[Path(name).suffix])

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _events(self) -> None:
        hub = self.server.hub
        with hub.cond:
            full = hub.clients >= hub.max_clients
            if not full:
                hub.clients += 1
        if full:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "too many companion pages open")
            return
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            with hub.cond:
                seq, song_ver = hub.seq, hub.song_ver
                first = b"retry: 1000\n\n" + _frame("song", hub.song_payload())
                if hub.state_ver:
                    first += _frame("state", hub.state)
            self.wfile.write(first)
            self.wfile.flush()
            while True:
                with hub.cond:
                    hub.cond.wait_for(lambda s=seq: hub.seq != s or hub.stop.is_set(), timeout=1.0)
                    if hub.stop.is_set():
                        return
                    if hub.seq == seq:
                        continue
                    seq, data = hub.seq, hub.frame
                    if hub.song_ver != song_ver:  # this client has not sent the current song yet
                        song_ver = hub.song_ver
                        data = _frame("song", hub.song_payload()) + data
                self.wfile.write(data)
                self.wfile.flush()
        except OSError:
            pass  # the page went away (BrokenPipe, ConnectionReset, ConnectionAborted, timeout)
        finally:
            with hub.cond:
                hub.clients -= 1


class _Server(ThreadingHTTPServer):
    # Daemon client threads: an open page never keeps the process alive, so the
    # app's atexit handlers (FFB stop) still run. close() joins them explicitly.
    daemon_threads = True
    block_on_close = False
    # POSIX: SO_REUSEADDR skips TIME_WAIT but still refuses a live listener.
    # Windows: it would let two servers share the port; use SO_EXCLUSIVEADDRUSE.
    allow_reuse_address = sys.platform != "win32"
    hub: _Hub

    def __init__(self, addr, allowed_hosts: list[str]) -> None:
        super().__init__(addr, _Handler, bind_and_activate=False)
        self._conns: dict[threading.Thread, socket.socket] = {}
        self._conns_lock = threading.Lock()
        self._serve_lock = threading.Lock()
        self.serving = False
        self.stopping = False
        self._lan: list[str] | None = None
        self._names = {h.lower() for h in allowed_hosts}

    def server_bind(self) -> None:
        if sys.platform == "win32":
            self.socket.setsockopt(socket.SOL_SOCKET, getattr(socket, "SO_EXCLUSIVEADDRUSE", 0), 1)
        super().server_bind()

    def process_request(self, request, client_address) -> None:
        t = threading.Thread(target=self._client, args=(request, client_address), name="companion-client",
                             daemon=True)
        with self._conns_lock:
            full = len(self._conns) >= MAX_CONNECTIONS
            if not full:
                self._conns[t] = request
        if full:
            self.shutdown_request(request)
            return
        try:
            t.start()
        except RuntimeError:  # no thread left: the slot must not stay taken
            with self._conns_lock:
                self._conns.pop(t, None)
            self.shutdown_request(request)

    def _client(self, request, client_address) -> None:
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            with self._conns_lock:
                self._conns.pop(threading.current_thread(), None)
            self.shutdown_request(request)

    def handle_error(self, request, client_address) -> None:
        exc = sys.exc_info()[1]
        if not isinstance(exc, OSError):
            print(f"companion: request from {client_address[0]} failed: {exc!r}", file=sys.stderr)

    def close_clients(self, timeout: float) -> None:
        with self._conns_lock:
            conns = list(self._conns.items())
        for _, sock in conns:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        deadline = time.monotonic() + timeout
        for t, _ in conns:
            t.join(max(0.0, deadline - time.monotonic()))

    def serve(self) -> None:
        """Serve thread: slow name lookups run here, not on the game thread, and
        before serving, so no early request sees an empty list. A failed lookup
        never stops serve_forever() from starting."""
        try:
            self._names |= own_names()
        except Exception:  # e.g. UnicodeError from a host name that fails IDNA
            self._names.add("localhost")
        try:
            self._lan = lan_addresses()
        except Exception:
            self._lan = []
        with self._serve_lock:
            if self.stopping:
                return
            self.serving = True
        self.serve_forever(poll_interval=0.1)

    def stop_serving(self) -> None:
        """shutdown() waits for serve_forever(); skip it when that never started."""
        with self._serve_lock:
            self.stopping = True
            serving = self.serving
        if serving:
            self.shutdown()

    def host_allowed(self, name: str) -> bool:
        """DNS rebinding needs a name, so any IP literal is safe (second interface,
        hotspot, USB tether). Names must be this machine's or in allowed_hosts."""
        return is_ip(name) or name in self._names

    def info(self) -> dict:
        host, port = self.server_address[:2]
        hosts = (self._lan or ["localhost"]) if host in ("0.0.0.0", "") else [host]  # known so far
        return {
            "port": port,
            "pages": [{"path": p, "title": d} for p, (_, d) in PAGES.items() if p != "/"],
            "urls": [f"http://{h}:{port}" for h in hosts],
        }


class CompanionServer:
    """The live server. `start()` binds (CompanionError on failure) and serves
    from background threads; `close()` stops them all within about a second.
    `port_flag` names the option that changes the port, for the error text."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, *,
                 allowed_hosts: list[str] | None = None, max_clients: int = MAX_CLIENTS,
                 min_interval: float = 1.0 / STREAM_HZ, port_flag: str = "--companion PORT"):
        self.host, self.port = check_host(host), check_port(port)
        self.port_flag = port_flag
        self._allowed = list(allowed_hosts or [])
        self._hub = _Hub(min_interval, max_clients)
        self._server: _Server | None = None
        self._threads: list[threading.Thread] = []

    def start(self) -> CompanionServer:
        if self._server is not None:
            return self
        self._hub.stop.clear()  # restartable after close(); published song and state are kept
        server = _Server((self.host, self.port), self._allowed)
        try:
            server.server_bind()
            server.server_activate()
        except (OSError, OverflowError, ValueError, UnicodeError) as e:
            server.server_close()
            why = getattr(e, "strerror", None) or str(e)
            hint = f"; pick another port with {self.port_flag}" if getattr(e, "errno", None) == errno.EADDRINUSE else ""
            raise CompanionError(f"companion: cannot listen on {self.host}:{self.port} ({why}){hint}") from e
        server.hub = self._hub
        self._server = server
        self.port = server.server_address[1]
        self._threads = [
            threading.Thread(target=server.serve, name="companion", daemon=True),
            threading.Thread(target=self._hub.broadcast, name="companion-broadcast", daemon=True),
        ]
        for t in self._threads:
            t.start()
        return self

    @property
    def urls(self) -> list[str]:
        return self._server.info()["urls"] if self._server else []

    def start_line(self) -> str:
        """The start-up line. On all interfaces the LAN addresses are looked up in the
        background; until they are known, the line points to the / page, which lists them."""
        line = f"companion pages: {', '.join(self.urls)}  (/dash, /top)"
        srv = self._server
        if srv is not None and srv.server_address[0] in ("0.0.0.0", "") and srv._lan is None:
            line += f"; addresses for other devices: http://localhost:{self.port}/"
        return line

    def publish(self, snapshot: Snapshot | dict | str) -> None:
        self._hub.publish(snapshot)

    def publish_song(self, info, chart) -> None:
        self._hub.publish_song(info, chart)

    def close(self) -> None:
        server = self._server
        if server is None:
            return
        self._server = None
        with self._hub.cond:
            self._hub.stop.set()
            self._hub.cond.notify_all()
        server.stop_serving()
        server.server_close()
        server.close_clients(CLOSE_TIMEOUT_S)
        for t in self._threads:
            t.join(CLOSE_TIMEOUT_S)
        self._threads = []

    def __enter__(self) -> CompanionServer:
        return self if self._server else self.start()

    def __exit__(self, *exc) -> None:
        self.close()


class NullCompanion:
    """Stand-in when the companion is off or could not start: every call is a no-op."""

    urls: list[str] = []

    def publish(self, snapshot) -> None:
        pass

    def publish_song(self, info, chart) -> None:
        pass

    def close(self) -> None:
        pass


def start_companion(host: str | None = None, port: int | None = None, *,
                    log=print) -> CompanionServer | NullCompanion:
    """For the app: start with the saved settings unless overridden. A `host` given
    is saved to companion.json once the server listens; `port` is for this run. Any failure
    (bad settings file, bad host or port, port in use) logs one line and returns
    a NullCompanion, so the game continues."""
    try:
        s = CompanionSettings.load()
        if not s.enabled:
            log("companion pages are off (enabled: false in companion.json).")
            return NullCompanion()
        server = CompanionServer(host or s.host, s.port if port is None else port,
                                 allowed_hosts=s.allowed_hosts).start()
    except Exception as e:  # the game must continue whatever went wrong here
        msg = str(e) if isinstance(e, CompanionError) else f"companion: {e!r}"
        log(f"{msg}. Continuing without the companion pages.")
        return NullCompanion()
    if host and host != s.host:   # --companion-host is saved; --companion PORT is for this run only
        s.host = server.host
        try:
            s.save()
        except OSError as e:
            log(f"companion: cannot save {SETTINGS_FILE}: {e}")
    log(server.start_line())
    return server
