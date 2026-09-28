import json
import socket
import struct
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from torquehero.chart import Chart
from torquehero.companion import CompanionError, CompanionServer, CompanionSettings, NullCompanion, start_companion
from torquehero.companion import replay as rp
from torquehero.companion import server as srv
from torquehero.companion.server import host_of, static_file
from torquehero.config import Config
from torquehero.game import Game
from torquehero.state import NoteView, Snapshot

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def config_dir(tmp_path, monkeypatch) -> Path:
    """Never read the real user config dir."""
    d = tmp_path / "config"
    monkeypatch.setenv("TORQUEHERO_CONFIG_DIR", str(d))
    return d


def get(server, path: str, timeout: float = 3.0, headers: dict | None = None) -> tuple[int, str, str]:
    req = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.headers["Content-Type"], r.read().decode()


def read_events(server, n: int, timeout: float = 3.0) -> list[tuple[str, str]]:
    """The first `n` named events from /events, as (event, data)."""
    out: list[tuple[str, str]] = []
    with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/events", timeout=timeout) as r:
        assert r.headers["Content-Type"] == "text/event-stream"
        event = None
        while len(out) < n:
            line = r.readline().decode().rstrip("\n")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event:
                out.append((event, line[6:]))
                event = None
    return out


def wait_for(cond, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def client_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "companion-client" and t.is_alive()]


def in_socket_send() -> bool:
    """A companion-client thread is blocked writing to its socket."""
    names = {t.ident: t.name for t in threading.enumerate()}
    for ident, frame in sys._current_frames().items():
        if names.get(ident) != "companion-client":
            continue
        f = frame
        while f:
            code = f.f_code
            if code.co_name in ("write", "sendall") and code.co_filename.endswith(("socket.py", "socketserver.py")):
                return True
            f = f.f_back
    return False


def stalled_client(port: int) -> socket.socket:
    """Open /events with a tiny receive buffer and never read."""
    c = socket.socket()
    c.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    c.connect(("127.0.0.1", port))
    c.sendall(b"GET /events HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
    return c


def stall(server, timeout: float = 5.0) -> None:
    """Publish large snapshots until a client thread blocks in send."""
    big = "x" * 200_000
    deadline = time.monotonic() + timeout
    i = 0
    while not in_socket_send():
        assert time.monotonic() < deadline, "client thread never blocked in send"
        i += 1
        server.publish({"phase": "play", "pad": big, "score": i})
        time.sleep(0.03)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    s = CompanionServer("127.0.0.1", 0, min_interval=0.0).start()
    yield s
    s.close()


@pytest.fixture
def snapshot_dict() -> dict:
    return json.loads((FIXTURES / "snapshot.json").read_text())


# --- routes ---

def test_state_song_results_before_publish(server):
    assert get(server, "/state")[1:] == ("application/json", "null")
    assert json.loads(get(server, "/song")[2]) is None
    assert json.loads(get(server, "/results")[2]) == {}


def test_state_returns_latest_snapshot(server, snapshot_dict):
    server.publish(Snapshot(score=1))
    server.publish(Snapshot(score=2, now=float("nan")))
    d = json.loads(get(server, "/state")[2])
    assert d["score"] == 2 and d["now"] is None
    server.publish(snapshot_dict)
    assert json.loads(get(server, "/state")[2]) == snapshot_dict


def test_dict_and_string_snapshots_are_sanitised(server):
    server.publish({"now": float("nan"), "score": float("inf"), "phase": "play"})
    assert json.loads(get(server, "/state")[2]) == {"now": None, "score": None, "phase": "play"}
    server.publish('{"now": NaN,\n "text": "a\\nb"}')
    state = get(server, "/state")[2]
    assert "\n" not in state and json.loads(state) == {"now": None, "text": "a\nb"}
    server.publish("not json")  # dropped with one warning, stream keeps going
    server.publish({"score": 5})
    assert [e for e, _ in read_events(server, 2)] == ["song", "state"]
    assert json.loads(get(server, "/state")[2]) == {"score": 5}


def test_pages_are_embedded_and_self_contained(server):
    pages = (("/", "companion"), ("/dash", "TURN BACK"), ("/top", "Timeline"), ("/common.js", "EventSource"))
    for path, marker in pages:
        status, _, body = get(server, path)
        assert status == 200 and marker in body
        assert "http://" not in body.replace("http://HOST", "") and "https://" not in body, path
    assert get(server, "/common.css")[1].startswith("text/css")
    assert static_file("dash.html").startswith(b"<!doctype html>")


def test_info_lists_pages_and_urls(server):
    info = json.loads(get(server, "/info")[2])
    assert [p["path"] for p in info["pages"]] == ["/dash", "/top"]
    assert info["urls"] == [f"http://127.0.0.1:{server.port}"]


def test_unknown_path_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server, "/nope")
    assert e.value.code == 404


def test_foreign_host_header_is_403(server):
    assert get(server, "/state", headers={"Host": f"localhost:{server.port}"})[0] == 200
    assert get(server, "/state", headers={"Host": socket.gethostname()})[0] == 200
    for ip in ("10.9.8.7:8765", "172.20.10.2", "[fe80::1]:8765"):  # second interface, hotspot, tether
        assert get(server, "/state", headers={"Host": ip})[0] == 200, ip
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server, "/state", headers={"Host": "evil.example:8765"})
    assert e.value.code == 403
    assert host_of("[::1]:8765") == "::1" and host_of("Rig.local:80") == "rig.local"


def test_allowed_hosts_setting():
    with CompanionServer("127.0.0.1", 0, allowed_hosts=["rig.lan"]) as s:
        assert get(s, "/state", headers={"Host": "rig.lan:1"})[0] == 200


# --- event stream ---

def test_events_send_song_first_then_state(server, fixture_chart):
    game = Game(fixture_chart, Config())
    server.publish(game.snapshot())
    server.publish_song(game.song_info(), fixture_chart)
    events = read_events(server, 2)
    assert [e for e, _ in events] == ["song", "state"]
    song = json.loads(events[0][1])
    assert song["info"]["title"] == fixture_chart.title
    assert len(song["chart"]["notes"]) == len(fixture_chart.notes)
    assert song["results"] == {}
    assert json.loads(events[1][1])["length"] == fixture_chart.length
    assert json.loads(get(server, "/song")[2]) == song


def test_song_is_null_on_connect_without_a_song(server):
    server.publish(Snapshot())
    assert read_events(server, 1) == [("song", "null")]


def test_events_stream_new_snapshots(server):
    server.publish(Snapshot(score=1))
    got = []
    done = threading.Event()

    def reader():
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/events", timeout=3) as r:
            while len(got) < 3:
                line = r.readline().decode()
                if line.startswith("data: ") and line != "data: null\n":
                    got.append(json.loads(line[6:])["score"])
        done.set()

    t = threading.Thread(target=reader)
    t.start()
    for i in range(2, 60):
        if done.is_set():
            break
        server.publish(Snapshot(score=i))
        time.sleep(0.03)
    t.join(3)
    assert len(got) == 3 and got == sorted(got) and got[0] >= 1


def test_results_accumulate_and_reset_per_song(server, fixture_chart):
    game = Game(fixture_chart, Config())
    server.publish_song(game.song_info(), fixture_chart)
    server.publish(Snapshot(notes=[NoteView(0, True, "perfect"), NoteView(1, False)]))
    server.publish(Snapshot(notes=[NoteView(1, True, "miss"), NoteView(8, True, None)]))  # 8: listen gate
    server.publish(Snapshot(notes=[]))  # window slid past: results stay
    expected = {"0": "perfect", "1": "miss", "8": "done"}
    assert json.loads(get(server, "/results")[2]) == expected
    assert json.loads(read_events(server, 1)[0][1])["results"] == expected  # late join gets them all
    server.publish_song(game.song_info(), fixture_chart)
    assert json.loads(get(server, "/results")[2]) == {}


def test_results_reset_when_song_clock_goes_back(server):
    server.publish(Snapshot(now=10.0, notes=[NoteView(3, True, "good")]))
    server.publish(Snapshot(now=10.5, notes=[]))
    assert json.loads(get(server, "/results")[2]) == {"3": "good"}
    server.publish(Snapshot(now=9.8, notes=[]))  # less than 1 s back: jitter, keep
    assert json.loads(get(server, "/results")[2]) == {"3": "good"}
    server.publish(Snapshot(now=0.2, notes=[NoteView(0, True, "miss")]))  # replay loop / restart
    assert json.loads(get(server, "/results")[2]) == {"0": "miss"}


def test_slow_client_still_gets_a_new_song(server):
    server.publish_song({"title": "A"}, {"notes": []})
    c = stalled_client(server.port)
    try:
        stall(server)
        server.publish_song({"title": "B"}, {"notes": []})
        for i in range(10):  # later ticks overwrite the shared frame
            server.publish({"phase": "play", "score": -i})
            time.sleep(0.06)
        c.settimeout(5)
        buf, titles = b"", []
        deadline = time.monotonic() + 5
        while "B" not in titles and time.monotonic() < deadline:
            buf += c.recv(1 << 20)
            *lines, buf = buf.split(b"\n\n")
            titles += [json.loads(ln.split(b"data: ", 1)[1])["info"]["title"]
                       for ln in lines if ln.startswith(b"event: song") or b"\nevent: song" in ln]
        assert titles[0] == "A" and titles[-1] == "B"
    finally:
        c.close()


def test_client_limit_answers_503():
    with CompanionServer("127.0.0.1", 0, max_clients=1) as s:
        s.publish(Snapshot())
        with urllib.request.urlopen(f"http://127.0.0.1:{s.port}/events", timeout=3) as r:
            r.readline()
            with pytest.raises(urllib.error.HTTPError) as e:
                get(s, "/events")
            assert e.value.code == 503
        wait_for(lambda: s._hub.clients == 0)
        assert read_events(s, 1)[0][0] == "song"


# --- publish side ---

def test_publish_serializes_at_most_stream_rate():
    calls = []

    class Counting(Snapshot):
        def to_json(self):
            calls.append(1)
            return super().to_json()

    with CompanionServer("127.0.0.1", 0) as s:
        start = time.perf_counter()
        for _ in range(5000):
            s.publish(Counting(phase="play"))
        assert time.perf_counter() - start < 0.5
        assert len(calls) <= 3
        s.publish(Counting(phase="paused"))  # a phase change is never dropped
        assert json.loads(get(s, "/state")[2])["phase"] == "paused"


def test_bad_song_is_logged_once_and_ignored(server, capfd):
    server.publish_song({"title": "ok"}, {"notes": []})
    server.publish_song(object(), {"notes": []})
    server.publish_song({"x": object()}, {})
    assert json.loads(get(server, "/song")[2])["info"]["title"] == "ok"
    err = capfd.readouterr().err
    assert err.count("ignored a song") == 1


def test_mutation_after_publish_does_not_reach_clients(server):
    snap = Snapshot(score=1)
    server.publish(snap)
    snap.score = 99
    snap.ffb["torque"] = float("nan")
    assert json.loads(get(server, "/state")[2])["score"] == 1


# --- start, settings, errors ---

def test_port_in_use_is_a_clear_error():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        with pytest.raises(CompanionError, match=f"cannot listen on 127.0.0.1:{port}.*--companion PORT"):
            CompanionServer("127.0.0.1", port).start()
        with pytest.raises(CompanionError, match="--port"):
            CompanionServer("127.0.0.1", port, port_flag="--port").start()
        logs = []
        c = start_companion("127.0.0.1", port, log=logs.append)
        assert isinstance(c, NullCompanion) and len(logs) == 1 and "Continuing without" in logs[0]
        c.publish(Snapshot())
        c.publish_song({}, {})
        c.close()


@pytest.mark.parametrize("content, host, port, needle", [
    ("{not json", None, None, "cannot read"),
    ("[1, 2]", None, None, "JSON object"),
    ('{"port": 70000}', None, None, "0 to 65535"),
    ('{"port": "8765"}', None, None, "0 to 65535"),
    ('{"enabled": "yes"}', None, None, "true or false"),
    ('{"host": 5}', None, None, "non-empty string"),
    (None, "no-such-host.invalid", 0, "cannot listen on no-such-host.invalid"),
    (None, "127.0.0.1", 70000, "0 to 65535"),
    (None, "127.0.0.1", "abc", "0 to 65535"),
])
def test_bad_input_logs_one_line_and_continues(config_dir, content, host, port, needle):
    if content is not None:
        config_dir.mkdir(parents=True)
        (config_dir / "companion.json").write_text(content)
    logs = []
    c = start_companion(host, port, log=logs.append)
    assert isinstance(c, NullCompanion)
    assert len(logs) == 1 and needle in logs[0] and "Continuing without" in logs[0]
    assert "\n" not in logs[0]


def test_disabled_setting_binds_nothing(config_dir):
    port = free_port()
    CompanionSettings(host="127.0.0.1", port=port, enabled=False).save()
    logs = []
    assert isinstance(start_companion(log=logs.append), NullCompanion)
    assert "off" in logs[0]
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))  # still free


def test_settings_defaults_and_roundtrip(config_dir):
    s = CompanionSettings.load()
    assert (s.host, s.port, s.enabled) == ("0.0.0.0", 8765, True)
    CompanionSettings(host="127.0.0.1", port=9001).save()
    assert CompanionSettings.load().port == 9001
    assert json.loads((config_dir / "companion.json").read_text())["host"] == "127.0.0.1"


# --- shutdown ---

def test_close_leaves_no_thread_running():
    before = set(threading.enumerate())
    s = CompanionServer("127.0.0.1", 0).start()
    s.publish(Snapshot())
    r = urllib.request.urlopen(f"http://127.0.0.1:{s.port}/events", timeout=3)  # client keeps a stream open
    r.readline()
    get(s, "/state")
    t0 = time.monotonic()
    s.close()
    assert time.monotonic() - t0 < 1.0
    r.close()
    assert set(threading.enumerate()) - before == set()
    s.close()  # idempotent


def test_close_is_fast_with_stalled_and_half_open_clients(monkeypatch):
    """close() must wake a thread blocked in send by shutting its socket down.
    With a 30 s socket timeout and a 5 s join, anything else takes 5 s."""
    monkeypatch.setattr(srv._Handler, "timeout", 30)
    monkeypatch.setattr(srv, "CLOSE_TIMEOUT_S", 5)
    s = CompanionServer("127.0.0.1", 0, min_interval=0.0).start()
    stalled = stalled_client(s.port)
    half_open = socket.create_connection(("127.0.0.1", s.port))  # never sends a request
    try:
        stall(s)
        t0 = time.monotonic()
        s.close()
        assert time.monotonic() - t0 < 1.0
        assert client_threads() == []  # before either client socket is closed
    finally:
        stalled.close()
        half_open.close()


def test_silent_disconnect_logs_nothing(server, capfd):
    server.publish(Snapshot())
    c = socket.create_connection(("127.0.0.1", server.port))
    c.sendall(b"GET /events HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
    c.recv(100)
    wait_for(lambda: server._hub.clients == 1)
    c.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    c.close()  # RST, no FIN
    deadline = time.monotonic() + 5
    i = 0
    while server._hub.clients:
        assert time.monotonic() < deadline
        i += 1
        server.publish(Snapshot(score=i))
        time.sleep(0.03)
    assert capfd.readouterr().err == ""


def test_start_close_start_on_one_object():
    s = CompanionServer("127.0.0.1", 0, min_interval=0.0)
    for score in (1, 2):
        s.start()
        s.publish(Snapshot(score=score))
        assert json.loads(get(s, "/state")[2])["score"] == score
        assert [e for e, _ in read_events(s, 2)] == ["song", "state"]
        s.close()


def test_open_stream_does_not_block_process_exit(tmp_path):
    """No close(): daemon client threads must not keep the process alive, and
    atexit handlers (the app's FFB stop) must still run."""
    script = tmp_path / "exit.py"
    script.write_text(textwrap.dedent("""
        import atexit, sys, threading, urllib.request
        from torquehero.companion import CompanionServer
        from torquehero.state import Snapshot
        atexit.register(lambda: print("atexit ran", flush=True))
        s = CompanionServer("127.0.0.1", 0).start()
        s.publish(Snapshot())
        opened = threading.Event()
        def page():
            r = urllib.request.urlopen(f"http://127.0.0.1:{s.port}/events", timeout=30)
            r.readline()
            opened.set()
            r.read()
        threading.Thread(target=page, daemon=True).start()
        opened.wait(5)
        if sys.argv[1] == "raise":
            raise RuntimeError("app crashed")
        sys.exit(0)
    """))
    for how, code in (("exit", 0), ("raise", 1)):
        r = subprocess.run([sys.executable, str(script), how], capture_output=True, text=True, timeout=10)
        assert r.returncode == code, r.stderr
        assert "atexit ran" in r.stdout


def test_failed_name_lookup_still_serves_and_closes(monkeypatch):
    def boom():
        raise UnicodeError("label empty or too long")

    monkeypatch.setattr(srv, "own_names", boom)
    monkeypatch.setattr(srv, "lan_addresses", boom)
    s = CompanionServer("0.0.0.0", 0).start()
    try:
        assert get(s, "/state")[0] == 200
        assert get(s, "/state", headers={"Host": "localhost"})[0] == 200
        assert s.urls == [f"http://localhost:{s.port}"]
    finally:
        t0 = time.monotonic()
        s.close()
        assert time.monotonic() - t0 < 1.0


def test_close_before_serving_started(monkeypatch):
    """A slow resolver on the serve thread: start() returns at once, close() does not wait for it."""
    release = threading.Event()
    monkeypatch.setattr(srv, "own_names", lambda: (release.wait(5), {"localhost"})[1])
    t0 = time.monotonic()
    s = CompanionServer("127.0.0.1", 0).start()
    assert time.monotonic() - t0 < 0.5
    s.close()
    assert time.monotonic() - t0 < 1.0
    release.set()
    wait_for(lambda: not any(t.name == "companion" for t in threading.enumerate()))


def test_connection_cap_closes_new_connections(monkeypatch):
    monkeypatch.setattr(srv, "MAX_CONNECTIONS", 4)
    with CompanionServer("127.0.0.1", 0) as s:
        held = [socket.create_connection(("127.0.0.1", s.port)) for _ in range(4)]  # idle, never send
        try:
            wait_for(lambda: len(client_threads()) == 4)
            extra = socket.create_connection(("127.0.0.1", s.port))
            extra.settimeout(2)
            try:
                assert extra.recv(10) == b""  # closed at once, before any request
            except ConnectionResetError:
                pass
            extra.close()
        finally:
            for c in held:
                c.close()
        wait_for(lambda: client_threads() == [])
        assert get(s, "/state")[0] == 200


def test_restart_on_same_port():
    s = CompanionServer("127.0.0.1", 0).start()
    port = s.port
    s.publish(Snapshot())
    read_events(s, 1)
    s.close()
    with CompanionServer("127.0.0.1", port) as s2:
        assert s2.port == port and get(s2, "/state")[2] == "null"


# --- replay ---

def test_package_sample_is_a_valid_chart_and_snapshot():
    sample = rp.default_fixture()
    assert sample.parent == Path(rp.__file__).parent / "sample"
    Chart.load(sample.with_name("chart.json"))
    snap = json.loads(sample.read_text())
    assert set(Snapshot.__dataclass_fields__) <= set(snap)


def test_package_sample_matches_fixtures():
    """The sample is a copy of tests/fixtures so an installed wheel can replay.
    Other tests do not depend on it; this one only reports drift."""
    sample = rp.default_fixture().parent
    for name in ("snapshot.json", "chart.json"):
        assert json.loads((sample / name).read_text()) == json.loads((FIXTURES / name).read_text()), (
            f"{name} drifted: cp tests/fixtures/{{chart,snapshot}}.json src/torquehero/companion/sample/")


def test_replay_animates_the_fixture():
    r = rp.load(rp.default_fixture())
    info, chart = r.song()
    length = info["length"]
    assert info["title"] and chart["notes"]
    frames = [r.frame(t / 30) for t in range(int((length + 1) * 30))]  # past the loop point
    nows = [f["now"] for f in frames]
    assert nows[0] == 0.0 and max(nows) > length - 1 and nows[-1] < 2
    assert all(f["phase"] == "play" for f in frames)
    assert max(f["combo"] for f in frames) > 3
    assert any(not f["layers"]["melody"]["alive"] or f["counts"]["miss"] for f in frames)
    assert any(f["fader_targets"][0] is not None for f in frames)
    spins = [n for n in chart["notes"] if n["kind"] == "spin"]
    offsets = {f["steer_offset_deg"] for f in frames}
    assert 0.0 in offsets and (len(offsets) > 1) == bool(spins)  # the scripted player completes its spins
    # Game.steer_unwind is also true mid-spin (wheel over half a turn from the offset); never outside one.
    in_spin = [any(n["t"] <= f["now"] <= n["t"] + n["dur"] for n in spins) for f in frames]
    assert all(ok for f, ok in zip(frames, in_spin, strict=True) if f["steer_unwind"])
    for f in frames[::50]:
        json.dumps(f, allow_nan=False)


def test_replay_without_chart_animates_clock(tmp_path, snapshot_dict):
    p = tmp_path / "snap.json"
    p.write_text(json.dumps(snapshot_dict))
    r = rp.load(p)
    assert r.song() is None
    a, b = r.frame(1.0), r.frame(2.5)
    assert (a["now"], b["now"]) == (1.0, 2.5) and a["score"] == snapshot_dict["score"]


def test_cli_replay_serves_pages(capsys):
    from torquehero.__main__ import main

    result = {}
    t = threading.Thread(target=lambda: result.setdefault("code", main(
        ["companion", "--replay", "--host", "127.0.0.1", "--port", "0", "--seconds", "1.5"])))
    t.start()
    deadline = time.monotonic() + 3
    out = ""
    while "companion pages" not in out and time.monotonic() < deadline:
        time.sleep(0.05)
        out += capsys.readouterr().out
    port = int(out.split("http://127.0.0.1:")[1].split()[0])
    events = read_events(SimpleNamespace(port=port), 2)
    assert [e for e, _ in events] == ["song", "state"]
    assert json.loads(events[0][1])["info"]["title"]
    assert json.loads(events[1][1])["phase"] == "play"
    t.join(5)
    assert result["code"] == 0


def test_cli_bad_settings_and_port_in_use(config_dir, capsys):
    from torquehero.__main__ import main

    config_dir.mkdir(parents=True)
    (config_dir / "companion.json").write_text("{broken")
    assert main(["companion", "--seconds", "0"]) == 1
    assert "cannot read" in capsys.readouterr().err
    (config_dir / "companion.json").unlink()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        assert main(["companion", "--host", "127.0.0.1", "--port", str(s.getsockname()[1]), "--seconds", "0"]) == 1
    assert "--port" in capsys.readouterr().err


def test_host_given_to_the_app_is_saved_but_not_the_port(config_dir):
    config_dir.mkdir(parents=True)
    port, run_port = free_port(), free_port()
    CompanionSettings(port=port).save()
    logs = []
    start_companion("127.0.0.1", run_port, log=logs.append).close()
    saved = json.loads((config_dir / "companion.json").read_text())
    assert saved["host"] == "127.0.0.1" and saved["port"] == port
    c = start_companion(log=logs.append)                     # no flags: the saved values
    try:
        assert c.host == "127.0.0.1" and c.port == port
    finally:
        c.close()


def test_start_line_points_to_the_index_until_lan_addresses_are_known():
    with CompanionServer("127.0.0.1", 0) as s:
        assert s.start_line() == f"companion pages: http://127.0.0.1:{s.port}  (/dash, /top)"
        s._server.server_address = ("0.0.0.0", s.port)
        s._server._lan = None
        assert s.start_line().endswith(f"addresses for other devices: http://localhost:{s.port}/")
        s._server._lan = ["192.168.1.5"]
        assert s.start_line() == f"companion pages: http://192.168.1.5:{s.port}  (/dash, /top)"


def test_a_client_thread_that_cannot_start_frees_its_slot(monkeypatch):
    class NoThread(threading.Thread):
        def start(self):
            raise RuntimeError("can't start new thread")

    with CompanionServer("127.0.0.1", 0) as s:
        monkeypatch.setattr(srv.threading, "Thread", NoThread)
        a, b = socket.socketpair()
        try:
            s._server.process_request(a, ("127.0.0.1", 0))
            assert s._server._conns == {}
        finally:
            b.close()
