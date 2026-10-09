import functools
import http.server
import threading
import urllib.error
import urllib.request

from guitarize.cli import RangeRequestHandler


def test_range_request_returns_206(tmp_path):
    (tmp_path / "a.bin").write_bytes(bytes(range(100)))
    handler = functools.partial(RangeRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/a.bin"
        req = urllib.request.Request(url, headers={"Range": "bytes=10-19"})
        with urllib.request.urlopen(req) as r:
            assert r.status == 206
            assert r.headers["Content-Range"] == "bytes 10-19/100"
            assert r.read() == bytes(range(10, 20))
        with urllib.request.urlopen(url) as r:
            assert r.status == 200 and len(r.read()) == 100
        req = urllib.request.Request(url, headers={"Range": "bytes=90-"})
        with urllib.request.urlopen(req) as r:
            assert r.read() == bytes(range(90, 100))
    finally:
        server.shutdown()


def test_suffix_and_unsatisfiable_ranges(tmp_path):
    (tmp_path / "a.bin").write_bytes(bytes(range(100)))
    handler = functools.partial(RangeRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/a.bin"
        req = urllib.request.Request(url, headers={"Range": "bytes=-5"})
        with urllib.request.urlopen(req) as r:
            assert r.status == 206 and r.read() == bytes(range(95, 100))
        req = urllib.request.Request(url, headers={"Range": "bytes=200-"})
        try:
            urllib.request.urlopen(req)
            raise AssertionError("expected HTTP 416")
        except urllib.error.HTTPError as e:
            assert e.code == 416 and e.headers["Content-Range"] == "bytes */100"
    finally:
        server.shutdown()
