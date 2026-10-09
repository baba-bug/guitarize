import functools
import http.server
import threading
import urllib.error
import urllib.request

import pytest

from guitarize.cli import RangeRequestHandler


@pytest.fixture
def server(tmp_path):
    (tmp_path / "a.bin").write_bytes(bytes(range(100)))
    (tmp_path / "sub").mkdir()
    handler = functools.partial(RangeRequestHandler, directory=str(tmp_path))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_range_request_returns_206(server):
    url = f"{server}/a.bin"
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


def test_suffix_and_unsatisfiable_ranges(server):
    url = f"{server}/a.bin"
    req = urllib.request.Request(url, headers={"Range": "bytes=-5"})
    with urllib.request.urlopen(req) as r:
        assert r.status == 206 and r.read() == bytes(range(95, 100))
    req = urllib.request.Request(url, headers={"Range": "bytes=200-"})
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req)
    assert exc.value.code == 416 and exc.value.headers["Content-Range"] == "bytes */100"


def test_foreign_host_header_rejected(server):
    req = urllib.request.Request(f"{server}/a.bin", headers={"Host": "evil.example"})
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req)
    assert exc.value.code == 403
    req = urllib.request.Request(f"{server}/a.bin", headers={"Host": "localhost:8765"})
    with urllib.request.urlopen(req) as r:
        assert r.status == 200


def test_directory_listing_disabled(server):
    for path in ("/", "/sub/"):
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"{server}{path}")
        assert exc.value.code == 404
