"""Security and resource-bound tests for public URL ingestion."""

from __future__ import annotations

import contextlib
import socket
import ssl
import time
from email.message import Message
from typing import Any

import pytest

import app.remote_sources as remote
from app.remote_sources import (
    RemoteSourceError,
    RemoteSourceTooLargeError,
    UnsafeRemoteSourceError,
    fetch_remote_source,
)


class _FakeResponse:
    def __init__(
        self,
        content: bytes = b"track-data",
        *,
        status: int = 200,
        headers: list[tuple[str, str]] | None = None,
    ) -> None:
        self.status = status
        self.headers = Message()
        for key, value in headers or []:
            self.headers[key] = value
        self._content = content
        self._offset = 0
        self.fp = None

    def getheader(self, name: str, default: Any = None) -> Any:
        return self.headers.get(name, default)

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._content) - self._offset
        result = self._content[self._offset : self._offset + size]
        self._offset += len(result)
        return result

    def close(self) -> None:
        return None


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _hostname, port, **_kwargs: [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("93.184.216.34", port),
            )
        ],
    )


def _responses(
    monkeypatch: pytest.MonkeyPatch,
    *responses: _FakeResponse,
) -> list[str]:
    pending = list(responses)
    requested: list[str] = []

    def request_once(
        target: remote._Target,
        *,
        deadline: float,
        context: Any,
    ) -> contextlib.AbstractContextManager[_FakeResponse]:
        del deadline, context
        requested.append(target.url)
        return contextlib.nullcontext(pending.pop(0))

    monkeypatch.setattr(remote, "_request_once", request_once)
    return requested


def test_fetch_returns_bounded_bytes_and_response_metadata(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,
) -> None:
    requested = _responses(
        monkeypatch,
        _FakeResponse(
            b"chr1\t0\t10\t1\n",
            headers=[
                ("Content-Type", "text/plain; charset=utf-8"),
                ("Content-Length", "12"),
                ("Content-Disposition", 'attachment; filename="ENCFF001.bedGraph"'),
                ("Accept-Ranges", "bytes"),
                ("ETag", '"abc"'),
                ("Last-Modified", "Sun, 23 Aug 2026 10:00:00 GMT"),
            ],
        ),
    )

    result = fetch_remote_source("https://www.encodeproject.org/download/no-extension")

    assert requested == ["https://www.encodeproject.org/download/no-extension"]
    assert result.filename == "ENCFF001.bedGraph"
    assert result.content == b"chr1\t0\t10\t1\n"
    assert result.size == 12
    assert result.content_type == "text/plain"
    assert result.accept_ranges is True
    assert result.etag == '"abc"'
    assert result.last_modified == "Sun, 23 Aug 2026 10:00:00 GMT"


@pytest.mark.parametrize("hostname", ["hgdownload.soe.ucsc.edu", "hgdownload.cse.ucsc.edu", "HGDOWNLOAD.SOE.UCSC.EDU."])
@pytest.mark.parametrize("port", ["", ":443"])
def test_old_ucsc_https_links_use_current_verified_host_without_changing_the_file(monkeypatch, public_dns, hostname, port):
    requested = _responses(monkeypatch, _FakeResponse(b"bigwig"))
    source_url = f"https://{hostname}{port}/goldenPath/hg19/a%2Fb.bigWig?sig=a%2Fb%2Bc"
    result = fetch_remote_source(source_url)
    expected = f"https://hgdownload.gi.ucsc.edu{port}/goldenPath/hg19/a%2Fb.bigWig?sig=a%2Fb%2Bc"
    assert requested == [expected]
    assert result.requested_url == source_url
    assert result.final_url == expected
    assert result.redirects == ()  # Host migration does not invent an HTTP redirect.
    assert result.content == b"bigwig"


@pytest.mark.parametrize("url", [
    "http://hgdownload.soe.ucsc.edu/a.bw",
    "https://hgdownload.soe.ucsc.edu.attacker.example/a.bw",
    "https://example.org/a.bw",
])
def test_ucsc_migration_does_not_rewrite_other_hosts_or_plain_http(monkeypatch, public_dns, url):
    requested = _responses(monkeypatch, _FakeResponse(b"bigwig"))
    assert fetch_remote_source(url).final_url == url
    assert requested == [url]


def test_migrated_ucsc_host_still_requires_public_dns_addresses(monkeypatch):
    resolved = []

    def private_dns(hostname, port, **kwargs):
        resolved.append(hostname)
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", port))]

    monkeypatch.setattr(socket, "getaddrinfo", private_dns)
    requested = _responses(monkeypatch, _FakeResponse(b"unreachable"))
    with pytest.raises(UnsafeRemoteSourceError, match="private"):
        fetch_remote_source("https://hgdownload.soe.ucsc.edu/a.bw")
    assert resolved == ["hgdownload.gi.ucsc.edu"]
    assert requested == []


@pytest.mark.parametrize("failure,message", [
    (ssl.SSLCertVerificationError(1, "hostname mismatch"), "certificate could not be verified"),
    (ssl.SSLError(1, "handshake failure"), "secure connection"),
    (TimeoutError("handshake timed out"), "did not respond in time"),
    (ConnectionResetError("connection reset"), "could not be reached"),
])
def test_certificate_timeout_and_network_failures_have_distinct_messages(monkeypatch, public_dns, failure, message):
    connections = []

    class FailedConnection:
        def __init__(self, *args):
            self.closed = False
            connections.append(self)

        def request(self, *args, **kwargs):
            raise failure

        def close(self):
            self.closed = True

    monkeypatch.setattr(remote, "_PinnedHTTPSConnection", FailedConnection)
    target = remote._validated_target("https://example.org/file.bw")
    with pytest.raises(RemoteSourceError, match=message) as raised:
        remote._request_once(target, deadline=time.monotonic() + 20, context=ssl.create_default_context())
    assert raised.value.__cause__ is failure
    assert all(connection.closed for connection in connections)


@pytest.mark.parametrize(
    "url, message",
    [
        ("file:///etc/passwd", "Only public HTTP"),
        ("ftp://example.org/data.bw", "Only public HTTP"),
        ("https://user:secret@example.org/data.bw", "usernames or passwords"),
        ("https://example.org/data.bw#section", "#fragment"),
        ("https://example.org:8443/data.bw", "standard HTTPS"),
        ("https://example.org/data.bw\nX-Test: injected", "valid public"),
    ],
)
def test_unsafe_url_shapes_are_rejected_before_network_access(
    url: str,
    message: str,
) -> None:
    with pytest.raises(UnsafeRemoteSourceError, match=message):
        fetch_remote_source(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/data.bw",
        "http://127.1/data.bw",
        "http://[::1]/data.bw",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.8/data.bw",
        "http://192.168.1.2/data.bw",
        "http://100.64.0.1/data.bw",
    ],
)
def test_private_local_and_non_global_ip_literals_are_rejected(url: str) -> None:
    with pytest.raises(UnsafeRemoteSourceError, match="Private|private"):
        fetch_remote_source(url)


def test_dns_answer_with_any_private_address_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 443)),
        ],
    )

    with pytest.raises(UnsafeRemoteSourceError, match="resolves to a private"):
        fetch_remote_source("https://mixed.example/data.bw")


def test_redirect_destination_is_revalidated_and_cannot_reach_localhost(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,
) -> None:
    requested = _responses(
        monkeypatch,
        _FakeResponse(status=302, headers=[("Location", "http://127.0.0.1/private")]),
    )

    with pytest.raises(UnsafeRemoteSourceError, match="Private"):
        fetch_remote_source("https://public.example/data.bw")
    assert requested == ["https://public.example/data.bw"]


def test_redirects_are_bounded_and_loop_checked(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,
) -> None:
    _responses(
        monkeypatch,
        _FakeResponse(status=302, headers=[("Location", "/again")]),
        _FakeResponse(status=302, headers=[("Location", "/third")]),
    )

    with pytest.raises(RemoteSourceError, match="redirected too many"):
        fetch_remote_source("https://example.org/start", max_redirects=1)


def test_redirect_preserves_signed_query_escapes(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,
) -> None:
    requested = _responses(
        monkeypatch,
        _FakeResponse(
            status=302,
            headers=[("Location", "https://cdn.example/data.bw?sig=a%2Fb%2Bc")],
        ),
        _FakeResponse(b"bigwig"),
    )

    result = fetch_remote_source("https://example.org/start")

    assert requested[-1].endswith("?sig=a%2Fb%2Bc")
    assert result.redirects == ("https://cdn.example/data.bw?sig=a%2Fb%2Bc",)
    assert result.filename == "data.bw"


def test_declared_size_is_rejected_before_reading(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,
) -> None:
    response = _FakeResponse(b"unread", headers=[("Content-Length", "11")])
    _responses(monkeypatch, response)

    with pytest.raises(RemoteSourceTooLargeError, match="limit"):
        fetch_remote_source("https://example.org/data.bw", max_bytes=10)
    assert response._offset == 0


def test_chunked_or_untrusted_length_response_is_bounded_while_reading(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,
) -> None:
    _responses(monkeypatch, _FakeResponse(b"12345678901"))

    with pytest.raises(RemoteSourceTooLargeError, match="limit"):
        fetch_remote_source("https://example.org/data.bw", max_bytes=10)


def test_conflicting_content_lengths_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,
) -> None:
    _responses(
        monkeypatch,
        _FakeResponse(
            b"1234",
            headers=[("Content-Length", "4"), ("Content-Length", "5")],
        ),
    )

    with pytest.raises(RemoteSourceError, match="conflicting Content-Length"):
        fetch_remote_source("https://example.org/data.bw")


def test_http_level_compression_is_rejected_but_gz_file_bytes_are_allowed(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,
) -> None:
    _responses(
        monkeypatch,
        _FakeResponse(b"compressed", headers=[("Content-Encoding", "gzip")]),
    )
    with pytest.raises(RemoteSourceError, match="HTTP compression"):
        fetch_remote_source("https://example.org/data.bedgraph")

    _responses(monkeypatch, _FakeResponse(b"\x1f\x8bfile-payload"))
    result = fetch_remote_source("https://example.org/data.bedgraph.gz")
    assert result.filename == "data.bedgraph.gz"


def test_response_filename_cannot_escape_a_future_temporary_directory(
    monkeypatch: pytest.MonkeyPatch,
    public_dns: None,
) -> None:
    _responses(
        monkeypatch,
        _FakeResponse(
            b"data",
            headers=[("Content-Disposition", 'attachment; filename="../../CON.bigWig"')],
        ),
    )

    result = fetch_remote_source("https://example.org/download")

    assert result.filename == "_CON.bigWig"
    assert "/" not in result.filename
    assert "\\" not in result.filename


def test_limits_cannot_be_disabled_with_extreme_arguments() -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        fetch_remote_source("https://example.org/data.bw", max_bytes=0)
    with pytest.raises(ValueError, match="max_bytes"):
        fetch_remote_source(
            "https://example.org/data.bw",
            max_bytes=remote.ABSOLUTE_MAX_BYTES + 1,
        )
    with pytest.raises(ValueError, match="timeout"):
        fetch_remote_source("https://example.org/data.bw", timeout=0)
    with pytest.raises(ValueError, match="max_redirects"):
        fetch_remote_source("https://example.org/data.bw", max_redirects=9)


def test_https_connection_pins_ip_but_uses_hostname_for_tls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_socket = object()
    wrapped_socket = object()
    calls: list[tuple[remote._Address, int, float]] = []

    def connected(address: remote._Address, port: int, timeout: float) -> object:
        calls.append((address, port, timeout))
        return raw_socket

    class _Context:
        def wrap_socket(self, sock: object, *, server_hostname: str) -> object:
            assert sock is raw_socket
            assert server_hostname == "files.example.org"
            return wrapped_socket

    monkeypatch.setattr(remote, "_connected_socket", connected)
    address = remote._Address(socket.AF_INET, "93.184.216.34")
    connection = remote._PinnedHTTPSConnection(
        "files.example.org",
        address,
        443,
        5.0,
        _Context(),  # type: ignore[arg-type]
    )

    connection.connect()

    assert connection.sock is wrapped_socket
    assert calls == [(address, 443, 5.0)]
