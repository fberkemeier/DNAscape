"""Small, bounded HTTP(S) downloader for public genomic-track URLs.

The Data Manager accepts links supplied by the person running the local app.
Fetching those links with a general-purpose URL opener would allow redirects or
DNS changes to reach services on the user's own machine or private network.
This module therefore validates every address, disables implicit redirects and
connects to the validated IP address while preserving the original hostname for
the HTTP ``Host`` header and TLS certificate verification.

The function is synchronous by design.  NiceGUI callers should run it in an
``io_bound`` worker so DNS and network I/O never block the UI event loop.
"""

from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import ssl
import time
from dataclasses import dataclass
from email.message import Message
from pathlib import PurePosixPath
from types import TracebackType
from typing import Any, NamedTuple
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
ABSOLUTE_MAX_BYTES = 128 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_REDIRECTS = 4

_ALLOWED_PORTS = {"http": 80, "https": 443}
# UCSC's current download hostname has a valid HTTPS certificate; its older
# aliases can fail before the server can redirect. Keep the path/query intact
# and validate the canonical host through the same DNS and TLS checks.
_LEGACY_UCSC_HTTPS_HOSTS = {
    "hgdownload.soe.ucsc.edu": "hgdownload.gi.ucsc.edu",
    "hgdownload.cse.ucsc.edu": "hgdownload.gi.ucsc.edu",
}
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_READ_CHUNK_BYTES = 256 * 1024
_MAX_CONNECT_ATTEMPT_SECONDS = 5.0
_MAX_TIMEOUT_SECONDS = 120.0
_MAX_REDIRECT_LIMIT = 8
_MAX_FILENAME_LENGTH = 180
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
_UNSAFE_FILENAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._() +\-]+")
_AMBIGUOUS_NUMERIC_HOST = re.compile(r"^(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+))*$", re.I)
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


class RemoteSourceError(ValueError):
    """A remote source could not be fetched or was not a usable response."""


class UnsafeRemoteSourceError(RemoteSourceError):
    """A remote URL would cross the downloader's network-safety boundary."""


class RemoteSourceTooLargeError(RemoteSourceError):
    """A declared or streamed response exceeded the configured byte limit."""


@dataclass(frozen=True, slots=True)
class RemoteSource:
    """One fully downloaded, bounded public HTTP(S) resource."""

    requested_url: str
    final_url: str
    filename: str
    content: bytes
    content_type: str | None
    etag: str | None
    last_modified: str | None
    accept_ranges: bool
    redirects: tuple[str, ...]

    @property
    def size(self) -> int:
        """Number of downloaded bytes."""

        return len(self.content)


class _Address(NamedTuple):
    family: socket.AddressFamily
    value: str


@dataclass(frozen=True, slots=True)
class _Target:
    url: str
    scheme: str
    hostname: str
    port: int
    request_target: str
    addresses: tuple[_Address, ...]


def _is_public_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    # ``is_global`` excludes loopback, link-local, private, shared, reserved,
    # multicast and unspecified ranges for both IPv4 and IPv6.
    return bool(address.is_global)


def _resolve_public_addresses(hostname: str, port: int) -> tuple[_Address, ...]:
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None

    if literal is not None:
        if not literal.is_global:
            raise UnsafeRemoteSourceError(
                "Private, local, reserved, and non-routable network addresses are not allowed."
            )
        family = socket.AF_INET6 if literal.version == 6 else socket.AF_INET
        return (_Address(family, literal.compressed),)

    try:
        answers = socket.getaddrinfo(
            hostname,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except OSError as exc:
        raise RemoteSourceError("The remote hostname could not be resolved.") from exc

    addresses: list[_Address] = []
    seen: set[tuple[int, str]] = set()
    for family, _type, _protocol, _canonical, sockaddr in answers:
        if family not in {socket.AF_INET, socket.AF_INET6}:
            continue
        value = str(sockaddr[0]).split("%", 1)[0]
        key = (int(family), value)
        if key not in seen:
            seen.add(key)
            addresses.append(_Address(family, value))

    if not addresses:
        raise RemoteSourceError("The remote hostname has no usable IPv4 or IPv6 address.")
    if any(not _is_public_address(address.value) for address in addresses):
        raise UnsafeRemoteSourceError(
            "The remote hostname resolves to a private, local, or non-routable address."
        )
    return tuple(addresses)


def _quoted_request_target(path: str, query: str) -> str:
    # Keep existing escapes verbatim.  Decoding and re-encoding ``%2F`` or a
    # signed query would change the origin resource (and can invalidate cloud
    # storage signatures).
    safe_path = quote(path or "/", safe="/%:@!$&'()*+,;=-._~")
    # Preserve URI query delimiters and existing percent escapes, but encode
    # unicode and spaces before handing the origin-form target to http.client.
    safe_query = quote(query, safe="!$&'()*+,;=:/?@%+-._~")
    return safe_path + (f"?{safe_query}" if safe_query else "")


def _validated_target(url: str) -> _Target:
    candidate = str(url).strip()
    if not candidate or "\\" in candidate or _CONTROL_CHARACTERS.search(candidate):
        raise UnsafeRemoteSourceError("Enter a valid public HTTP or HTTPS URL.")
    try:
        parts = urlsplit(candidate)
    except ValueError as exc:
        raise UnsafeRemoteSourceError("Enter a valid public HTTP or HTTPS URL.") from exc

    scheme = parts.scheme.casefold()
    if scheme not in _ALLOWED_PORTS:
        raise UnsafeRemoteSourceError("Only public HTTP and HTTPS links are supported.")
    if parts.username is not None or parts.password is not None:
        raise UnsafeRemoteSourceError("Links containing usernames or passwords are not supported.")
    if parts.fragment:
        raise UnsafeRemoteSourceError("Remove the #fragment from the remote file link.")
    if not parts.hostname:
        raise UnsafeRemoteSourceError("The remote link must include a hostname.")

    try:
        hostname = parts.hostname.rstrip(".").encode("idna").decode("ascii").casefold()
    except (UnicodeError, ValueError) as exc:
        raise UnsafeRemoteSourceError("The remote link contains an invalid hostname.") from exc
    if not hostname or len(hostname) > 253 or "%" in hostname:
        raise UnsafeRemoteSourceError("The remote link contains an invalid hostname.")
    if _AMBIGUOUS_NUMERIC_HOST.fullmatch(hostname):
        # Some resolvers accept shortened, octal or hexadecimal IPv4 forms
        # (for example ``127.1``).  Reject them instead of relying on
        # platform-specific normalization before the private-address check.
        raise UnsafeRemoteSourceError(
            "Private, local, reserved, and non-routable network addresses are not allowed."
        )

    try:
        port = parts.port or _ALLOWED_PORTS[scheme]
    except ValueError as exc:
        raise UnsafeRemoteSourceError("The remote link contains an invalid port.") from exc
    if port != _ALLOWED_PORTS[scheme]:
        raise UnsafeRemoteSourceError(
            "Remote files must use standard HTTPS (443) or HTTP (80) ports."
        )

    canonical_hostname = (
        _LEGACY_UCSC_HTTPS_HOSTS.get(hostname, hostname) if scheme == "https" else hostname
    )
    netloc = parts.netloc
    if canonical_hostname != hostname:
        hostname = canonical_hostname
        netloc = hostname + (f":{port}" if parts.port is not None else "")
    addresses = _resolve_public_addresses(hostname, port)
    normalized_url = urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))
    return _Target(
        url=normalized_url,
        scheme=scheme,
        hostname=hostname,
        port=port,
        request_target=_quoted_request_target(parts.path, parts.query),
        addresses=addresses,
    )


def _connected_socket(address: _Address, port: int, timeout: float) -> socket.socket:
    sock = socket.socket(address.family, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        if address.family == socket.AF_INET6:
            sock.connect((address.value, port, 0, 0))
        else:
            sock.connect((address.value, port))
        return sock
    except BaseException:
        sock.close()
        raise


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection whose TCP peer is an already validated address."""

    def __init__(
        self,
        hostname: str,
        address: _Address,
        port: int,
        timeout: float,
    ) -> None:
        super().__init__(hostname, port=port, timeout=timeout)
        self._remote_address = address

    def connect(self) -> None:
        timeout = (
            self.timeout if isinstance(self.timeout, (int, float)) else DEFAULT_TIMEOUT_SECONDS
        )
        self.sock = _connected_socket(self._remote_address, self.port, float(timeout))


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection pinned to an IP but verified for the URL hostname."""

    def __init__(
        self,
        hostname: str,
        address: _Address,
        port: int,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(hostname, port=port, timeout=timeout, context=context)
        self._remote_address = address
        self._ssl_context = context

    def connect(self) -> None:
        raw_socket = _connected_socket(
            self._remote_address,
            self.port,
            float(
                self.timeout if isinstance(self.timeout, (int, float)) else DEFAULT_TIMEOUT_SECONDS
            ),
        )
        try:
            self.sock = self._ssl_context.wrap_socket(raw_socket, server_hostname=self.host)
        except BaseException:
            raw_socket.close()
            raise


class _ResponseContext:
    """Close a response and its owning connection together."""

    def __init__(self, connection: http.client.HTTPConnection, response: Any) -> None:
        self.connection = connection
        self.response = response

    def __enter__(self) -> Any:
        return self.response

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.response.close()
        finally:
            self.connection.close()


def _request_once(
    target: _Target,
    *,
    deadline: float,
    context: ssl.SSLContext,
) -> _ResponseContext:
    failures: list[BaseException] = []
    for address in target.addresses:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RemoteSourceError("The remote file request timed out.")
        # One unresponsive A/AAAA record must not consume the entire overall
        # deadline and prevent trying the other validated addresses.
        attempt_timeout = min(_MAX_CONNECT_ATTEMPT_SECONDS, remaining)
        if target.scheme == "https":
            connection = _PinnedHTTPSConnection(
                target.hostname,
                address,
                target.port,
                attempt_timeout,
                context,
            )
        else:
            connection = _PinnedHTTPConnection(
                target.hostname,
                address,
                target.port,
                attempt_timeout,
            )
        try:
            connection.request(
                "GET",
                target.request_target,
                headers={
                    "Accept": "application/octet-stream,text/plain;q=0.9,*/*;q=0.1",
                    "Accept-Encoding": "identity",
                    "Connection": "close",
                    "User-Agent": "DNAscape-Data-Manager/1",
                },
            )
            return _ResponseContext(connection, connection.getresponse())
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            failures.append(exc)
            connection.close()
    if failures:
        certificate_error = next(
            (error for error in failures if isinstance(error, ssl.SSLCertVerificationError)), None
        )
        if certificate_error is not None:
            raise RemoteSourceError("The remote server's HTTPS certificate could not be verified.") from certificate_error
        if any(isinstance(error, ssl.SSLError) for error in failures):
            raise RemoteSourceError("The secure connection to the remote server failed.") from failures[-1]
        if all(isinstance(error, TimeoutError) for error in failures):
            raise RemoteSourceError("The remote server did not respond in time. Try again.") from failures[-1]
        raise RemoteSourceError("The remote server could not be reached. Try again later.") from failures[-1]
    raise RemoteSourceError("The remote hostname has no reachable public address.")


def _declared_content_length(response: Any) -> int | None:
    values = response.headers.get_all("Content-Length", [])
    if not values:
        return None
    parsed: list[int] = []
    for value in values:
        for item in str(value).split(","):
            try:
                number = int(item.strip())
            except ValueError as exc:
                raise RemoteSourceError("The server returned an invalid Content-Length.") from exc
            if number < 0:
                raise RemoteSourceError("The server returned an invalid Content-Length.")
            parsed.append(number)
    if not parsed or len(set(parsed)) != 1:
        raise RemoteSourceError("The server returned conflicting Content-Length values.")
    return parsed[0]


def _read_bounded(response: Any, *, max_bytes: int, deadline: float) -> bytes:
    declared = _declared_content_length(response)
    if declared is not None and declared > max_bytes:
        raise RemoteSourceTooLargeError(
            f"The remote file is larger than the {max_bytes / 1024**2:g} MiB limit."
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            raise RemoteSourceError("The remote file request timed out.")
        if getattr(response, "fp", None) is not None:
            raw = getattr(response.fp, "raw", None)
            sock = getattr(raw, "_sock", None)
            if sock is not None:
                sock.settimeout(remaining_time)
        try:
            chunk = response.read(min(_READ_CHUNK_BYTES, max_bytes - total + 1))
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            raise RemoteSourceError("The remote file download was interrupted.") from exc
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise RemoteSourceTooLargeError(
                f"The remote file is larger than the {max_bytes / 1024**2:g} MiB limit."
            )

    if declared is not None and total != declared:
        raise RemoteSourceError("The remote file ended before its declared size was received.")
    if total == 0:
        raise RemoteSourceError("The remote file is empty.")
    return b"".join(chunks)


def _safe_filename(response: Any, final_url: str) -> str:
    disposition = response.getheader("Content-Disposition")
    candidate: str | None = None
    if disposition:
        message = Message()
        message["Content-Disposition"] = disposition
        candidate = message.get_filename()
    if not candidate:
        candidate = unquote(PurePosixPath(urlsplit(final_url).path).name)
    candidate = PurePosixPath(str(candidate or "remote_data").replace("\\", "/")).name
    candidate = _CONTROL_CHARACTERS.sub("", candidate).strip(" .")
    candidate = _UNSAFE_FILENAME_CHARACTERS.sub("_", candidate)
    if not candidate:
        candidate = "remote_data"
    if candidate.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        candidate = f"_{candidate}"
    if len(candidate) > _MAX_FILENAME_LENGTH:
        suffixes = "".join(PurePosixPath(candidate).suffixes)[-_MAX_FILENAME_LENGTH // 2 :]
        stem_limit = max(1, _MAX_FILENAME_LENGTH - len(suffixes))
        candidate = candidate[:stem_limit] + suffixes
    return candidate


def _content_type(response: Any) -> str | None:
    value = response.getheader("Content-Type")
    if not value:
        return None
    media_type = value.split(";", 1)[0].strip().casefold()
    return media_type or None


def fetch_remote_source(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
) -> RemoteSource:
    """Download one public HTTP(S) resource within strict local-app limits.

    Redirects are followed manually and each destination is independently
    resolved, validated and pinned.  HTTP content coding is rejected: a real
    ``.gz`` file is fine, but a server may not transparently gzip an otherwise
    uncompressed representation because downstream format detection operates on
    the downloaded file bytes.
    """

    max_bytes = int(max_bytes)
    timeout = float(timeout)
    max_redirects = int(max_redirects)
    if max_bytes <= 0 or max_bytes > ABSOLUTE_MAX_BYTES:
        raise ValueError(f"max_bytes must be between 1 and {ABSOLUTE_MAX_BYTES:,}.")
    if not 0 < timeout <= _MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout must be between 0 and {_MAX_TIMEOUT_SECONDS:g} seconds.")
    if not 0 <= max_redirects <= _MAX_REDIRECT_LIMIT:
        raise ValueError(f"max_redirects must be between 0 and {_MAX_REDIRECT_LIMIT}.")

    requested_url = str(url).strip()
    current_url = requested_url
    redirects: list[str] = []
    deadline = time.monotonic() + timeout
    context = ssl.create_default_context()

    while True:
        if time.monotonic() >= deadline:
            raise RemoteSourceError("The remote file request timed out.")
        target = _validated_target(current_url)
        with _request_once(target, deadline=deadline, context=context) as response:
            status = int(response.status)
            if status in _REDIRECT_STATUSES:
                location = response.getheader("Location")
                if not location:
                    raise RemoteSourceError(
                        "The remote server returned a redirect without a target."
                    )
                if len(redirects) >= max_redirects:
                    raise RemoteSourceError("The remote file redirected too many times.")
                next_url = urljoin(target.url, location)
                # Validate before recording or issuing the next request.  This
                # prevents a public origin redirecting into localhost/LAN.
                _validated_target(next_url)
                if next_url in {requested_url, *redirects}:
                    raise RemoteSourceError("The remote file returned a redirect loop.")
                redirects.append(next_url)
                current_url = next_url
                continue

            if status != 200:
                raise RemoteSourceError(f"The remote server returned HTTP {status}.")
            content_encoding = (response.getheader("Content-Encoding") or "identity").casefold()
            if content_encoding != "identity":
                raise RemoteSourceError(
                    "The server applied HTTP compression; use its direct file-download link."
                )
            content = _read_bounded(response, max_bytes=max_bytes, deadline=deadline)
            final_url = target.url
            accept_ranges = "bytes" in {
                token.strip().casefold()
                for token in (response.getheader("Accept-Ranges") or "").split(",")
            }
            return RemoteSource(
                requested_url=requested_url,
                final_url=final_url,
                filename=_safe_filename(response, final_url),
                content=content,
                content_type=_content_type(response),
                etag=response.getheader("ETag"),
                last_modified=response.getheader("Last-Modified"),
                accept_ranges=accept_ranges,
                redirects=tuple(redirects),
            )


__all__ = [
    "ABSOLUTE_MAX_BYTES",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_REDIRECTS",
    "DEFAULT_TIMEOUT_SECONDS",
    "RemoteSource",
    "RemoteSourceError",
    "RemoteSourceTooLargeError",
    "UnsafeRemoteSourceError",
    "fetch_remote_source",
]
