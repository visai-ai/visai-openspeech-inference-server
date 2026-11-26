"""Minimal RFC 6455 WebSocket (stdlib only): the server handshake and framing for the live /stream endpoint, plus a
small client used by the tests. No extensions (no permessage-deflate), no subprotocols.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import threading

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


class WSClosed(Exception):
    pass


class WSTooBig(Exception):
    pass


class WSTimeout(Exception):
    """No complete message before the deadline."""


class WSProtocol(Exception):
    """A frame the server does not accept here (bad opcode, unmasked, fragmented before auth, too many fragments)."""


MAX_FRAGMENTS = 64          # frames per message, empty continuation frames included
MAX_CONTROL = 125           # RFC 6455: control frame payloads


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()


def _read_exact(rf, n: int) -> bytes:
    b = rf.read(n)
    if b is None or len(b) < n:
        raise WSClosed("connection closed")
    return b


def read_frame(rf, max_payload: int, require_mask: bool = True):
    """-> (fin, opcode, payload). WSTooBig before reading a payload above max_payload."""
    h = _read_exact(rf, 2)
    fin, op = bool(h[0] & 0x80), h[0] & 0x0F
    masked, n = bool(h[1] & 0x80), h[1] & 0x7F
    if n == 126:
        n = struct.unpack(">H", _read_exact(rf, 2))[0]
    elif n == 127:
        n = struct.unpack(">Q", _read_exact(rf, 8))[0]
    if require_mask and not masked:
        raise WSClosed("client frames must be masked")
    if n > max_payload:
        raise WSTooBig(n)
    mask = _read_exact(rf, 4) if masked else None
    data = _read_exact(rf, n) if n else b""
    if mask:
        data = bytes(b ^ mask[i & 3] for i, b in enumerate(data)) if n < 64 else _unmask(data, mask)
    return fin, op, data


def _unmask(data: bytes, mask: bytes) -> bytes:
    n = len(data)
    m = int.from_bytes((mask * (n // 4 + 1))[:n], "little")
    return (int.from_bytes(data, "little") ^ m).to_bytes(n, "little")


def frame(op: int, payload: bytes = b"", mask: bool = False) -> bytes:
    n = len(payload)
    h = bytes([0x80 | op])
    mb = 0x80 if mask else 0
    if n < 126:
        h += bytes([mb | n])
    elif n < 1 << 16:
        h += bytes([mb | 126]) + struct.pack(">H", n)
    else:
        h += bytes([mb | 127]) + struct.pack(">Q", n)
    if mask:
        k = os.urandom(4)
        return h + k + _unmask(payload, k)
    return h + payload


class ServerConn:
    """One accepted WebSocket on the raw socket (no buffered reader: a read deadline can expire and the read resume
    without losing bytes). The socket's own timeout is the send timeout (sendall raises past it); reads wait on a
    selectors (epoll / poll) registration up to a deadline (no select() fd limit). Writes are locked (the pinger
    thread and the reader both write).

    recv(deadline, max_payload, single_text): the next data message. single_text (before auth): exactly one unfragmented
    text frame of at most max_payload bytes, nothing else. Otherwise fragments are reassembled with a running byte total
    and at most MAX_FRAGMENTS frames (empty continuations count); pings are answered, a close frame raises WSClosed."""

    def __init__(self, sock, max_payload: int, send_timeout: float = 10.0):
        import selectors
        self.sock, self.max = sock, max_payload
        self.sock.settimeout(send_timeout)
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.closed = False
        self.sel = selectors.DefaultSelector()       # epoll / poll: no FD_SETSIZE (1024) limit, unlike select()
        self.sel.register(sock, selectors.EVENT_READ)

    def release(self):
        """Free the selector (the socket itself belongs to the HTTP server, which closes it)."""
        try:
            self.sel.close()
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------- writing
    def send(self, op: int, payload: bytes = b""):
        with self.lock:
            if self.closed:
                return
            try:
                self.sock.sendall(frame(op, payload))
            except OSError:                       # send timeout or reset: the connection is done
                self.closed = True
                raise

    def send_json(self, obj):
        self.send(OP_TEXT, json.dumps(obj, ensure_ascii=False).encode())

    def close(self, code: int = 1000, reason: str = ""):
        with self.lock:
            if self.closed:
                return
            try:
                self.sock.sendall(frame(OP_CLOSE, struct.pack(">H", code) + reason.encode()[:100]))
            except OSError:
                pass
            self.closed = True

    # ---------------------------------------------------------------- reading
    def _need(self, n: int, deadline):
        import time
        while len(self.buf) < n:
            wait = None if deadline is None else deadline - time.monotonic()
            if wait is not None and wait <= 0:
                raise WSTimeout()
            if not self.sel.select(wait):
                raise WSTimeout()
            try:
                b = self.sock.recv(65536)
            except BlockingIOError:
                continue
            if not b:
                raise WSClosed("connection closed")
            self.buf += b

    def _take(self, n: int) -> bytes:
        b = bytes(self.buf[:n])
        del self.buf[:n]
        return b

    def _frame(self, max_payload: int, deadline):
        """One frame, all-or-nothing: bytes stay in the buffer until the whole frame is in (a timeout can resume)."""
        self._need(2, deadline)
        h0, h1 = self.buf[0], self.buf[1]
        n, off = h1 & 0x7F, 2
        if n == 126:
            self._need(4, deadline)
            n, off = struct.unpack(">H", bytes(self.buf[2:4]))[0], 4
        elif n == 127:
            self._need(10, deadline)
            n, off = struct.unpack(">Q", bytes(self.buf[2:10]))[0], 10
        if not h1 & 0x80:
            raise WSProtocol("client frames must be masked")
        op = h0 & 0x0F
        if h0 & 0x70:
            raise WSProtocol("reserved bits set (no extension was negotiated)")
        if op not in (OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG):
            raise WSProtocol("unknown opcode")
        if op >= 0x8 and (n > MAX_CONTROL or not h0 & 0x80):
            raise WSProtocol("control frames must be short and unfragmented")
        if n > max_payload:
            raise WSTooBig(n)
        self._need(off + 4 + n, deadline)
        self._take(off)
        mask = self._take(4)
        data = self._take(n)
        return bool(h0 & 0x80), op, (_unmask(data, mask) if n else b"")

    def recv(self, deadline=None, max_payload: int | None = None, single_text: bool = False):
        limit = self.max if max_payload is None else max_payload
        if single_text:
            fin, op, data = self._frame(limit, deadline)
            if op != OP_TEXT or not fin:
                raise WSProtocol("expected one unfragmented text message")
            return op, data
        parts, op0, total, nfrag = [], None, 0, 0
        while True:
            fin, op, data = self._frame(limit, deadline)
            if op == OP_PING:
                self.send(OP_PONG, data)
                continue
            if op == OP_PONG:
                continue
            if op == OP_CLOSE:
                raise WSClosed("client closed")
            if op in (OP_TEXT, OP_BIN):
                if op0 is not None:
                    raise WSProtocol("new message inside a fragmented one")
                op0 = op
            elif op != OP_CONT or op0 is None:
                raise WSProtocol(f"unexpected opcode {op}")
            nfrag += 1
            total += len(data)
            if nfrag > MAX_FRAGMENTS:
                raise WSProtocol("too many fragments")
            if total > limit:
                raise WSTooBig(total)
            parts.append(data)
            if fin:
                return op0, b"".join(parts)


class Client:
    """Test / tool client (plain TCP): Client(host, port, path, headers=None)."""

    def __init__(self, host: str, port: int, path: str = "/stream", headers: dict | None = None,
                 timeout: float = 60.0):
        import socket
        s = socket.create_connection((host, port), timeout=timeout)
        self.sock = s
        self.rf = s.makefile("rb")
        key = base64.b64encode(os.urandom(16)).decode()
        req = [f"GET {path} HTTP/1.1", f"Host: {host}:{port}", "Upgrade: websocket", "Connection: Upgrade",
               f"Sec-WebSocket-Key: {key}", "Sec-WebSocket-Version: 13"]
        req += [f"{k}: {v}" for k, v in (headers or {}).items()]
        s.sendall(("\r\n".join(req) + "\r\n\r\n").encode())
        status = self.rf.readline().decode()
        hdrs = {}
        while True:
            line = self.rf.readline().decode().strip()
            if not line:
                break
            k, _, v = line.partition(":")
            hdrs[k.strip().lower()] = v.strip()
        self.status = int(status.split()[1])
        if self.status == 101 and hdrs.get("sec-websocket-accept") != accept_key(key):
            raise ConnectionError("bad Sec-WebSocket-Accept")

    def send(self, op, payload: bytes):
        self.sock.sendall(frame(op, payload, mask=True))

    def send_json(self, obj):
        self.send(OP_TEXT, json.dumps(obj).encode())

    def recv(self):
        """-> ("text", obj) | ("binary", bytes) | ("close", code); pings are answered and skipped."""
        while True:
            fin, op, data = read_frame(self.rf, 1 << 26, require_mask=False)
            if op == OP_PING:
                try:
                    self.send(OP_PONG, data)
                except OSError:                       # the server may already have closed its side
                    pass
                continue
            if op == OP_CLOSE:
                return "close", struct.unpack(">H", data[:2])[0] if len(data) >= 2 else 1005
            if op == OP_TEXT:
                return "text", json.loads(data)
            return "binary", data

    def close(self):
        try:
            self.send(OP_CLOSE, struct.pack(">H", 1000))
        except OSError:
            pass
        self.sock.close()
