"""Minimal HTTP helpers on the standard library (no third-party HTTP client needed)."""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from .base import ProviderError

log = logging.getLogger("atlas.http")


def _error_detail(err: urllib.error.HTTPError) -> str:
    try:
        body = err.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - best effort only
        return ""
    try:
        data = json.loads(body)
        if isinstance(data, dict):
            inner = data.get("error", data)
            if isinstance(inner, dict):
                return str(inner.get("message", inner))
            return str(inner)
    except json.JSONDecodeError:
        pass
    return body[:300]


def _open(req: urllib.request.Request, timeout: float, what: str):
    try:
        return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - URL comes from config
    except urllib.error.HTTPError as err:
        detail = _error_detail(err)
        if err.code in (401, 403):
            raise ProviderError(f"{what}: authentication failed ({err.code}). Check the API key.") from err
        if err.code == 404:
            raise ProviderError(f"{what}: not found (404). {detail}".strip()) from err
        raise ProviderError(f"{what}: HTTP {err.code}. {detail}".strip()) from err
    except urllib.error.URLError as err:
        raise ProviderError(f"{what}: cannot connect ({err.reason}).") from err
    except (socket.timeout, TimeoutError) as err:
        raise ProviderError(f"{what}: timed out after {timeout:.0f}s.") from err


def _request(url: str, payload: dict[str, Any] | None, headers: dict[str, str] | None) -> urllib.request.Request:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    return req


def get_json(url: str, timeout: float, what: str, headers: dict[str, str] | None = None) -> Any:
    with _open(_request(url, None, headers), timeout, what) as resp:
        try:
            return json.loads(resp.read().decode("utf-8"))
        except json.JSONDecodeError as err:
            raise ProviderError(f"{what}: server returned invalid JSON.") from err


def stream_lines(
    url: str, payload: dict[str, Any], timeout: float, what: str, headers: dict[str, str] | None = None
) -> Iterator[str]:
    """POST JSON and yield the response body line by line as it arrives.

    `timeout` is an idle timeout: it applies to each read, not the whole response.
    """
    log.debug("POST %s", url, extra={"model": payload.get("model")})
    resp = _open(_request(url, payload, headers), timeout, what)
    try:
        while True:
            try:
                raw = resp.readline()
            except (socket.timeout, TimeoutError) as err:
                raise ProviderError(f"{what}: no data for {timeout:.0f}s, gave up.") from err
            if not raw:
                break
            line = raw.decode("utf-8", "replace").strip()
            if line:
                yield line
    finally:
        resp.close()
