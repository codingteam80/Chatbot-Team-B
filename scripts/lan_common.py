from __future__ import annotations

import ipaddress
import json
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

DEFAULT_LAN_PORT = 8501
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"


def _usable_private_ipv4(value: str) -> bool:
    try:
        address = ipaddress.ip_address(str(value).strip())
    except ValueError:
        return False
    return bool(
        address.version == 4
        and address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_unspecified
    )


def detect_lan_ipv4_addresses() -> list[str]:
    """Best-effort private IPv4 discovery without external network access."""

    found: list[str] = []

    def add(value: str | None) -> None:
        if value and _usable_private_ipv4(value) and value not in found:
            found.append(value)

    # Prefer the address Windows would use for a normal routed connection.
    # UDP connect does not transmit application data and works even when the
    # destination is unreachable; it only asks the OS which interface to use.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.255.255.255", 9))
        add(sock.getsockname()[0])
    except OSError:
        pass
    finally:
        sock.close()

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            add(info[4][0])
    except OSError:
        pass

    return found


def preferred_lan_ipv4() -> str | None:
    addresses = detect_lan_ipv4_addresses()
    return addresses[0] if addresses else None


def build_employee_url(host: str, port: int = DEFAULT_LAN_PORT) -> str:
    return f"http://{host}:{int(port)}"


def port_is_available(port: int = DEFAULT_LAN_PORT, host: str = "0.0.0.0") -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, int(port)))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def local_tcp_port_is_open(port: int = DEFAULT_LAN_PORT) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.4)
    try:
        return probe.connect_ex(("127.0.0.1", int(port))) == 0
    finally:
        probe.close()


def ollama_models(timeout: float = 4.0) -> tuple[bool, set[str], str]:
    try:
        with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as error:
        return False, set(), f"{type(error).__name__}: {error}"

    names: set[str] = set()
    for item in payload.get("models") or []:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model"):
            value = str(item.get(key) or "").strip()
            if value:
                names.add(value)
    return True, names, ""


def missing_required_models(installed: Iterable[str], required: Iterable[str]) -> list[str]:
    installed_set = {str(value).strip() for value in installed if str(value).strip()}
    return [
        model
        for model in required
        if str(model).strip() and str(model).strip() not in installed_set
    ]


def write_employee_shortcut(root: Path, url: str) -> Path:
    path = Path(root) / "DocuBot_Employee_Link.url"
    content = "[InternetShortcut]\r\n" f"URL={url}\r\n"
    path.write_text(content, encoding="utf-8")
    return path
