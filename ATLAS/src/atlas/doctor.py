"""Environment checks: Python, GPU, Ollama, model, workspace, secrets file."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

from .config import Settings, env_file_is_private
from .providers import OllamaProvider, ProviderError, make_cloud_provider

Status = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


def check_python() -> Check:
    v = sys.version_info
    ok = v >= (3, 10)
    return Check("Python", "ok" if ok else "fail", f"{v.major}.{v.minor}.{v.micro}" + ("" if ok else " (need 3.10+)"))


def check_gpu() -> Check:
    if not shutil.which("nvidia-smi"):
        return Check("NVIDIA GPU", "warn", "nvidia-smi not found: Ollama will run on the CPU (much slower)")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as err:
        return Check("NVIDIA GPU", "warn", f"nvidia-smi failed: {err}")
    if not out:
        return Check("NVIDIA GPU", "warn", "no GPU reported by nvidia-smi")
    name, mem, driver = (p.strip() for p in out.splitlines()[0].split(","))
    return Check("NVIDIA GPU", "ok", f"{name}, {mem}, driver {driver}")


def check_ollama(settings: Settings) -> list[Check]:
    checks = []
    binary = shutil.which("ollama")
    checks.append(Check("Ollama installed", "ok" if binary else "warn",
                        binary or "not on PATH (fine if Ollama runs elsewhere)"))
    provider = OllamaProvider(settings.local_model, host=settings.ollama_host)
    try:
        checks.append(Check("Ollama server", "ok", provider.health()))
    except ProviderError as err:
        checks.append(Check("Ollama server", "fail", f"{err} Start it with: ollama serve"))
        return checks
    try:
        if provider.has_model():
            checks.append(Check("Local model", "ok", settings.local_model))
        else:
            checks.append(Check("Local model", "fail",
                                f"{settings.local_model} not downloaded. Run: ollama pull {settings.local_model}"))
    except ProviderError as err:
        checks.append(Check("Local model", "fail", str(err)))
    return checks


def check_workspace(settings: Settings) -> Check:
    ws = settings.workspace
    if not ws.is_dir():
        return Check("Workspace", "fail", f"{ws} does not exist")
    if not os.access(ws, os.R_OK | os.W_OK):
        return Check("Workspace", "fail", f"{ws} is not readable and writable")
    return Check("Workspace", "ok", str(ws))


def check_env_file(settings: Settings) -> Check:
    if settings.env_file is None:
        return Check(".env file", "warn", "none found; using defaults")
    if settings.cloud_api_key and not env_file_is_private(settings.env_file):
        return Check(".env file", "warn",
                     f"{settings.env_file} contains an API key but others can read it. Run: chmod 600 {settings.env_file}")
    return Check(".env file", "ok", str(settings.env_file))


def check_cloud(settings: Settings) -> Check:
    if not settings.cloud_configured:
        return Check("Cloud", "ok", "not configured (local only)")
    try:
        provider = make_cloud_provider(settings)
    except ProviderError as err:
        return Check("Cloud", "fail", str(err))
    state = "allowed" if settings.cloud_allowed else "configured but disabled (ATLAS_CLOUD_ALLOWED=false)"
    # Deliberately no network call here: doctor must not contact the cloud without consent.
    return Check("Cloud", "ok", f"{provider.model} at {getattr(provider, 'host', '?')}, {state}")


def run_checks(settings: Settings) -> list[Check]:
    return [
        check_python(),
        check_gpu(),
        *check_ollama(settings),
        check_workspace(settings),
        check_env_file(settings),
        check_cloud(settings),
    ]
