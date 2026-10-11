"""What this machine has: OS, CPU, memory, GPUs and disk. Standard library only; macOS, Linux and Windows.

Everything here is read-only and cheap except `gpus()` (one nvidia-smi call, ~50 ms), which the metrics
sampler calls every couple of seconds.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import struct
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from .paths import IS_LINUX, IS_MAC, IS_WIN

GiB = 1024**3
MiB = 1024**2


def _run(args: list[str], timeout: float = 5.0) -> str:
    """stdout of a command, or "" if it is missing, fails or hangs. Never opens a console window on Windows."""
    try:
        res = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return res.stdout if res.returncode == 0 else ""


# ------------------------------------------------------------------------------------------------ memory


def memory() -> dict[str, int]:
    """Physical memory in bytes: total and available (what can be used without swapping)."""
    if IS_LINUX:
        info: dict[str, int] = {}
        try:
            with open("/proc/meminfo", encoding="ascii") as f:
                for line in f:
                    key, _, rest = line.partition(":")
                    info[key] = int(rest.split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            return {"total": 0, "available": 0}
        return {"total": info.get("MemTotal", 0), "available": info.get("MemAvailable", info.get("MemFree", 0))}
    if IS_MAC:
        total = int(_run(["sysctl", "-n", "hw.memsize"]).strip() or 0)
        vm = _run(["vm_stat"])
        m = re.search(r"page size of (\d+) bytes", vm)
        page = int(m.group(1)) if m else 16384

        def pages(name: str) -> int:
            hit = re.search(rf"{name}:\s+(\d+)", vm)
            return int(hit.group(1)) if hit else 0

        free = pages("Pages free") + pages("Pages inactive") + pages("Pages speculative") + pages("Pages purgeable")
        return {"total": total, "available": min(total, free * page)}
    if IS_WIN:

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MemoryStatusEx()
        st.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
            return {"total": int(st.ullTotalPhys), "available": int(st.ullAvailPhys)}
    return {"total": 0, "available": 0}


# ------------------------------------------------------------------------------------------------ CPU


def _linux_physical_cores(text: str) -> int | None:
    cores = set()
    for block in text.split("\n\n"):
        phys = re.search(r"^physical id\s*:\s*(\d+)", block, re.M)
        core = re.search(r"^core id\s*:\s*(\d+)", block, re.M)
        if phys and core:
            cores.add((phys.group(1), core.group(1)))
    return len(cores) or None


def _win_physical_cores() -> int | None:
    k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    size = ctypes.c_ulong(0)
    k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))  # RelationProcessorCore = 0
    if not size.value:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
        return None
    count, offset = 0, 0
    while offset < size.value:
        relationship, entry_size = struct.unpack_from("<II", buf, offset)
        if relationship == 0:
            count += 1
        if entry_size == 0:
            break
        offset += entry_size
    return count or None


def _win_cpu_name() -> str:
    try:
        import winreg  # noqa: PLC0415 - Windows only

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except OSError:
        return platform.processor()


@lru_cache(maxsize=1)
def cpu() -> dict[str, Any]:
    logical = os.cpu_count() or 1
    name, physical, avx2 = platform.processor() or platform.machine(), None, None
    if IS_LINUX:
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
        name = m.group(1).strip() if m else name
        physical = _linux_physical_cores(text)
        flags = re.search(r"^flags\s*:\s*(.+)$", text, re.M)
        avx2 = bool(flags and " avx2" in f" {flags.group(1)}")
    elif IS_MAC:
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or name
        physical = int(_run(["sysctl", "-n", "hw.physicalcpu"]).strip() or 0) or None
        if platform.machine() == "x86_64":
            avx2 = "AVX2" in _run(["sysctl", "-n", "machdep.cpu.leaf7_features"])
    elif IS_WIN:
        name = _win_cpu_name()
        try:
            physical = _win_physical_cores()
            avx2 = bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass
    return {
        "name": name,
        "arch": platform.machine().lower(),
        "logical": logical,
        "physical": physical or max(1, logical // 2),
        "avx2": avx2,
    }


# ------------------------------------------------------------------------------------------------ GPUs


def _nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if IS_WIN:
        for candidate in (
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "NVIDIA Corporation"
            / "NVSMI"
            / "nvidia-smi.exe",
        ):
            if candidate.exists():
                return str(candidate)
    return None


_NV_FIELDS = "index,name,memory.total,memory.used,utilization.gpu,driver_version,compute_cap,power.draw,temperature.gpu"


def nvidia() -> list[dict[str, Any]]:
    smi = _nvidia_smi()
    if not smi:
        return []
    out = _run([smi, f"--query-gpu={_NV_FIELDS}", "--format=csv,noheader,nounits"])
    fields = _NV_FIELDS.split(",")
    if not out:  # old drivers do not know compute_cap
        fields = [f for f in fields if f != "compute_cap"]
        out = _run([smi, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.strip().splitlines():
        row = dict(zip(fields, (v.strip() for v in line.split(",")), strict=False))

        def num(key: str, row: dict[str, str] = row) -> float | None:
            try:
                return float(row.get(key, ""))
            except ValueError:
                return None

        total = num("memory.total") or 0
        used = num("memory.used") or 0
        gpus.append(
            {
                "vendor": "nvidia",
                "index": int(num("index") or 0),
                "name": row.get("name", "NVIDIA GPU"),
                "memory_total": int(total * MiB),
                "memory_used": int(used * MiB),
                "utilization": num("utilization.gpu"),
                "power_w": num("power.draw"),
                "temperature_c": num("temperature.gpu"),
                "driver": row.get("driver_version", ""),
                "compute_cap": row.get("compute_cap", ""),
            }
        )
    return gpus


def _metal_limit(total: int) -> int:
    """How much unified memory macOS lets the GPU wire by default (iogpu.wired_limit_mb overrides it)."""
    override = _run(["sysctl", "-n", "iogpu.wired_limit_mb"]).strip()
    if override.isdigit() and int(override) > 0:
        return int(override) * MiB
    return int(total * (0.75 if total >= 64 * GiB else 0.67))


def apple(mem_total: int) -> list[dict[str, Any]]:
    if not (IS_MAC and platform.machine() == "arm64"):
        return []
    chip = _apple_chip()
    return [
        {
            "vendor": "apple",
            "index": 0,
            "name": f"{chip} GPU",
            "unified": True,
            "memory_total": mem_total,
            "metal_limit": _metal_limit(mem_total),
        }
    ]


@lru_cache(maxsize=1)
def _apple_chip() -> str:
    return _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or "Apple Silicon"


@lru_cache(maxsize=1)
def other_gpus() -> list[dict[str, Any]]:
    """AMD and Intel GPUs, for the Vulkan engine. Names only; their memory is not reliably readable here.

    Cached: on Windows this asks PowerShell, which takes a second or two, and the hardware does not change.
    """
    found: list[dict[str, Any]] = []
    if IS_LINUX:
        vendors = {"0x1002": "amd", "0x8086": "intel"}
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]")):
            try:
                vendor = (card / "device" / "vendor").read_text().strip()
            except OSError:
                continue
            if vendor not in vendors:
                continue
            vram = card / "device" / "mem_info_vram_total"
            total = int(vram.read_text()) if vram.exists() else 0
            found.append(
                {
                    "vendor": vendors[vendor],
                    "index": len(found),
                    "name": f"{vendors[vendor].upper()} GPU",
                    "memory_total": total,
                }
            )
    elif IS_WIN:
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }",
            ],
            timeout=8,
        )
        for name in out.splitlines():
            name = name.strip()
            if name and not name.lower().startswith(("nvidia", "microsoft")):
                found.append(
                    {
                        "vendor": "amd" if "amd" in name.lower() or "radeon" in name.lower() else "intel",
                        "index": len(found),
                        "name": name,
                        "memory_total": 0,
                    }
                )
    return found


def gpus(mem_total: int | None = None) -> list[dict[str, Any]]:
    return nvidia() or apple(mem_total if mem_total is not None else memory()["total"])


# ------------------------------------------------------------------------------------------------ all


def disk_free(path: Path) -> int:
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 0


@lru_cache(maxsize=1)
def _os_name() -> str:
    if IS_MAC:
        return f"macOS {platform.mac_ver()[0]}"
    if IS_WIN:
        return f"Windows {platform.release()} ({platform.version()})"
    try:
        text = Path("/etc/os-release").read_text(encoding="utf-8")
        m = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', text, re.M)
        if m:
            return m.group(1)
    except OSError:
        pass
    return f"{platform.system()} {platform.release()}"


def info(models_dir: Path) -> dict[str, Any]:
    mem = memory()
    gpu_list = gpus(mem["total"])
    return {
        "os": _os_name(),
        "platform": "macos" if IS_MAC else "windows" if IS_WIN else "linux",
        "python": platform.python_version(),
        "cpu": cpu(),
        "memory": mem,
        "gpus": gpu_list,
        "other_gpus": [] if gpu_list else other_gpus(),
        "disk_free": disk_free(models_dir),
        "models_dir": str(models_dir),
    }
