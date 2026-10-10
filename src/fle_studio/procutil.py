"""Per-OS helpers for the engine process: how much it has read (load progress), how much memory it holds, and
making sure it does not outlive FLE Studio. Standard library and ctypes only.
"""

from __future__ import annotations

import ctypes
import signal
import subprocess
from pathlib import Path
from typing import Any

from .paths import IS_LINUX, IS_MAC, IS_WIN

# ------------------------------------------------------------------------------------------------ spawning

_prctl: Any = None
if IS_LINUX:
    try:
        _prctl = ctypes.CDLL("libc.so.6", use_errno=True).prctl  # loaded before any fork
    except (OSError, AttributeError):
        _prctl = None


def _die_with_parent() -> None:  # runs in the child between fork and exec
    if _prctl is not None:
        _prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG: SIGTERM when the thread that forked us ends


def popen_kwargs() -> dict[str, Any]:
    """Extra Popen arguments. On Linux the engine gets SIGTERM if FLE Studio dies; the Popen call must therefore come
    from a thread that lives as long as the engine (see engine.Supervisor)."""
    if IS_WIN:
        return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    if IS_LINUX and _prctl is not None:
        return {"preexec_fn": _die_with_parent}
    return {}


class KillOnCloseJob:
    """Windows job object that kills its processes when FLE Studio's handle to it closes (also on a crash or when
    the console window is closed). A no-op elsewhere."""

    def __init__(self) -> None:
        self.handle = None
        if not IS_WIN:
            return
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        k32.CreateJobObjectW.restype = ctypes.c_void_p
        k32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32),
                        ("SchedulingClass", ctypes.c_uint32)]

        class Io(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount",
                                                       "OtherOperationCount", "ReadTransferCount",
                                                       "WriteTransferCount", "OtherTransferCount")]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", Io), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        handle = k32.CreateJobObjectW(None, None)
        if not handle:
            return
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if k32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.handle = handle

    def add(self, proc: subprocess.Popen[Any]) -> None:
        if self.handle is not None:
            ctypes.windll.kernel32.AssignProcessToJobObject(self.handle, int(proc._handle))  # type: ignore[attr-defined]


def ask_to_stop(proc: subprocess.Popen[Any]) -> None:
    """The polite stop: SIGTERM lets llama-server finish its cleanup (and FLE save its expert profile)."""
    if IS_WIN:
        proc.terminate()
    else:
        proc.send_signal(signal.SIGTERM)


# ------------------------------------------------------------------------------------------------ reading


if IS_MAC:
    try:
        _libproc: Any = ctypes.CDLL("/usr/lib/libproc.dylib")
        _libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    except OSError:
        _libproc = None
else:
    _libproc = None


def _mac_rusage(pid: int) -> tuple[int, int] | None:
    """(bytes read from disk, physical footprint) from rusage_info_v2."""
    if _libproc is None:
        return None
    buf = (ctypes.c_uint64 * 40)()
    if _libproc.proc_pid_rusage(pid, 2, ctypes.byref(buf)) != 0:  # RUSAGE_INFO_V2
        return None
    # 16-byte uuid = two slots, then uint64 fields: [6] resident, [7] phys_footprint, [16] diskio_bytesread
    return int(buf[2 + 16]), int(buf[2 + 7])


def _win_counters(proc: subprocess.Popen[Any]) -> tuple[int, int] | None:
    k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

    class Io(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                   "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class Mem(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    handle = ctypes.c_void_p(int(proc._handle))  # type: ignore[attr-defined]
    io, mem = Io(), Mem()
    mem.cb = ctypes.sizeof(Mem)
    ok_io = k32.GetProcessIoCounters(handle, ctypes.byref(io))
    ok_mem = k32.K32GetProcessMemoryInfo(handle, ctypes.byref(mem), mem.cb)
    if not (ok_io or ok_mem):
        return None
    return (int(io.ReadTransferCount) if ok_io else 0), (int(mem.WorkingSetSize) if ok_mem else 0)


def _linux_counters(pid: int) -> tuple[int | None, int | None]:
    # /proc/PID/io of another account's process is closed to us; its status (and so the memory) stays readable.
    read = rss = None
    try:
        io = Path(f"/proc/{pid}/io").read_text()
        vals = dict(line.split(": ", 1) for line in io.splitlines() if ": " in line)
        read = max(int(vals.get("rchar", 0)), int(vals.get("read_bytes", 0)))
    except (OSError, ValueError):
        pass
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return read, rss


def read_and_memory_pid(pid: int) -> tuple[int | None, int | None]:
    """Like read_and_memory for a process FLE Studio did not start, such as the main process of a service (Linux)."""
    if not IS_LINUX or pid <= 0:
        return None, None
    return _linux_counters(pid)


def read_and_memory(proc: subprocess.Popen[Any] | None) -> tuple[int | None, int | None]:
    """(bytes the process has read so far, memory it holds) or None where this OS does not tell."""
    if proc is None or proc.poll() is not None:
        return None, None
    try:
        if IS_LINUX:
            return _linux_counters(proc.pid)
        if IS_MAC:
            got = _mac_rusage(proc.pid)
            return got if got else (None, None)
        if IS_WIN:
            got = _win_counters(proc)
            return got if got else (None, None)
    except (OSError, ValueError, AttributeError):
        pass
    return None, None
