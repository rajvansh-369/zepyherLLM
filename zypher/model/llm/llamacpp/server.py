"""llama-server as a child process: start it, wait for it, take it down.

It listens on a loopback port nobody else was told, behind a key made up for
this run. The zypherLL API is what the outside talks to; this is its engine.
"""

import atexit
import os
import secrets
import shlex
import socket
import subprocess
import time

from zypher import config

LOG_NAME = "llama-server.log"
START_TIMEOUT = 600


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))

        return probe.getsockname()[1]


def _kill_with_parent(process):
    """Tie the child's life to ours (Windows).

    atexit covers a normal exit. A killed console window or a crashed
    interpreter runs no handlers, and would leave llama-server holding the
    whole GPU with nothing to tell it to stop. A job object set to kill on
    close is closed by the OS when this process dies, however it dies.
    """

    if os.name != "nt":
        return None

    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    KILL_ON_JOB_CLOSE = 0x2000
    EXTENDED_LIMIT_INFORMATION = 9

    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        job = kernel.CreateJobObjectW(None, None)

        if not job:
            return None

        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = KILL_ON_JOB_CLOSE

        kernel.SetInformationJobObject(
            job, EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)
        )
        kernel.AssignProcessToJobObject(job, int(process._handle))

        # The handle has to outlive this function: closing it is the kill.
        return job

    except Exception:
        return None


class LlamaServer:

    def __init__(self, executable, model_path, plan):
        self.executable = executable
        self.model_path = model_path
        self.plan = plan
        self.port = _free_port()
        self.key = secrets.token_urlsafe(24)
        self.process = None
        self.log_path = os.path.join(config.LOG_DIR, LOG_NAME)
        self._log = None
        self._job = None

    def command(self):
        plan = self.plan

        args = [
            self.executable,
            "--model", self.model_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--alias", config.MODEL_ID,
            "--ctx-size", str(plan["context"]),
            "--gpu-layers", str(plan["gpu_layers"]),
            # With the layers decided here, llama.cpp is not to second-guess
            # them: its own fitting keeps a 1 GiB margin, which on a 6 GB card
            # pushes layers of a model that fits onto the CPU.
            "--fit", "on" if plan["fit"] else "off",
            "--flash-attn", "on",
            "--cache-type-k", plan["kv_type"],
            "--cache-type-v", plan["kv_type"],
            # One generation at a time: every slot would be handed a share of
            # the context window, and there is one GPU.
            "--parallel", "1",
            # When old turns are dropped, what follows them moves up. This
            # lets the cached state for that moved text be shifted into place
            # instead of computed again.
            "--cache-reuse", "256",
            "--cache-ram", str(config.CACHE_RAM_MB),
        ]

        if plan["load_mode"] != "auto":
            args += ["--load-mode", plan["load_mode"]]

        if config.IDLE_UNLOAD_SECONDS > 0:
            args += ["--sleep-idle-seconds", str(config.IDLE_UNLOAD_SECONDS)]

        if config.LLAMA_ARGS.strip():
            args += shlex.split(config.LLAMA_ARGS, posix=os.name != "nt")

        return args

    def start(self):
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self._log = open(self.log_path, "w", encoding="utf-8", errors="replace")

        # The key goes through the environment: a command line is readable
        # by every process on the machine.
        #
        # The built-in web page is switched off the same way. Its flag has
        # been renamed once already (--no-webui, now --no-ui); an environment
        # variable a build does not know is ignored, where a flag it does not
        # know stops it from starting.
        environment = dict(
            os.environ,
            LLAMA_API_KEY=self.key,
            LLAMA_ARG_UI="0",
            LLAMA_ARG_WEBUI="0",
        )

        self.process = subprocess.Popen(
            self.command(),
            stdout=self._log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=environment,
            cwd=os.path.dirname(self.executable) or None,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        self._job = _kill_with_parent(self.process)
        atexit.register(self.stop)

    def wait_ready(self, probe, on_slow=None, slow_after=8.0):
        """Block until probe() says the model is loaded. probe returns True
        when ready and False while loading. on_slow is called once if the
        wait passes slow_after seconds."""

        started = time.monotonic()
        deadline = started + START_TIMEOUT

        while time.monotonic() < deadline:
            if on_slow is not None and time.monotonic() - started > slow_after:
                on_slow()
                on_slow = None

            if self.process.poll() is not None:
                raise RuntimeError(
                    "llama-server exited with code {} while loading.\n{}".format(
                        self.process.returncode, self.log_tail()
                    )
                )

            if probe():
                return

            time.sleep(0.25)

        self.stop()

        raise RuntimeError("llama-server did not come up in {} s.\n{}".format(
            START_TIMEOUT, self.log_tail()
        ))

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def log_tail(self, lines=25):
        try:
            if self._log is not None:
                self._log.flush()

            with open(self.log_path, "r", encoding="utf-8", errors="replace") as handle:
                tail = handle.read().splitlines()[-lines:]
        except OSError:
            return "(no log at {})".format(self.log_path)

        return "--- last lines of {} ---\n{}".format(self.log_path, "\n".join(tail))

    def stop(self):
        process, self.process = self.process, None

        if process is not None and process.poll() is None:
            process.terminate()

            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(10)

        if self._log is not None:
            self._log.close()
            self._log = None
