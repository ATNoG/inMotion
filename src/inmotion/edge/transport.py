"""Execute commands and move files on a device: locally or over SSH.

Two transports, one interface. ``LocalTransport`` runs the same commands on
this host inside a directory that stands in for the router's
``/mnt/extra/inmotion/edge``, which is what makes the ``smoke`` profile a real
end-to-end run instead of a mock. ``SshTransport`` addresses the two lab
routers: key authentication for device 1, password authentication through the
device 1 jump host for device 2.

Three rules this module exists to enforce:

* **A command always has a hard timeout and its whole process group is killed.**
  A busybox command that ignores SIGTERM must not wedge a campaign.
* **The device-2 password is never stored, printed or put in argv.** It is read
  from ``$WAVECOM_PASSWORD`` by an askpass helper created in a private
  (``0700``) temporary directory at run time and removed afterwards.
* **Everything that leaves this module is redacted.** Error strings and captured
  stderr pass through :func:`redact`, so a command that echoes its environment
  cannot leak the secret into a log or a manifest.

Commands are sent as a **single line**: both routers' login shells ``eval`` the
command, and a newline is flattened into a space there, which turns a multi-line
``if ... then`` into a syntax error. Loops and conditionals are written with
``;`` separators instead; the stress window is a file executed by ``sh``, so it
is unaffected.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import shlex
import shutil
import signal
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from posixpath import dirname as rdirname
from typing import Protocol, runtime_checkable

from inmotion.edge import contracts

#: Environment variable the device-2 password is read from. Never a file.
PASSWORD_ENV = "WAVECOM_PASSWORD"

#: Seconds before a plain command is considered wedged. Callers pass a larger
#: value for long windows; this is the floor.
DEFAULT_TIMEOUT_S = 120.0


class TransportError(RuntimeError):
    """A transport operation failed; the message is always safe to print."""


class TransportTimeout(TransportError):
    """The command exceeded its hard timeout and its process group was killed."""


def redact(text: str, secrets: tuple[str, ...] = ()) -> str:
    """Remove known secrets from ``text``.

    Collects ``$WAVECOM_PASSWORD`` at call time rather than import time, so a
    test that sets it after import still gets redaction, and nothing is cached.
    """
    values = [value for value in (*secrets, os.environ.get(PASSWORD_ENV, "")) if value]
    for value in values:
        text = text.replace(value, "***")
    return text


def shell_quote(value: str | Path) -> str:
    """POSIX shell quoting for one argument, safe for busybox ``sh``."""
    return shlex.quote(str(value))


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    """SIGKILL the command's process group, tolerating an already-dead child."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        process.kill()


def _communicate(
    argv: list[str],
    *,
    timeout: float,
    stdin: bytes | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run ``argv`` to completion, killing the whole group on timeout."""
    process = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        _partial_out, partial_err = process.communicate()
        detail = redact(_decode(partial_err).strip())[-400:]
        raise TransportTimeout(
            f"command timed out after {timeout:.1f}s and was killed: {' '.join(argv[:6])}"
            + (f"; stderr: {detail}" if detail else "")
        ) from None
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


@runtime_checkable
class DeviceTransport(Protocol):
    """What bench/replay/power/stress need from a device, and nothing more."""

    key: str
    label: str
    root: str

    def run(
        self,
        cmd: str,
        timeout: float = DEFAULT_TIMEOUT_S,
        *,
        input: bytes | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]: ...

    def upload(self, local: Path, remote: str | Path, *, replace: bool = True) -> None: ...

    def download(self, remote: str | Path, local: Path) -> Path: ...

    def read_text(self, remote: str | Path) -> str: ...

    def write_text(self, remote: str | Path, text: str) -> None: ...

    def mkdir(self, remote: str | Path) -> None: ...

    def exists(self, remote: str | Path) -> bool: ...

    def sha256(self, remote: str | Path) -> str: ...

    def preflight(self) -> dict[str, object]: ...


@dataclass
class LocalTransport:
    """A device that happens to be this host, rooted in a local directory."""

    root: str
    key: str = "local"
    label: str = "local host"

    def __post_init__(self) -> None:
        Path(self.root).mkdir(parents=True, exist_ok=True)

    def run(
        self,
        cmd: str,
        timeout: float = DEFAULT_TIMEOUT_S,
        *,
        input: bytes | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        try:
            raw = _communicate(["/bin/sh", "-c", cmd], timeout=timeout, stdin=input)
        except TransportTimeout as exc:
            raise TransportTimeout(str(exc)) from None
        result = subprocess.CompletedProcess(
            raw.args, raw.returncode, redact(_decode(raw.stdout)), redact(_decode(raw.stderr))
        )
        if check and result.returncode != 0:
            raise TransportError(
                f"command failed ({result.returncode}): {result.stderr.strip()[:400]}"
            )
        return result

    def upload(self, local: Path, remote: str | Path, *, replace: bool = True) -> None:
        source = Path(local)
        target = Path(str(remote))
        if not source.exists():
            raise TransportError(f"upload: {source} does not exist")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and replace:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)

    def download(self, remote: str | Path, local: Path) -> Path:
        source = Path(str(remote))
        if not source.exists():
            raise TransportError(f"download: {source} does not exist")
        target = Path(local)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        return target

    def read_text(self, remote: str | Path) -> str:
        return Path(str(remote)).read_text(encoding="utf-8", errors="replace")

    def write_text(self, remote: str | Path, text: str) -> None:
        target = Path(str(remote))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def mkdir(self, remote: str | Path) -> None:
        Path(str(remote)).mkdir(parents=True, exist_ok=True)

    def exists(self, remote: str | Path) -> bool:
        return Path(str(remote)).exists()

    def sha256(self, remote: str | Path) -> str:
        path = Path(str(remote))
        if not path.is_file():
            return ""
        return _hash_file(path)

    def preflight(self) -> dict[str, object]:
        """The same shape as the SSH probe, so a caller never branches on transport."""
        return local_preflight(Path(self.root))


def local_preflight(root: Path) -> dict[str, object]:
    """Gather the facts a campaign manifest records about a device."""
    import platform

    def read(path: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""

    mem_kb = 0
    for line in read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            mem_kb = int(line.split()[1])
            break
    tools = ("taskset", "gzip", "tar", "curl", "nc", "nproc", "getconf", "stat", "gcc", "sha256sum")
    return {
        "transport": "local",
        "os": platform.system(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "euid": os.geteuid(),
        "cpu": platform.processor() or "unknown",
        "cores": os.cpu_count() or 0,
        "mem_kb": mem_kb,
        "root": str(root),
        "root_ok": root.is_dir(),
        **{f"have_{tool}": shutil.which(tool) is not None for tool in tools},
    }


class SshTransport:
    """One router, over SSH, with a hard timeout on every command."""

    def __init__(
        self,
        spec: contracts.DeviceSpec,
        root: str = contracts.REMOTE_ROOT,
    ) -> None:
        self.spec = spec
        self.key = spec.key
        self.label = spec.label
        self.root = root
        self._askpass_dir: Path | None = None
        self._closed = False
        if spec.auth == "password" and not os.environ.get(PASSWORD_ENV):
            raise TransportError(
                f"{self.key} uses password authentication: export {PASSWORD_ENV} first "
                "(the value is in wavecom-devices.md; never commit it)"
            )

    # ── SSH plumbing ─────────────────────────────────────────────────────

    def _options(self) -> list[str]:
        options = [
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "LogLevel=ERROR",
        ]
        if self.spec.auth == "key":
            options += ["-o", "BatchMode=yes", "-o", "PreferredAuthentications=publickey"]
        else:
            options += ["-o", "NumberOfPasswordPrompts=1"]
        return options

    def _askpass(self) -> Path:
        """Create the askpass helper once, in a private directory."""
        if self._askpass_dir is not None:
            return self._askpass_dir / "askpass.sh"
        directory = Path(tempfile.mkdtemp(prefix="inmotion-askpass-"))
        directory.chmod(0o700)
        helper = directory / "askpass.sh"
        # The password itself is never written here: the helper prints the
        # environment variable, so the file carries no secret and the directory
        # is removed by `close()`.
        helper.write_text(
            "#!/bin/sh\n"
            "# inMotion device-2 askpass: reads the secret from the environment.\n"
            f"printf '%s\\n' \"${PASSWORD_ENV}\"\n",
            encoding="utf-8",
        )
        helper.chmod(0o700)
        self._askpass_dir = directory
        return helper

    def _env(self) -> dict[str, str] | None:
        if self.spec.auth != "password":
            return None
        env = dict(os.environ)
        env["SSH_ASKPASS"] = str(self._askpass())
        env["SSH_ASKPASS_REQUIRE"] = "force"
        env.setdefault("DISPLAY", ":0")
        return env

    def _argv(self, remote_cmd: str) -> list[str]:
        argv = ["ssh", *self._options()]
        if self.spec.jump:
            argv += ["-J", self.spec.jump]
        argv += ["-p", str(self.spec.port), f"{self.spec.user}@{self.spec.host}", remote_cmd]
        return argv

    def _run_bytes(
        self, remote_cmd: str, timeout: float, stdin: bytes | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        return _communicate(self._argv(remote_cmd), timeout=timeout, stdin=stdin, env=self._env())

    def run(
        self,
        cmd: str,
        timeout: float = DEFAULT_TIMEOUT_S,
        *,
        input: bytes | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        raw = self._run_bytes(cmd, timeout, input)
        result = subprocess.CompletedProcess(
            raw.args, raw.returncode, redact(_decode(raw.stdout)), redact(_decode(raw.stderr))
        )
        if check and result.returncode != 0:
            raise TransportError(
                f"{self.key}: command failed ({result.returncode}): "
                f"{result.stderr.strip()[:400] or result.stdout.strip()[:400]}"
            )
        return result

    # ── files ────────────────────────────────────────────────────────────

    def _remote_parent(self, remote: str | Path) -> str:
        return rdirname(str(remote).rstrip("/")) or "/"

    def upload(self, local: Path, remote: str | Path, *, replace: bool = True) -> None:
        """Ship a file or directory as a tar stream; tar exists on both devices."""
        source = Path(local)
        if not source.exists():
            raise TransportError(f"upload: {source} does not exist")
        try:
            packed = _communicate(
                ["tar", "-C", str(source.parent), "-cf", "-", source.name], timeout=120.0
            )
        except TransportTimeout as exc:
            raise TransportError(str(exc)) from None
        if packed.returncode != 0:
            raise TransportError(f"upload: tar failed: {_decode(packed.stderr).strip()[:300]}")
        remote_path = str(remote).rstrip("/")
        parent = self._remote_parent(remote_path)
        cleanup = f"rm -rf {shell_quote(remote_path)}" if replace else "true"
        script = (
            f"mkdir -p {shell_quote(parent)} && {cleanup} && tar -C {shell_quote(parent)} -xf -"
        )
        result = self._run_bytes(
            script,
            timeout=max(120.0, 60.0 + source.stat().st_size / 1e5),
            stdin=packed.stdout,
        )
        if result.returncode != 0:
            raise TransportError(
                f"upload to {self.key}:{remote_path} failed: {_decode(result.stderr).strip()[:300]}"
            )

    def download(self, remote: str | Path, local: Path) -> Path:
        """Fetch a file or directory as a tar stream, extracting under ``local.parent``."""
        remote_path = str(remote).rstrip("/")
        parent = self._remote_parent(remote_path)
        name = remote_path.rsplit("/", 1)[-1]
        result = self._run_bytes(
            f"tar -C {shell_quote(parent)} -cf - {shell_quote(name)}",
            timeout=300.0,
        )
        if result.returncode != 0:
            raise TransportError(
                f"download from {self.key}:{remote_path} failed: "
                f"{_decode(result.stderr).strip()[:300]}"
            )
        target = Path(local)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
            archive.extractall(target.parent, filter="data")
        return target

    def read_text(self, remote: str | Path) -> str:
        result = self.run(f"cat {shell_quote(remote)}", timeout=60.0)
        if result.returncode != 0:
            raise TransportError(f"{self.key}: cannot read {remote}: {result.stderr.strip()[:300]}")
        return result.stdout

    def write_text(self, remote: str | Path, text: str) -> None:
        parent = self._remote_parent(remote)
        result = self.run(
            f"mkdir -p {shell_quote(parent)} && cat > {shell_quote(remote)}",
            timeout=120.0,
            input=text.encode(),
        )
        if result.returncode != 0:
            raise TransportError(
                f"{self.key}: cannot write {remote}: {result.stderr.strip()[:300]}"
            )

    def mkdir(self, remote: str | Path) -> None:
        self.run(f"mkdir -p {shell_quote(remote)}", timeout=60.0, check=True)

    def exists(self, remote: str | Path) -> bool:
        return self.run(f"test -e {shell_quote(remote)}", timeout=60.0).returncode == 0

    def sha256(self, remote: str | Path) -> str:
        """SHA-256 of one file on the device, or ``""`` when sha256sum is absent."""
        result = self.run(f"sha256sum {shell_quote(remote)}", timeout=120.0)
        if result.returncode != 0:
            return ""
        parts = result.stdout.split()
        return parts[0] if parts else ""

    # ── inventory ────────────────────────────────────────────────────────

    def preflight(self) -> dict[str, object]:
        """One round trip: kernel, CPU, memory, tools, the deployed binary.

        Single line on purpose: the devices' login shell ``eval``s the command,
        so a newline would be flattened into a space and break ``if ... then``.
        """
        root = shell_quote(self.root)
        statements = [
            'echo "os=$(uname -s 2>/dev/null || echo unknown)"',
            'echo "kernel=$(uname -r 2>/dev/null || echo unknown)"',
            'echo "machine=$(uname -m 2>/dev/null || echo unknown)"',
            'echo "euid=$(id -u 2>/dev/null || echo -1)"',
            "echo \"cpu=$(sed -n 's/^model name[[:space:]]*: //p' "
            '/proc/cpuinfo 2>/dev/null | head -1)"',
            "echo \"cores=$(grep -c '^processor' /proc/cpuinfo 2>/dev/null || echo -1)\"",
            "echo \"mem_kb=$(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null)\"",
            f'if [ -d {root} ]; then echo "root_ok=True"; else echo "root_ok=False"; fi',
            "for tool in taskset gzip tar curl nc nproc getconf stat gcc sha256sum; do "
            'if command -v "$tool" >/dev/null 2>&1; then echo "have_$tool=True"; '
            'else echo "have_$tool=False"; fi; done',
            f'echo "binary=$(test -x {root}/bin/inmotion-infer && echo True || echo False)"',
        ]
        result = self.run("; ".join(statements), timeout=90.0)
        facts: dict[str, object] = {"transport": "ssh", "host": self.spec.host, "root": self.root}
        for line in result.stdout.splitlines():
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip()
            if value in ("True", "False"):
                facts[key] = value == "True"
            else:
                try:
                    facts[key] = int(value)
                except ValueError:
                    facts[key] = value
        facts.setdefault("root_ok", False)
        facts["root_ok"] = bool(facts.get("root_ok"))
        facts["transport"] = "ssh"
        return facts

    def close(self) -> None:
        """Remove the askpass helper; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        if self._askpass_dir is not None and self._askpass_dir.exists():
            shutil.rmtree(self._askpass_dir, ignore_errors=True)
            self._askpass_dir = None

    def __enter__(self) -> SshTransport:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Interpreter shutdown can already have torn down modules; keep it quiet.
        with contextlib.suppress(Exception):
            self.close()


def open_device(key: str, root: str | Path | None = None) -> DeviceTransport:
    """Open the transport for ``local`` or either router."""
    if key == "local":
        return LocalTransport(root=str(root or Path(".inmotion-edge/local")))
    if key not in contracts.DEVICES:
        raise KeyError(f"unknown device {key!r}; known: local, {', '.join(contracts.DEVICES)}")
    return SshTransport(contracts.DEVICES[key], root=str(root or contracts.REMOTE_ROOT))


def collect_artifact_sha256(artifact_dir: Path) -> str:
    """SHA-256 of ``<artifact>/model.safetensors``, the campaign's join key."""
    target = Path(artifact_dir) / contracts.ARTIFACT_HASH_FILE
    if not target.is_file():
        return ""
    return _hash_file(target)


def binary_sha256(path: Path | str) -> str:
    """SHA-256 of the deployed runtime binary."""
    return _hash_file(Path(path))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
