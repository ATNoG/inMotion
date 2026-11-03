"""Transports: local execution, timeouts, redaction, SSH argv and askpass."""

from __future__ import annotations

import hashlib
import os
import shlex
import stat
import time
from pathlib import Path

import pytest

from inmotion.edge import contracts, transport


@pytest.fixture
def local(tmp_path: Path) -> transport.LocalTransport:
    return transport.LocalTransport(root=str(tmp_path / "device"))


def test_local_run_captures_and_reports_nonsense(local: transport.LocalTransport) -> None:
    result = local.run("echo hello; echo oops >&2", timeout=10)
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"
    assert result.stderr.strip() == "oops"

    failed = local.run("exit 7", timeout=10)
    assert failed.returncode == 7
    with pytest.raises(transport.TransportError, match=r"failed \(7\)"):
        local.run("exit 7", timeout=10, check=True)


def test_local_run_pipes_stdin(local: transport.LocalTransport) -> None:
    target = Path(local.root) / "piped.txt"
    result = local.run(f"cat > {target}", timeout=10, input=b"payload")
    assert result.returncode == 0
    assert target.read_text() == "payload"


def test_local_run_timeout_kills_the_process_group(local: transport.LocalTransport) -> None:
    started = time.monotonic()
    with pytest.raises(transport.TransportTimeout, match="timed out"):
        local.run("sleep 30", timeout=0.3)
    assert time.monotonic() - started < 5.0


def test_redaction_uses_the_environment_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(transport.PASSWORD_ENV, "hunter2")
    assert transport.redact("prefix hunter2 suffix") == "prefix *** suffix"
    assert transport.redact("prefix hunter2", secrets=("hunter2",)) == "prefix ***"
    monkeypatch.delenv(transport.PASSWORD_ENV)
    assert transport.redact("hunter2") == "hunter2"


def test_local_upload_download_and_read(local: transport.LocalTransport, tmp_path: Path) -> None:
    source = tmp_path / "artifact"
    (source / "nested").mkdir(parents=True)
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "nested" / "x.txt").write_text("x")

    remote = Path(local.root) / "artifacts" / "lejepa-fp32"
    local.upload(source, remote)
    assert (remote / "model.safetensors").read_bytes() == b"weights"
    assert local.read_text(remote / "nested" / "x.txt") == "x"
    assert local.exists(remote / "model.safetensors")
    assert not local.exists(remote / "missing")

    fetched = tmp_path / "fetched"
    local.download(remote, fetched)
    assert (fetched / "model.safetensors").read_bytes() == b"weights"

    # replace=True clears the old tree rather than merging into it.
    local.upload(source, remote)
    (remote / "stale.txt").write_text("stale")
    local.upload(source, remote)
    assert not (remote / "stale.txt").exists()


def test_local_write_text_and_mkdir(local: transport.LocalTransport) -> None:
    path = Path(local.root) / "scripts" / "w.sh"
    local.write_text(path, "#!/bin/sh\necho hi\n")
    assert local.read_text(path).startswith("#!/bin/sh")
    local.mkdir(Path(local.root) / "empty")
    assert (Path(local.root) / "empty").is_dir()


def test_local_preflight_has_the_device_shape(local: transport.LocalTransport) -> None:
    facts = local.preflight()
    assert facts["transport"] == "local"
    assert facts["root_ok"] is True
    assert isinstance(facts["cores"], int)
    assert facts["cores"] >= 1
    assert isinstance(facts["mem_kb"], int)
    assert facts["mem_kb"] > 0
    assert "have_tar" in facts


def test_artifact_sha256_is_the_weights_file_only(tmp_path: Path) -> None:
    artifact = tmp_path / "lejepa-fp32"
    artifact.mkdir()
    weights = artifact / contracts.ARTIFACT_HASH_FILE
    weights.write_bytes(b"tensors")
    (artifact / "meta.json").write_text("{}")
    expected = hashlib.sha256(b"tensors").hexdigest()
    assert transport.collect_artifact_sha256(artifact) == expected
    # A missing weights file means "no join key", never a directory hash.
    assert transport.collect_artifact_sha256(tmp_path / "lejepa-q4") == ""


def test_binary_sha256(tmp_path: Path) -> None:
    binary = tmp_path / "inmotion-infer"
    binary.write_bytes(b"\x7fELF")
    assert transport.binary_sha256(binary) == hashlib.sha256(b"\x7fELF").hexdigest()


def test_open_device_local_and_unknown() -> None:
    node = transport.open_device("local", root="/tmp/edge-test-device")
    assert node.key == "local"
    assert Path(node.root).is_dir()
    with pytest.raises(KeyError, match="unknown device"):
        transport.open_device("wave-router-9")


def test_device_one_uses_key_auth_and_the_contract_endpoint() -> None:
    node = transport.SshTransport(contracts.DEVICES["wave-router"])
    argv = node._argv("echo hi")
    assert argv[0] == "ssh"
    assert "BatchMode=yes" in argv
    assert "admin@192.168.93.13" in argv
    assert "-p" in argv
    assert argv[argv.index("-p") + 1] == "56022"
    assert argv[-1] == "echo hi"
    assert node._env() is None
    assert "-J" not in argv


def test_device_two_uses_the_jump_host_and_askpass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(transport.PASSWORD_ENV, "s3cret-value")
    node = transport.SshTransport(contracts.DEVICES["wave-router-2"])
    argv = node._argv("echo hi")
    assert "-J" in argv
    assert argv[argv.index("-J") + 1] == "admin@192.168.93.13:56022"
    assert "admin@10.33.0.113" in argv
    assert "NumberOfPasswordPrompts=1" in argv

    helper = node._askpass()
    assert helper.is_file()
    assert stat.S_IMODE(helper.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(helper.stat().st_mode) == 0o700
    text = helper.read_text()
    assert transport.PASSWORD_ENV in text
    assert "s3cret-value" not in text

    env = node._env()
    assert env is not None
    assert env["SSH_ASKPASS"] == str(helper)
    assert env["SSH_ASKPASS_REQUIRE"] == "force"
    assert env["DISPLAY"] == ":0"

    directory = helper.parent
    node.close()
    assert not directory.exists()


def test_device_two_refuses_without_the_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(transport.PASSWORD_ENV, raising=False)
    with pytest.raises(transport.TransportError, match=transport.PASSWORD_ENV):
        transport.SshTransport(contracts.DEVICES["wave-router-2"])


def test_askpass_helper_prints_the_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(transport.PASSWORD_ENV, "p@ss word")
    node = transport.SshTransport(contracts.DEVICES["wave-router-2"])
    helper = node._askpass()
    import subprocess

    output = subprocess.run(
        [str(helper)],
        capture_output=True,
        text=True,
        env={**os.environ, transport.PASSWORD_ENV: "p@ss word"},
        check=True,
    ).stdout
    assert output.strip() == "p@ss word"
    node.close()


def test_shell_quote_handles_spaces_and_quotes() -> None:
    assert transport.shell_quote("a b") == "'a b'"
    assert transport.shell_quote("it's") == shlex.quote("it's")
    assert transport.shell_quote(Path("/tmp/x y/z")) == "'/tmp/x y/z'"
