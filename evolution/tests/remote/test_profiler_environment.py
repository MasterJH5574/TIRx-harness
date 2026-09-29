"""Exercise profiler application launches with real subprocesses, without a GPU."""

import json
import os
import sys

import pytest

from evolution.remote import kcoral_tool as tool
from evolution.remote import kcoral_tool_worker as worker


@pytest.mark.parametrize("profiler", ["iket", "ncu"])
@pytest.mark.parametrize("command", ["python", "python3", "explicit"])
@pytest.mark.parametrize("mode", ["-c", "-m"])
def test_profiler_launch_uses_worker_python_despite_path(
    tmp_path, monkeypatch, profiler, command, mode
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    # Any accidental PATH lookup of the application interpreter must fail.
    for name in ["python", "python3"]:
        executable = fake_bin / name
        executable.write_text("#!/bin/sh\nexit 97\n")
        executable.chmod(0o755)
    # Stand in for the profiler, but actually launch the supplied application.
    launcher = fake_bin / ("run-iket" if profiler == "iket" else "ncu")
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, subprocess, sys\n"
        "args = sys.argv[1:]\n"
        "if '--output-dir' in args:\n"
        "    output = pathlib.Path(args[1]); output.mkdir(); output = output / 'probe.json'\n"
        "    app = args[args.index('--') + 1:]\n"
        "else:\n"
        "    output = pathlib.Path(args[3]); app = args[4:]\n"
        "result = subprocess.run(app, capture_output=True)\n"
        "output.write_bytes(result.stdout)\n"
        "sys.stderr.buffer.write(result.stderr)\n"
        "sys.exit(result.returncode)\n"
    )
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ["PATH"])
    monkeypatch.delenv("TVM_IKET_OFFICIAL_PROFILE", raising=False)
    source = (
        "import json, os, sys; print(json.dumps(dict(executable=sys.executable, "
        'argv=sys.argv[1:], profile=os.environ.get("TVM_IKET_OFFICIAL_PROFILE"), '
        'value=os.environ["PROBE_VALUE"])))'
    )
    module = tmp_path / "probe.py"
    module.write_text(source)
    args = ["--", "a b", "literal; $HOME", "python"]
    explicit_python = tmp_path / "explicit-python"
    explicit_python.symlink_to(sys.executable)
    app = [
        str(explicit_python) if command == "explicit" else command,
        mode,
        source if mode == "-c" else "probe",
        *args,
    ]
    archive = tool.pack_inputs([module])
    overrides = {"PROBE_VALUE": "a b; $HOME"}
    if profiler == "iket":
        result = worker.iket(archive, ["profile", "--", *app], overrides)
        payload = result["files"]["probe.json"]
    else:
        result = worker.ncu(archive, [], app, overrides)
        payload = result["report"]
    assert result["returncode"] == 0
    assert json.loads(payload) == dict(
        executable=str(explicit_python) if command == "explicit" else sys.executable,
        argv=args,
        profile="cutlass-4.6.0" if profiler == "iket" else None,
        value=overrides["PROBE_VALUE"],
    )
