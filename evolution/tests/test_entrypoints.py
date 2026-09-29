"""Public experiment entry points work independently of the caller's cwd."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLIS = [
    "evolution/benchmark/adapter.py",
    *[
        f"evolution/remote/{name}.py"
        for name in (
            "kcoral_remote",
            "kcoral_ncu",
            "kcoral_iket",
            "kcoral_python",
            "kcoral_compute_sanitizer",
        )
    ],
]


@pytest.mark.parametrize("entry", CLIS)
@pytest.mark.parametrize("arguments,expected_code", [(["--help"], 0), (["--not-an-option"], 2)])
def test_cli_from_outside_checkout(tmp_path, entry, arguments, expected_code):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / entry), *arguments],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected_code, result.stderr
    assert "usage:" in (result.stdout if expected_code == 0 else result.stderr)
    assert "Traceback" not in result.stderr


def test_remote_payload_sources_run_without_repository(tmp_path):
    from evolution.remote import kcoral_remote, kcoral_tool

    # Uploaded workers execute on a server without the local repository.
    for name, source in (
        ("diagnostics", kcoral_tool.WORKER_PATH.read_text()),
        ("benchmark", kcoral_remote.DRIVER_PATH.read_text()),
    ):
        script = tmp_path / f"{name}.py"
        script.write_text(source)
        subprocess.run([sys.executable, "-I", str(script)], cwd=tmp_path, check=True)


@pytest.mark.parametrize("form", ["key", "relative", "absolute", "cwd"])
def test_candidate_paths_resolve_to_registered_workload(tmp_path, monkeypatch, form):
    from evolution.benchmark import adapter

    root = tmp_path / "candidates"
    workload = root / "gdn" / "decode"
    workload.mkdir(parents=True)
    monkeypatch.setattr(adapter, "CANDIDATES_ROOT", root)
    monkeypatch.chdir(workload if form == "cwd" else tmp_path)
    argument = {
        "key": "gdn/decode",
        "relative": "candidates/gdn/decode",
        "absolute": str(workload),
        "cwd": ".",
    }[form]
    assert adapter.workload_key(argument) == "gdn/decode"
