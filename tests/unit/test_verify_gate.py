"""The verify_repo.sh gate must actually fail when a step fails.

Regression test for the `set -e` AND-list blind spot: every step of the gate
used to be written as `cmd && ok "..."`. When the left side of a top-level
AND-list fails, bash's -e abort is exempt, so a gate whose python (or ruff,
or pytest) could not run still printed the final success banner. The fix runs
each command bare; this test fault-injects a PATH whose `python3` always
exits 1 and asserts the gate now exits non-zero without the banner.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "scripts" / "verify_repo.sh"
SUCCESS_BANNER = "repository verification passed"


def _stub_path(tmp_path: Path) -> str:
    """PATH with a failing `python3` and a no-op `ruff` shadowing the real ones.

    Real coreutils (dirname, git, ...) stay reachable via the original PATH
    appended after the stub dir.
    """
    stub = tmp_path / "stub-bin"
    stub.mkdir()
    for name, body in (("python3", "#!/bin/sh\nexit 1\n"), ("ruff", "#!/bin/sh\nexit 0\n")):
        exe = stub / name
        exe.write_text(body)
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return f"{stub}{os.pathsep}{os.environ.get('PATH', '')}"


class TestVerifyGateAbortsOnFailure:
    def test_failing_python3_aborts_the_gate(self, tmp_path: Path):
        env = os.environ.copy()
        env["PATH"] = _stub_path(tmp_path)
        result = subprocess.run(
            ["bash", str(GATE)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        combined = result.stdout + result.stderr
        assert result.returncode != 0, (
            "gate must exit non-zero when a step's command fails; " f"output was:\n{combined}"
        )
        assert (
            SUCCESS_BANNER not in combined
        ), "gate must not print the success banner after a failed step"

    def test_gate_is_executable_and_present(self):
        assert GATE.is_file()
        assert (
            GATE.stat().st_mode & stat.S_IXUSR
        ), "gate must be runnable via ./scripts/verify_repo.sh"
