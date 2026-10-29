"""Small POSIX regression: preserve a real venv Python symlink and its prefix."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import venv

from campaign_runner import command


def main():
    if os.name != "posix":
        print(json.dumps(dict(status="NOT_CHECKED", reason="requires POSIX Python symlinks")))
        return 77
    with tempfile.TemporaryDirectory(prefix="campaign-venv-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
        python = environment / "bin/python"
        assert python.is_symlink(), "Test requires an actual interpreter symlink"
        argv = command([str(python), "-c", "import sys; print(sys.prefix)"], root)
        assert argv[0] == str(python), "Validation must preserve the requested entrypoint"
        result = subprocess.run(argv, text=True, capture_output=True, timeout=10, check=True)
        assert Path(result.stdout.strip()).resolve() == environment.resolve(), "Child lost its venv"
    print(json.dumps(dict(status="PASS", actual_python_symlink_preserved=True,
                          child_venv_prefix_preserved=True, dependencies_installed=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
