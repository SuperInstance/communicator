#!/usr/bin/env python3
"""Guard test files against missing optional dependencies.

The suite's contract (README): under a bare interpreter, dependency-
needing files SKIP with a reason; they never FAIL on ImportError. New
lanes keep landing with new imports, so this tool exists — run it after
adding tests, it finds files that fail under a dependency-free python
and injects the module-level importorskip they need.

    python3 tools/guard_test_deps.py            # scan + report
    python3 tools/guard_test_deps.py --fix      # scan + inject guards
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNOWN = ("typesafe_sdk", "numpy", "miniaudio", "faster_whisper", "onnxruntime")


def probe(path: Path) -> str | None:
    """Return the missing module a test file fails on, if any."""
    r = subprocess.run([sys.executable, "-m", "pytest", str(path), "-x", "-q",
                        "--co"], capture_output=True, text=True, timeout=120)
    m = re.search(r"No module named '([a-z_]+)'", r.stdout + r.stderr)
    return m.group(1) if m else None


def guard(path: Path, dep: str) -> None:
    lines = path.read_text().split("\n")
    if any("importorskip" in l and dep in l for l in lines[:15]):
        return
    idx = next((i + 1 for i, l in enumerate(lines)
                if l.startswith("from __future__")), None)
    if idx is None:
        idx = next(i + 1 for i, l in enumerate(lines)
                   if l.strip() == '"""' and i > 0)
    lines.insert(idx, f'import pytest as _pytest  # noqa: E402\n'
                      f'_pytest.importorskip("{dep}")  # noqa: E402')
    path.write_text("\n".join(lines))


def main() -> None:
    fix = "--fix" in sys.argv
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        missing = probe(path)
        if missing in KNOWN:
            print(f"{path.name}: needs {missing}"
                  + ("  [guarded]" if fix else "  (run --fix)"))
            if fix:
                guard(path, missing)
        elif missing:
            print(f"{path.name}: UNKNOWN dep {missing} -- add to KNOWN")


if __name__ == "__main__":
    main()
