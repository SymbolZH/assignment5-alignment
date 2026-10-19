"""Report the pinned upstream image's known OpenCV declaration conflict.

The release image includes OpenCV 5 with NumPy 1.26.4, which Triton Ascend
requires. Preserve that official runtime rather than patching its packages.
Every other dependency conflict is an error. The full diagnostic is retained.
"""

import os
from pathlib import Path
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    env={**os.environ, "PYTHONPATH": ""}, text=True, capture_output=True,
)
diagnostic = result.stdout + result.stderr
Path("/opt/assignment5/pip-check.txt").write_text(diagnostic)
print(diagnostic, end="")
known = (
    'opencv-python-headless 5.0.0.93 has requirement numpy>=2; '
    'python_version >= "3.9", but you have numpy 1.26.4.'
)
if result.returncode:
    lines = [line.strip() for line in diagnostic.splitlines() if line.strip()]
    if lines != [known]:
        raise SystemExit(result.returncode)
    print("Known upstream image declaration conflict; native framework packages are unmodified.")
