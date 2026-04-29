"""API package — adds src/ to sys.path so intra-package imports resolve."""

import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
