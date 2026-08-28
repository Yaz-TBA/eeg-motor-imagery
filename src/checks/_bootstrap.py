"""Put `src/` on the import path so `from common import ...` works when a script in
this folder is run directly.

Every module here imports this first, before anything that reaches for common.py.
It exists so that no module has to import a *data* module for its path side effect,
which is what the code used to do and which reads as a workaround rather than a
decision. Nothing but path setup belongs in this file.
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
