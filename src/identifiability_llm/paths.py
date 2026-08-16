"""Repository paths shared by experiment entry points.

Generated data defaults to ``<repository>/data``. Set
``ATTENTION_IDENTIFIABILITY_DATA`` to place large artifacts elsewhere without
changing any configuration or code.
"""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get(
        "ATTENTION_IDENTIFIABILITY_DATA",
        str(REPOSITORY_ROOT / "data"),
    )
).expanduser().resolve()

