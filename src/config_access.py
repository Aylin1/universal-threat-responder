"""Read optional settings from `config.cfg` with defaults.

config.py wasn't part of the review, so the new code never assumes the new fields exist.
Add them to config.py whenever convenient (see CHANGES.md); until then the defaults apply.
"""
from __future__ import annotations

from typing import Any

from config import cfg


def cfg_get(section: str, name: str, default: Any) -> Any:
    sec = getattr(cfg, section, None)
    if sec is None:
        return default
    return getattr(sec, name, default)
