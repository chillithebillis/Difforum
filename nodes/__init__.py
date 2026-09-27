"""Difforum v1 nodes."""

from __future__ import annotations

from . import bridges, curves, direction, export, post, render, setup

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

for _m in (setup, direction, curves, render, bridges, export, post):
    NODE_CLASS_MAPPINGS.update(_m.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_m.NODE_DISPLAY_NAME_MAPPINGS)
