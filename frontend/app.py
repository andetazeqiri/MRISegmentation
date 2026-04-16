#!/usr/bin/env python3
"""Frontend entrypoint that mirrors the full thesis UI in src/frontend_ui.py."""

import runpy


runpy.run_module("src.frontend_ui", run_name="__main__")
