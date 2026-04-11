#!/usr/bin/env python3
"""Root Streamlit entrypoint that forwards to the UI module in src."""

import runpy


# Streamlit reruns this script often; run_module ensures the UI code is executed
# on each rerun instead of being skipped due to Python import caching.
runpy.run_module("src.frontend_ui", run_name="__main__")
