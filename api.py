#!/usr/bin/env python3
"""Root API entrypoint for thesis section 4.2 examples.

Run with:
    ./venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000
"""

from src.api_server import app
