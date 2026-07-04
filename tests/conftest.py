"""Pytest session setup.

Force Hugging Face and Transformers into offline mode BEFORE any test imports a module
that could reach the network. `.claude/verify.sh` runs ``pytest -q`` on every session
stop; without this tripwire a single accidental ``from_pretrained`` anywhere in an
imported module's graph would try to download multi-GB weights inside the gate. Offline
mode turns any such attempt into an immediate, clean error instead.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
