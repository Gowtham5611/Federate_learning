"""FedSecureAI Backend Package Initialization.

Ensures the backend directory is in sys.path so domain, infrastructure,
application, and interfaces subpackages are importable from any entrypoint.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

ROOT_DIR = BACKEND_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

