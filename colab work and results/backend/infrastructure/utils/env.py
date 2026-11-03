"""Environment file loading.

The project reads credentials from environment variables only (see
:mod:`infrastructure.data.downloaders.credentials`). This module is the bridge
that lets a developer keep those variables in a git-ignored ``.env`` file
instead of exporting them in every shell.

Loading happens once, at the composition root, and never during a test: the
suite must not depend on whatever happens to be in a local ``.env``.

**No value read from the file is ever logged.** Only the file's path and the
number of variables it defined are reported, so a log line can confirm that
configuration was picked up without disclosing what it contained.
"""

from __future__ import annotations

import logging
from pathlib import Path

from infrastructure.utils.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

ENV_FILENAME = ".env"
"""Conventional name of the local environment file. Git-ignored."""


def load_env_file(path: Path | None = None, *, override: bool = False) -> int:
    """Load ``.env`` into :data:`os.environ` and report how many keys it set.

    A missing file is not an error. The project is designed to run with no
    ``.env`` at all — every metadata operation and the whole test suite work
    without one, and only an explicit download needs the variables it holds.

    Args:
        path: File to read. Defaults to ``.env`` beside the project root.
        override: Whether a value in the file replaces one already exported in
            the environment. ``False`` by default, so an explicit shell export
            always wins over a stale file.

    Returns:
        The number of variables the file defined, or ``0`` if it was absent.
        The count is a count only; no name or value is returned.
    """
    target = path if path is not None else PROJECT_ROOT / ENV_FILENAME

    if not target.is_file():
        logger.debug("No %s file at %s; using the ambient environment only", ENV_FILENAME, target)
        return 0

    # Imported here rather than at module scope so that the absence of
    # python-dotenv degrades to "no .env support" instead of breaking startup.
    try:
        from dotenv import dotenv_values, load_dotenv
    except ImportError:
        logger.warning(
            "%s exists at %s but python-dotenv is not installed, so it was ignored; "
            "export the variables in the shell or install requirements/base.txt",
            ENV_FILENAME,
            target,
        )
        return 0

    defined = {key for key, value in dotenv_values(target).items() if value}
    load_dotenv(target, override=override)

    logger.info(
        "Loaded %d variable(s) from %s (values are never logged)", len(defined), target
    )
    return len(defined)
