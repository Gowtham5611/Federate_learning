"""Frontend source tree.

A package marker, not an importable application package: only ``backend/`` is
placed on ``sys.path``. It exists so that static tooling scanning the repository
resolves the dashboard as ``frontend.interfaces.dashboard`` rather than as a
second, colliding ``interfaces`` package.
"""
