"""Single source of truth for the platform version.

Consumed by pyproject.toml (dynamic version), the /healthz endpoint,
user-agent headers, and the release workflow. Bump via the release process
described in docs/operations/runbook.md — never edit ad hoc.
"""

from __future__ import annotations

__version__ = "1.0.0"

#: Calendar date of this release (ISO 8601).
RELEASE_DATE = "2026-10-18"

#: Semantic-version tuple for programmatic comparisons.
VERSION_TUPLE = (1, 0, 0)

#: User-agent fragment reported by HTTP clients of this platform.
USER_AGENT = f"multimodal-agent/{__version__} " "(+https://github.com/AdilShamim8/Multi-Modal-105)"

__all__ = ["__version__", "RELEASE_DATE", "VERSION_TUPLE", "USER_AGENT"]
