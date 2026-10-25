# Require Python 3.11 for Pillow 13 security fixes

Date: 2026-10-25

Decision: Eric Fitz approved raising renamer's minimum Python version from
3.10 to 3.11 and requiring Pillow 13.0.0 or newer. In this session, after
being shown these changes and the loss of Python 3.10 support, the user
explicitly instructed: "do it".

Rationale: The lockfile audit reported 37 advisories affecting Pillow 12.3.0
with fixes in Pillow 13.0.0. Pillow 13 drops Python 3.10 support. Keeping
the prior runtime minimum would prevent resolving those fixes for every
supported Python version.

Consequences: Running the application now requires Python 3.11 or newer.
Standalone generated apply/revert scripts retain their Python 3.10 minimum:
they do not depend on Pillow or the application environment.

Reference: https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html
