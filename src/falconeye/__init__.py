"""
FalconEYE - AI-powered security code review tool.

This version uses Hexagonal Architecture for better modularity and maintainability.
All security analysis is performed purely through AI with no pattern matching.
"""

# The one runtime version string: the banner, --help, `info`, SARIF output
# and generated configs all read it, and `falconeye upgrade` parses this
# line from the file. Keep it equal to pyproject.toml's version
# (tests/unit/test_version.py fails otherwise).
__version__ = "2.1.0"