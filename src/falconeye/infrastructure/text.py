"""
Utilities for reading source files that may not be UTF-8.

Real-world codebases are rarely uniformly UTF-8:
- .NET / Windows projects default to CP1252 (aka Windows-1252) unless
  the developer explicitly changes it.
- Older Java, C, and PHP files are often Latin-1.
- Some editors emit a UTF-8 BOM.
- Japanese source may be Shift-JIS, Chinese may be GB2312, etc.

The strict `Path.read_text(encoding="utf-8")` call chain in the indexer
and analyzer raises `UnicodeDecodeError` on any of these, which kills
the file (and, in the concurrent case, one worker slot until the next
scheduled coroutine takes over).

The helper here is a small, ordered fallback:
  1. UTF-8 strict            -- fast path, catches 95%+ of files
  2. UTF-8 with BOM stripped -- catches VS-saved-as-UTF-8 files
  3. CP1252                  -- catches the vast majority of Western
                                non-UTF-8 files. Single-byte encoding,
                                so it never raises on 8-bit input.
  4. UTF-8 with replacement  -- absolute last resort; only reachable if
                                the CP1252 step itself is disabled.

Content stays as accurate as possible: the string returned reflects the
original characters (not their bytes). The LLM sees the same text a
human developer would see when opening the file in the editor it was
authored with.
"""

from pathlib import Path


def read_source_file(path: Path) -> str:
    """
    Read a source file as text, tolerating non-UTF-8 encodings.

    Never raises `UnicodeDecodeError`. If all defined encodings fail
    to fully decode, falls back to UTF-8 with error='replace' so the
    caller always gets a usable string.
    """
    raw = Path(path).read_bytes()

    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    # Absolute fallback -- never raises. Undecodable bytes become U+FFFD.
    return raw.decode("utf-8", errors="replace")


def read_source_file_with_encoding(path: Path) -> tuple[str, str]:
    """
    Same as read_source_file() but also returns the encoding used.

    Useful for logging when the LLM sees garbled comments and you want
    to know why. Not used on the hot path.
    """
    raw = Path(path).read_bytes()

    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="replace"), "utf-8+replace"
