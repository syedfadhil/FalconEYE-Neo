"""
Workspace file walking shared by skill discovery, the taint scan, and
reachability confirmation.

Each of those used to run its own ``root.rglob("*")`` loop and drop
files whose ABSOLUTE path contained a skip fragment such as
``/build/``. Two problems with that:

* A workspace that merely lives under a directory named ``build``,
  ``out``, ``bin``, ``target``, ``env`` ... (a Docker ``WORKDIR
  /build``, a CI output folder) had every file skipped, silently:
  skill discovery found nothing and confirmation built an empty call
  graph.
* ``rglob`` enumerates skipped trees in full before the filter
  discards them, so a frontend ``node_modules`` with 100k files was
  walked once per caller -- once per discovery pattern in the grep
  runner's case.

``iter_workspace_files`` matches skip fragments against the path
RELATIVE to the workspace root and prunes skipped directories before
``os.walk`` descends into them. Entries are visited in sorted order,
so results (and which hits survive a cap) are identical across runs
and operating systems. Symlinked directories are not followed.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Iterable, Iterator, Optional


def iter_workspace_files(
    root: Path,
    *,
    extensions: Iterable[str],
    skip_fragments: Iterable[str] = (),
    max_file_bytes: Optional[int] = None,
    name_patterns: Iterable[str] = (),
) -> Iterator[Path]:
    """Yield files under ``root`` whose suffix is in ``extensions``.

    ``extensions`` are compared case-insensitively, with or without a
    leading dot. ``name_patterns`` are shell-style globs matched
    case-insensitively against the file NAME (``package.json``,
    ``*.csproj``, ``requirements*.txt``); a file is yielded when it
    matches either an extension or a name pattern, so manifests with
    no useful extension can be selected. ``skip_fragments`` use the
    existing ``"/name/"`` convention (multi-segment fragments like
    ``"/a/b/"`` work too) and are matched against each directory's
    path relative to ``root``, so the location of the workspace itself
    never causes a skip. Files larger than ``max_file_bytes`` (when
    given) are skipped.
    """
    root = Path(root)
    exts = {
        e if e.startswith(".") else f".{e}"
        for e in (x.strip().lower() for x in extensions)
        if e
    }
    globs = tuple(p.strip().lower() for p in name_patterns if p.strip())
    if (not exts and not globs) or not root.is_dir():
        return
    fragments = tuple(skip_fragments)

    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root).as_posix()
        base = "/" if rel == "." else f"/{rel}/"
        # Prune in place: os.walk never descends into a removed entry.
        dirnames[:] = sorted(
            d for d in dirnames
            if not any(frag in f"{base}{d}/" for frag in fragments)
        )
        for name in sorted(filenames):
            lowered = name.lower()
            if os.path.splitext(lowered)[1] not in exts and not any(
                fnmatch.fnmatchcase(lowered, g) for g in globs
            ):
                continue
            path = Path(dirpath) / name
            if max_file_bytes is not None:
                try:
                    if path.stat().st_size > max_file_bytes:
                        continue
                except OSError:
                    continue
            yield path
