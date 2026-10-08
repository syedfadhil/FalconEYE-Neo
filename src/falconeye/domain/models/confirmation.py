"""
Domain models for review-finding taint confirmation.

These support the inversion of the classic taint pipeline: instead of
using specs to *find* vulnerabilities and having the LLM verify them,
the LLM finds candidate vulnerabilities (via ``falconeye review``) and
we use taint-style call-graph analysis to *confirm reachability* from
recognized entry points.

Design principle: confirmation is ADDITIVE signal, not subtractive.
A finding that we cannot confirm reachable is NOT dismissed -- it
retains its original confidence and stays in the report. All we're
doing is boosting confidence for findings whose enclosing method is
provably reachable from an HTTP entry point (or middleware, etc.).

Why: entry-point detection is not perfect. Dispatch via reflection,
DI, minimal APIs, or event handlers won't show up in the static call
graph. Marking a finding "not reachable" as a dismissal would drop
real vulnerabilities. Marking it "confirmed reachable" as a boost is
safe -- false positives in that direction just mean we boosted a
finding that was already going to be shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .taint import Location


# --------------------------------------------------------------------
# Confirmation status constants
# --------------------------------------------------------------------

#: The finding's enclosing method is reachable from at least one
#: detected entry point. Highest signal -- the vulnerable code IS
#: exercised by external input.
CONFIRMATION_STATUS_REACHABLE = "confirmed_reachable"

#: We identified an enclosing method for the finding but no path from
#: any detected entry point reaches it. Could be dead code, could be
#: reflection / DI / event dispatch that isn't call-graph-visible,
#: could be that the entry-point detector missed something. Treat as
#: "we don't know", not "safe".
CONFIRMATION_STATUS_NOT_CONFIRMED = "not_confirmed"

#: The finding's file_path + line_number couldn't be mapped to any
#: FunctionDef in the workspace. Usually means the finding is on a
#: top-level statement (not inside a method) or the file didn't parse.
CONFIRMATION_STATUS_NO_ENCLOSING_METHOD = "no_enclosing_method"

#: No entry points were detected at all in the workspace. Every
#: finding falls into this bucket; no reachability was checked. This
#: is a workspace-level condition, not per-finding.
CONFIRMATION_STATUS_NO_ENTRY_POINTS = "no_entry_points"

#: The confirmer wasn't run for this finding (missing language
#: support, no adapter for the file's language, etc). Distinguishes
#: "we tried and it was unreachable" from "we didn't try".
CONFIRMATION_STATUS_SKIPPED = "skipped"


# --------------------------------------------------------------------
# Entry point identification
# --------------------------------------------------------------------


#: Kinds of entry point we recognize. Kept as string constants (not an
#: Enum) for JSON-friendliness in reports and for easy extension by
#: future detectors.
ENTRY_POINT_KIND_HTTP = "http"
ENTRY_POINT_KIND_MIDDLEWARE = "middleware"
ENTRY_POINT_KIND_MAIN = "main"
ENTRY_POINT_KIND_OTHER = "other"


@dataclass(frozen=True)
class EntryPoint:
    """A method identified as an external code entry point.

    Entry points are the roots of the reachability graph. For a web
    application these are HTTP-facing controller actions, middleware
    Invoke methods, and route handlers; for a CLI or worker they're
    ``main``, message handlers, or scheduled-job methods.

    ``canonical_name`` MUST match the canonical name a
    :class:`FunctionDef` would have in the call graph, so the
    reachability confirmer can look up the node without additional
    resolution. The detector is responsible for producing this
    correctly.

    ``reason`` is a short human-readable explanation ("[HttpGet(...)]",
    "Middleware.InvokeAsync") shown in reports so operators can
    understand why the confirmer treated this method as a root.
    """
    canonical_name: str
    reason: str
    location: Location
    kind: str = ENTRY_POINT_KIND_OTHER


# --------------------------------------------------------------------
# Confirmation result
# --------------------------------------------------------------------


@dataclass(frozen=True)
class TaintConfirmation:
    """Reachability-confirmation result for a single finding.

    Attached to :class:`SecurityFinding.taint_confirmation` after the
    confirmer runs. See status constants above for interpretation.

    ``reachable_entry_points`` is the set of entry-point canonical
    names from which the finding's enclosing method is reachable.
    Empty for any status other than ``confirmed_reachable``.

    ``shortest_chain`` is the canonical-name path from the closest
    entry point to the finding's enclosing method, in order. Length
    of the chain minus 1 is the hop count (0 means the finding is
    directly inside an entry-point method).

    ``reason`` is a plain-English summary of the outcome for
    displaying in reports.
    """
    status: str
    enclosing_function: Optional[str] = None
    reachable_entry_points: Tuple[str, ...] = ()
    shortest_chain: Tuple[str, ...] = ()
    hop_count: Optional[int] = None
    reason: str = ""

    @property
    def is_reachable(self) -> bool:
        """Convenience: True only for confirmed_reachable status.

        Callers should prefer status-based dispatch when they need to
        distinguish "not reachable" from "we couldn't check".
        """
        return self.status == CONFIRMATION_STATUS_REACHABLE
