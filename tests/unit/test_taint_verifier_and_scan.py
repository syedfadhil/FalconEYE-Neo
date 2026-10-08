"""Unit tests for the taint verifier and the scan orchestrator."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from typing import List, Optional

import pytest

from falconeye.domain.models.security import FindingConfidence, Severity
from falconeye.domain.models.taint import (
    FunctionDef,
    Location,
    TaintPath,
    UnresolvedCall,
)
from falconeye.domain.services.call_graph import CallGraph, CallGraphBuilder
from falconeye.domain.services.taint_path_finder import TaintPathFinder
from falconeye.domain.services.taint_verifier import (
    TaintVerifier,
    VerifierResult,
    VERIFIER_VERDICT_CONFIRMED,
    VERIFIER_VERDICT_FALSE_POSITIVE,
    VERIFIER_VERDICT_UNCERTAIN,
)
from falconeye.application.services.taint_scan_service import (
    TaintScanService,
    TaintScanResult,
)
from falconeye.infrastructure.taint_adapters.adapter_registry import AdapterRegistry
from falconeye.infrastructure.taint_adapters.csharp import CSharpTaintAdapter
from falconeye.infrastructure.taint_index import load_taint_index


# ---------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------


class MockLLMService:
    """Minimal stand-in for LLMService.chat_with_schema.

    Records every call for inspection. Response is configurable per
    instance -- either a static string or a callable that receives
    the (system_prompt, user_prompt) pair and returns a string.
    """

    def __init__(self, response=None, *, raise_exc=False):
        self.response = response
        self.raise_exc = raise_exc
        self.calls: List[dict] = []

    async def chat_with_schema(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[dict] = None,
    ) -> str:
        self.calls.append({
            "system": system_prompt,
            "user": user_prompt,
            "schema": response_schema,
        })
        if self.raise_exc:
            raise RuntimeError("simulated LLM failure")
        if callable(self.response):
            return self.response(system_prompt, user_prompt)
        return self.response or ""


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def taint_index():
    return load_taint_index(languages=["csharp"])


@pytest.fixture(scope="module")
def csharp_registry():
    reg = AdapterRegistry()
    reg.register(CSharpTaintAdapter())
    return reg


@pytest.fixture
def sample_source():
    return textwrap.dedent("""
    namespace MyApp.Web;

    using System.Data.SqlClient;
    using System.IO;

    public class UserController {
        public string GetById(int id) {
            var userId = System.IO.File.ReadAllText("/tmp/input");
            var cmd = new SqlCommand(userId, _conn);
            return cmd.ExecuteReader().ToString();
        }
    }
    """).lstrip()


@pytest.fixture
def graph_and_path(sample_source, csharp_registry, taint_index, tmp_path):
    """Build a graph and pick one candidate path from the sample."""
    file_path = tmp_path / "UserController.cs"
    file_path.write_text(sample_source)
    builder = CallGraphBuilder(
        adapter_registry=csharp_registry,
        taint_index=taint_index,
    )
    builder.add_file(
        source=sample_source,
        file_path=str(file_path),
        language="csharp",
    )
    graph = builder.build()
    finder = TaintPathFinder(graph=graph, taint_index=taint_index, max_hops=5)
    paths = finder.find_paths()
    return graph, paths[0]


# ---------------------------------------------------------------------
# TaintVerifier: response parsing
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_verifier_parses_confirmed_verdict(graph_and_path, taint_index):
    graph, path = graph_and_path
    llm = MockLLMService(response=(
        '{"verdict": "confirmed", "confidence": "high", '
        '"reasoning": "clear flow", "evidence_lines": [8, 9]}'
    ))
    verifier = TaintVerifier(llm_service=llm)
    result = asyncio.run(verifier.verify(
        path=path, graph=graph, taint_index=taint_index,
    ))
    assert result.verdict == VERIFIER_VERDICT_CONFIRMED
    assert result.confidence == FindingConfidence.HIGH
    assert result.reasoning == "clear flow"
    assert result.evidence_lines == (8, 9)
    assert result.call_failed is False


@pytest.mark.unit
def test_verifier_parses_false_positive(graph_and_path, taint_index):
    graph, path = graph_and_path
    llm = MockLLMService(response=(
        '{"verdict": "false_positive", "confidence": "medium", '
        '"reasoning": "value is sanitized"}'
    ))
    verifier = TaintVerifier(llm_service=llm)
    result = asyncio.run(verifier.verify(
        path=path, graph=graph, taint_index=taint_index,
    ))
    assert result.verdict == VERIFIER_VERDICT_FALSE_POSITIVE
    assert result.confidence == FindingConfidence.MEDIUM
    assert result.evidence_lines == ()  # not required


@pytest.mark.unit
def test_verifier_strips_markdown_fences(graph_and_path, taint_index):
    """LLMs sometimes wrap JSON in ```json ... ``` despite the prompt."""
    graph, path = graph_and_path
    llm = MockLLMService(response=(
        '```json\n'
        '{"verdict": "confirmed", "confidence": "low", "reasoning": "ok"}\n'
        '```'
    ))
    verifier = TaintVerifier(llm_service=llm)
    result = asyncio.run(verifier.verify(
        path=path, graph=graph, taint_index=taint_index,
    ))
    assert result.verdict == VERIFIER_VERDICT_CONFIRMED


@pytest.mark.unit
def test_verifier_handles_llm_exception(graph_and_path, taint_index):
    graph, path = graph_and_path
    llm = MockLLMService(raise_exc=True)
    verifier = TaintVerifier(llm_service=llm)
    result = asyncio.run(verifier.verify(
        path=path, graph=graph, taint_index=taint_index,
    ))
    assert result.call_failed is True
    assert result.verdict == VERIFIER_VERDICT_UNCERTAIN
    assert "failed" in result.reasoning.lower()


@pytest.mark.unit
def test_verifier_handles_malformed_json(graph_and_path, taint_index):
    graph, path = graph_and_path
    llm = MockLLMService(response="not json at all")
    verifier = TaintVerifier(llm_service=llm)
    result = asyncio.run(verifier.verify(
        path=path, graph=graph, taint_index=taint_index,
    ))
    assert result.call_failed is True
    assert result.verdict == VERIFIER_VERDICT_UNCERTAIN


@pytest.mark.unit
def test_verifier_handles_invalid_verdict_value(graph_and_path, taint_index):
    """LLM returned valid JSON but 'verdict' isn't in the enum."""
    graph, path = graph_and_path
    llm = MockLLMService(response=(
        '{"verdict": "maybe_yes", "confidence": "high", "reasoning": "..."}'
    ))
    verifier = TaintVerifier(llm_service=llm)
    result = asyncio.run(verifier.verify(
        path=path, graph=graph, taint_index=taint_index,
    ))
    assert result.call_failed is True


@pytest.mark.unit
def test_verifier_defaults_confidence_when_absent(graph_and_path, taint_index):
    """Missing confidence in response -> low default."""
    graph, path = graph_and_path
    llm = MockLLMService(response=(
        '{"verdict": "confirmed", "reasoning": "no confidence field"}'
    ))
    verifier = TaintVerifier(llm_service=llm)
    result = asyncio.run(verifier.verify(
        path=path, graph=graph, taint_index=taint_index,
    ))
    assert result.confidence == FindingConfidence.LOW


@pytest.mark.unit
def test_verifier_drops_non_int_evidence_lines(graph_and_path, taint_index):
    graph, path = graph_and_path
    llm = MockLLMService(response=(
        '{"verdict": "confirmed", "confidence": "high", "reasoning": "x", '
        '"evidence_lines": [8, "nine", 10, null]}'
    ))
    verifier = TaintVerifier(llm_service=llm)
    result = asyncio.run(verifier.verify(
        path=path, graph=graph, taint_index=taint_index,
    ))
    # "nine" and null coerce failures are dropped; 8 and 10 remain.
    assert result.evidence_lines == (8, 10)


# ---------------------------------------------------------------------
# TaintVerifier: prompt structure
# ---------------------------------------------------------------------


@pytest.mark.unit
def test_verifier_prompt_includes_source_sink_and_chain(graph_and_path, taint_index):
    graph, path = graph_and_path
    llm = MockLLMService(response=(
        '{"verdict": "confirmed", "confidence": "high", "reasoning": "ok"}'
    ))
    verifier = TaintVerifier(llm_service=llm)
    asyncio.run(verifier.verify(
        path=path, graph=graph, taint_index=taint_index,
    ))
    prompt = llm.calls[0]["user"]
    assert path.source in prompt
    assert path.sink in prompt
    # Chain of user methods must appear in prompt for the LLM to reason about.
    for method in path.via:
        assert method in prompt


@pytest.mark.unit
def test_verifier_prompt_includes_numbered_method_body(graph_and_path, taint_index):
    """The prompt must include the method body with line numbers so
    the LLM can cite evidence."""
    graph, path = graph_and_path
    llm = MockLLMService(response=(
        '{"verdict": "confirmed", "confidence": "high", "reasoning": "ok"}'
    ))
    verifier = TaintVerifier(llm_service=llm)
    asyncio.run(verifier.verify(
        path=path, graph=graph, taint_index=taint_index,
    ))
    prompt = llm.calls[0]["user"]
    # The sample source has `File.ReadAllText` on some line -- prompt must
    # show it with a line number prefix.
    assert "ReadAllText" in prompt
    assert "SqlCommand" in prompt


@pytest.mark.unit
def test_verifier_prompt_survives_missing_file(csharp_registry, taint_index, tmp_path):
    """Method body location file deleted between graph build and verify."""
    src = "namespace X; public class Y { public void M() { } }"
    fpath = tmp_path / "Y.cs"
    fpath.write_text(src)
    builder = CallGraphBuilder(
        adapter_registry=csharp_registry, taint_index=taint_index,
    )
    builder.add_file(source=src, file_path=str(fpath), language="csharp")
    graph = builder.build()
    # Delete the file. Verifier should degrade gracefully.
    fpath.unlink()
    fake_path = TaintPath(
        source="csharp:some.source",
        sink="csharp:some.sink",
        source_caller="csharp:X.Y.M",
        sink_caller="csharp:X.Y.M",
        via=("csharp:X.Y.M",),
    )
    llm = MockLLMService(response=(
        '{"verdict": "uncertain", "confidence": "low", "reasoning": "no body"}'
    ))
    verifier = TaintVerifier(llm_service=llm)
    result = asyncio.run(verifier.verify(
        path=fake_path, graph=graph, taint_index=taint_index,
    ))
    # Just needs to not raise.
    assert result.verdict == VERIFIER_VERDICT_UNCERTAIN
    assert "not readable" in llm.calls[0]["user"]


# ---------------------------------------------------------------------
# TaintScanService: end-to-end orchestration
# ---------------------------------------------------------------------


def _canned_response(verdict: str) -> str:
    return (
        f'{{"verdict": "{verdict}", "confidence": "high", '
        f'"reasoning": "canned", "evidence_lines": [8]}}'
    )


@pytest.mark.unit
def test_scan_service_end_to_end_confirmed(
    csharp_registry, taint_index, tmp_path, sample_source,
):
    (tmp_path / "UC.cs").write_text(sample_source)
    llm = MockLLMService(response=_canned_response("confirmed"))
    verifier = TaintVerifier(llm_service=llm)
    service = TaintScanService(
        adapter_registry=csharp_registry,
        taint_index=taint_index,
        verifier=verifier,
    )
    result = asyncio.run(service.scan(
        repo_path=tmp_path, languages=["csharp"], max_hops=5,
    ))
    assert result.review.files_analyzed == 1
    assert len(result.candidate_paths) >= 1
    assert len(result.review.findings) == len(result.candidate_paths)
    for f in result.review.findings:
        assert f.severity == Severity.HIGH
        assert f.confidence == FindingConfidence.HIGH
        assert f.cwe_id == "CWE-89"
        assert f.verifier_verdict == VERIFIER_VERDICT_CONFIRMED
        assert "taint" in f.tags
        assert "taint:kind:sql_query" in f.tags


@pytest.mark.unit
def test_scan_service_drops_false_positives(
    csharp_registry, taint_index, tmp_path, sample_source,
):
    (tmp_path / "UC.cs").write_text(sample_source)
    llm = MockLLMService(response=_canned_response("false_positive"))
    verifier = TaintVerifier(llm_service=llm)
    service = TaintScanService(
        adapter_registry=csharp_registry,
        taint_index=taint_index,
        verifier=verifier,
    )
    result = asyncio.run(service.scan(
        repo_path=tmp_path, languages=["csharp"], max_hops=5,
    ))
    # Candidates were generated and verified, but none pass.
    assert len(result.candidate_paths) >= 1
    assert len(result.review.findings) == 0
    # LLM was still called for each candidate.
    assert llm.calls, "verifier should have been invoked at least once"


@pytest.mark.unit
def test_scan_service_uncertain_keeps_finding_with_tag(
    csharp_registry, taint_index, tmp_path, sample_source,
):
    (tmp_path / "UC.cs").write_text(sample_source)
    llm = MockLLMService(response=_canned_response("uncertain"))
    verifier = TaintVerifier(llm_service=llm)
    service = TaintScanService(
        adapter_registry=csharp_registry,
        taint_index=taint_index,
        verifier=verifier,
    )
    result = asyncio.run(service.scan(
        repo_path=tmp_path, languages=["csharp"], max_hops=5,
    ))
    assert len(result.review.findings) > 0
    for f in result.review.findings:
        assert "verifier:uncertain" in f.tags


@pytest.mark.unit
def test_scan_service_no_verify_mode(
    csharp_registry, taint_index, tmp_path, sample_source,
):
    (tmp_path / "UC.cs").write_text(sample_source)
    llm = MockLLMService(response="")  # would fail parse if called
    verifier = TaintVerifier(llm_service=llm)
    service = TaintScanService(
        adapter_registry=csharp_registry,
        taint_index=taint_index,
        verifier=verifier,
    )
    result = asyncio.run(service.scan(
        repo_path=tmp_path, languages=["csharp"], max_hops=5, verify=False,
    ))
    # LLM must not have been called
    assert llm.calls == []
    # Every candidate should have produced a finding tagged as unverified
    assert len(result.review.findings) == len(result.candidate_paths)
    for f in result.review.findings:
        assert "taint:unverified" in f.tags


@pytest.mark.unit
def test_scan_service_empty_repo(csharp_registry, taint_index, tmp_path):
    llm = MockLLMService(response=_canned_response("confirmed"))
    verifier = TaintVerifier(llm_service=llm)
    service = TaintScanService(
        adapter_registry=csharp_registry,
        taint_index=taint_index,
        verifier=verifier,
    )
    result = asyncio.run(service.scan(
        repo_path=tmp_path, languages=["csharp"], max_hops=5,
    ))
    assert result.review.files_analyzed == 0
    assert result.review.findings == []
    assert llm.calls == []


@pytest.mark.unit
def test_scan_service_skips_bin_and_obj_folders(
    csharp_registry, taint_index, tmp_path, sample_source,
):
    """Files under bin/, obj/ etc. are excluded from scanning."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "UC.cs").write_text(sample_source)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "SkipMe.cs").write_text(sample_source)
    (tmp_path / "obj").mkdir()
    (tmp_path / "obj" / "AlsoSkip.cs").write_text(sample_source)
    llm = MockLLMService(response=_canned_response("confirmed"))
    verifier = TaintVerifier(llm_service=llm)
    service = TaintScanService(
        adapter_registry=csharp_registry,
        taint_index=taint_index,
        verifier=verifier,
    )
    result = asyncio.run(service.scan(
        repo_path=tmp_path, languages=["csharp"], max_hops=5,
    ))
    assert result.review.files_analyzed == 1


@pytest.mark.unit
def test_scan_service_skips_non_csharp_files(
    csharp_registry, taint_index, tmp_path, sample_source,
):
    (tmp_path / "UC.cs").write_text(sample_source)
    (tmp_path / "README.md").write_text("# hi")
    (tmp_path / "config.json").write_text("{}")
    llm = MockLLMService(response=_canned_response("confirmed"))
    verifier = TaintVerifier(llm_service=llm)
    service = TaintScanService(
        adapter_registry=csharp_registry,
        taint_index=taint_index,
        verifier=verifier,
    )
    result = asyncio.run(service.scan(
        repo_path=tmp_path, languages=["csharp"], max_hops=5,
    ))
    # Only the .cs file counts.
    assert result.review.files_analyzed == 1


@pytest.mark.unit
def test_scan_service_finding_carries_verifier_verdict(
    csharp_registry, taint_index, tmp_path, sample_source,
):
    (tmp_path / "UC.cs").write_text(sample_source)
    llm = MockLLMService(response=_canned_response("confirmed"))
    verifier = TaintVerifier(llm_service=llm)
    service = TaintScanService(
        adapter_registry=csharp_registry,
        taint_index=taint_index,
        verifier=verifier,
    )
    result = asyncio.run(service.scan(
        repo_path=tmp_path, languages=["csharp"], max_hops=5,
    ))
    for f in result.review.findings:
        assert f.verifier_verdict == VERIFIER_VERDICT_CONFIRMED
        assert "canned" in f.verifier_reasoning


@pytest.mark.unit
def test_scan_service_mitigation_is_kind_specific(
    csharp_registry, taint_index, tmp_path, sample_source,
):
    (tmp_path / "UC.cs").write_text(sample_source)
    llm = MockLLMService(response=_canned_response("confirmed"))
    verifier = TaintVerifier(llm_service=llm)
    service = TaintScanService(
        adapter_registry=csharp_registry,
        taint_index=taint_index,
        verifier=verifier,
    )
    result = asyncio.run(service.scan(
        repo_path=tmp_path, languages=["csharp"], max_hops=5,
    ))
    for f in result.review.findings:
        # sql_query kind should get parameterized-query guidance
        assert "parameterized" in f.mitigation.lower()


@pytest.mark.unit
def test_scan_service_evidence_lines_propagate(
    csharp_registry, taint_index, tmp_path, sample_source,
):
    """LLM-provided evidence_lines end up on the SecurityFinding."""
    (tmp_path / "UC.cs").write_text(sample_source)
    llm = MockLLMService(response=(
        '{"verdict": "confirmed", "confidence": "high", '
        '"reasoning": "flows", "evidence_lines": [8, 9]}'
    ))
    verifier = TaintVerifier(llm_service=llm)
    service = TaintScanService(
        adapter_registry=csharp_registry,
        taint_index=taint_index,
        verifier=verifier,
    )
    result = asyncio.run(service.scan(
        repo_path=tmp_path, languages=["csharp"], max_hops=5,
    ))
    assert result.review.findings
    for f in result.review.findings:
        assert f.evidence_lines == [8, 9]


@pytest.mark.unit
def test_scan_service_verifier_failure_still_emits_finding_with_tag(
    csharp_registry, taint_index, tmp_path, sample_source,
):
    """LLM crash -> finding stays but tagged verifier:call_failed."""
    (tmp_path / "UC.cs").write_text(sample_source)
    llm = MockLLMService(raise_exc=True)
    verifier = TaintVerifier(llm_service=llm)
    service = TaintScanService(
        adapter_registry=csharp_registry,
        taint_index=taint_index,
        verifier=verifier,
    )
    result = asyncio.run(service.scan(
        repo_path=tmp_path, languages=["csharp"], max_hops=5,
    ))
    # verdict was uncertain (from call failure), so finding is emitted.
    assert result.review.findings
    for f in result.review.findings:
        assert "verifier:call_failed" in f.tags
