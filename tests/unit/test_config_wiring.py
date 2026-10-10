"""Unit tests for config settings that used to be ignored.

Coverage:
  * skills.max_discovery_hits caps discovery in `skill run` (CLI) and
    in run_all (service); _discovery_hit_cap falls back to 200
  * chunking.max_chunk_size caps adaptive chunk sizes, in the chunker
    and through _process_file
  * Taint `specs` / `framework_autodetect` (not used yet) still
    validate, so existing configs keep loading
"""

from __future__ import annotations

import asyncio
import json
import logging
import textwrap
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from falconeye.adapters.cli import commands_skill
from falconeye.adapters.cli.commands_skill import _discovery_hit_cap, skill_run_command
from falconeye.application.commands.index_codebase import (
    IndexCodebaseCommand,
    IndexCodebaseHandler,
)
from falconeye.application.services.skill_runner_service import SkillRunnerService
from falconeye.domain.models.codebase import Codebase
from falconeye.domain.services.checksum_service import ChecksumService
from falconeye.infrastructure.config.config_models import LanguageTaintConfig
from falconeye.infrastructure.skills.skill_loader import SkillLoader


_SKILL = textwrap.dedent("""\
    ---
    name: {name}
    description: d
    function: scan
    ---

    ## Step 1 -- Discover

    ```bash
    grep -rn -E "danger" --include="*.py" .
    ```
    """)


class _LLM:
    async def chat_with_schema(self, system_prompt, user_prompt, response_schema=None):
        return json.dumps({"findings": []})


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.py").write_text("".join(f"danger({i})\n" for i in range(10)), encoding="utf-8")
    return ws


@pytest.fixture
def skills_dir(tmp_path):
    d = tmp_path / "skills"
    d.mkdir()
    for name in ("one", "two"):
        (d / f"{name}.md").write_text(_SKILL.format(name=name), encoding="utf-8")
    return d


# ---------------------------------------------------------------
# skills.max_discovery_hits
# ---------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("config, expected", [
    (SimpleNamespace(skills=SimpleNamespace(max_discovery_hits=17)), 17),
    (SimpleNamespace(skills=None), 200),
    (None, 200),
])
def test_discovery_hit_cap(config, expected):
    assert _discovery_hit_cap(SimpleNamespace(config=config)) == expected


@pytest.mark.unit
def test_skill_run_honours_max_discovery_hits(workspace, skills_dir, monkeypatch):
    container = SimpleNamespace(
        llm_service=_LLM(), skill_loader=None, review_confirmation_service=None,
        config=SimpleNamespace(skills=SimpleNamespace(max_discovery_hits=3, max_llm_only_files=50)),
    )
    monkeypatch.setattr(
        commands_skill, "DIContainer",
        type("Stub", (), {"create": classmethod(lambda cls, **kw: container)}),
    )
    buf = StringIO()
    skill_run_command(
        skill_name="one", workspace=workspace, output_format="console",
        output_file=None, config_path=None, skills_dir_override=skills_dir,
        console=Console(file=buf, force_terminal=False, width=300),
    )
    assert "(3 discovery hits" in buf.getvalue()


@pytest.mark.unit
def test_run_all_passes_the_cap_to_every_skill(workspace, skills_dir):
    skills = SkillLoader(directory=skills_dir).load_all()
    _, batch = asyncio.run(SkillRunnerService(llm_service=_LLM()).run_all(
        skills=skills, workspace=workspace, max_total_hits=4,
    ))
    assert [o.summary.total_discovery_hits for o in batch.outcomes] == [4, 4]


# ---------------------------------------------------------------
# chunking.max_chunk_size
# ---------------------------------------------------------------


class _Recorder:
    def __init__(self):
        self.count = 0

    async def generate_embeddings_batch(self, texts):
        self.count += len(texts)
        return [[0.0] for _ in texts]

    def count_tokens(self, text):
        return len(text) // 4


class _Sink:
    def __init__(self):
        self.chunks = []

    async def store_chunks(self, chunks, collection="code"):
        self.chunks.extend(chunks)

    async def store_metadata(self, metadata):
        pass

    def save_file(self, metadata):
        pass

    def analyze_file(self, file_path, content):
        return object()


def _lines(chunks):
    return [c.metadata.end_line - c.metadata.start_line + 1 for c in chunks]


@pytest.mark.unit
def test_chunker_max_lines_caps_adaptive_size():
    handler = IndexCodebaseHandler.__new__(IndexCodebaseHandler)
    handler.llm_service, handler.logger = _Recorder(), logging.getLogger("t")
    content = "".join(f"l{i}\n" for i in range(50_000))
    assert max(_lines(handler._chunk_content(content, "f", "python", 50, 10))) == 200
    assert max(_lines(handler._chunk_content(content, "f", "python", 50, 10, max_lines=80))) == 80


@pytest.mark.unit
def test_process_file_passes_max_chunk_size(tmp_path):
    src = tmp_path / "big.py"
    src.write_text("".join(f"x{i} = {i}\n" for i in range(5000)), encoding="utf-8")
    sink = _Sink()
    handler = IndexCodebaseHandler(
        vector_store=sink, metadata_repo=sink, llm_service=_Recorder(),
        language_detector=None, ast_analyzer=sink, project_identifier=None,
        checksum_service=ChecksumService(), index_registry=sink,
    )
    handler.logger = logging.getLogger("t")
    command = IndexCodebaseCommand(codebase_path=tmp_path, chunk_size=50, chunk_overlap=10,
                                   max_chunk_size=60)
    codebase = Codebase.create(root_path=tmp_path, language="python", excluded_patterns=[])
    asyncio.run(handler._process_file(src, "python", command, codebase, "p"))
    # 5,000 lines -> adaptive target 100 lines, capped at 60
    assert max(_lines(sink.chunks)) == 60


# ---------------------------------------------------------------
# Taint settings that are documented as not used yet
# ---------------------------------------------------------------


@pytest.mark.unit
def test_unused_taint_settings_still_validate():
    cfg = LanguageTaintConfig(enabled=True, specs=["custom"], framework_autodetect=False)
    assert cfg.specs == ["custom"] and cfg.framework_autodetect is False
