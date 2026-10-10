# FalconEYE

```
███████╗ █████╗ ██╗      ██████╗ ██████╗ ███╗   ██╗███████╗██╗   ██╗███████╗
██╔════╝██╔══██╗██║     ██╔════╝██╔═══██╗████╗  ██║██╔════╝╚██╗ ██╔╝██╔════╝
█████╗  ███████║██║     ██║     ██║   ██║██╔██╗ ██║█████╗   ╚████╔╝ █████╗
██╔══╝  ██╔══██║██║     ██║     ██║   ██║██║╚██╗██║██╔══╝    ╚██╔╝  ██╔══╝
██║     ██║  ██║███████╗╚██████╗╚██████╔╝██║ ╚████║███████╗   ██║   ███████╗
╚═╝     ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚══════╝
```

**Next-Generation Security Code Analysis Powered by Local LLMs**

*FalconEYE-Neo – maintained by [syedfadhil](https://github.com/syedfadhil) · based on FalconEYE by [hardw00t](https://github.com/hardw00t) & [h4ckologic](https://github.com/h4ckologic)*

> [!NOTE]
> **About this project**
>
> - FalconEYE-Neo is an independent continuation of **FalconEYE**, originally developed by hardw00t & h4ckologic and published at
> `github.com/FalconEYE-ai/FalconEYE`. That repository is no longer publicly available (as of September 2026).
> - This project is **not affiliated with or endorsed by** the original authors. Please report issues with FalconEYE-Neo here, not to them.
> - Licensed under **AGPL-3.0-or-later**, like the original – see [License](#license).

FalconEYE represents a paradigm shift in static code analysis. Instead of relying on predefined vulnerability patterns, it leverages large language models to reason about your code the same way a security expert would — understanding context, intent, and subtle security implications that traditional tools miss.

---

## Table of Contents

- [Why FalconEYE?](#why-falconeye)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [MLX Backend (Apple Silicon)](#mlx-backend-apple-silicon)
- [LiteLLM Backend (OpenAI-compatible)](#litellm-backend-openai-compatible)
- [Supported Languages](#supported-languages)
- [CLI Reference](#cli-reference)
- [Taint Analysis & Reachability Confirmation](#taint-analysis--reachability-confirmation)
- [Audit Skills](#audit-skills)
- [Output Formats](#output-formats)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Development](#development)
- [FAQ](#faq)
- [Discoveries](#discoveries)
- [License](#license)

---

## Why FalconEYE?

Traditional security scanners are limited by their pattern databases. They can only find what they've been programmed to look for. FalconEYE is different:

- **No Pattern Matching** — Uses pure AI reasoning to understand your code semantically
- **Context-Aware Analysis** — Retrieval-Augmented Generation (RAG) provides relevant code context for deeper insights
- **Novel Vulnerability Detection** — Identifies security issues that don't match known patterns
- **LLM-Powered Enrichment** — Every finding gets AI-generated descriptions, mitigations, code snippets, and line numbers
- **Reduced False Positives** — Optional AI validation pass filters noise from false alarms
- **Rich Reporting** — Console, JSON, HTML, and SARIF output with interactive dashboards
- **Smart Re-indexing** — Incremental analysis means re-scans only process changed files
- **Reachability Confirmation** — Optional call-graph pass marks which findings are reachable from HTTP entry points (C# / ASP.NET Core today)
- **Audit Skills** — Run your own markdown audit playbooks (grep-guided or whole-file) against a codebase
- **Privacy-First** — Runs entirely locally with Ollama or MLX — your code never leaves your machine. With the LiteLLM backend, code goes to the endpoint you configure (e.g. an on-prem vLLM server).
- **Apple Silicon Accelerated** — Native MLX backend delivers 20-40% faster inference on M-series chips

---

## How It Works

FalconEYE follows a multi-stage analysis pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                     1. CODE INGESTION                          │
│  Scans repository -> Detects languages -> Parses AST structure │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    2. INTELLIGENT INDEXING                      │
│  Chunks code semantically -> Generates embeddings -> Stores in │
│  vector database for fast semantic search                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                   3. CONTEXT ASSEMBLY (RAG)                     │
│  For each code segment -> Retrieves similar code -> Gathers    │
│  relevant context from your entire codebase                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    4. AI SECURITY ANALYSIS                      │
│  LLM analyzes code with context -> Reasons about vulns ->      │
│  Understands data flow -> Identifies security implications     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                   5. LLM-POWERED ENRICHMENT                    │
│  Incomplete findings sent back to the LLM for detailed         │
│  reasoning, specific mitigations, code snippets & line numbers │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     6. VALIDATION & REPORTING                  │
│  Optional AI validation pass -> Formats findings -> Outputs in │
│  Console/JSON/HTML/SARIF format with actionable remediation    │
└─────────────────────────────────────────────────────────────────┘
```

**Semantic Understanding**: FalconEYE reads your code like a security engineer, understanding business logic, data flows, and architectural patterns to identify real vulnerabilities.

**RAG-Enhanced Analysis**: By retrieving similar code patterns from your entire codebase, the AI gets crucial context about how functions are used, what data they handle, and potential security implications across your application.

**LLM-Powered Enrichment**: When the initial analysis produces incomplete findings (missing line numbers, generic descriptions, or vague mitigations), FalconEYE sends them back to the LLM with the full source code for enrichment. Every displayed finding includes specific line numbers, the vulnerable code snippet, a detailed description of the exploit, and actionable remediation referencing actual identifiers from your code.

---

## Getting Started

### Prerequisites

- **Python 3.12** (recommended) · 3.12+ minimum
- **Ollama** running locally ([Install Ollama](https://ollama.ai))

### Installation

```bash
# Pull required AI models
ollama pull qwen3-coder:30b
ollama pull embeddinggemma:300m

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install FalconEYE
pip install -e .
```

### Apple Silicon Installation (MLX)

For native Apple Silicon acceleration:

```bash
# Pull embedding model (still required -- MLX uses Ollama for embeddings)
ollama pull embeddinggemma:300m

# Install FalconEYE with MLX support
pip install -e ".[mlx]"
```

> **Note**: MLX requires an Apple Silicon Mac (M1/M2/M3/M4). The MLX backend uses a hybrid approach — MLX handles LLM inference while Ollama handles embeddings. Ollama must still be running for the indexing step. The MLX analysis model is downloaded automatically from HuggingFace on first use.

### Keeping Up to Date

Once installed from a git clone, you can upgrade FalconEYE to the latest version with a single command:

```bash
falconeye upgrade
```

This will:
1. `git pull origin main` — fetch and merge the latest changes
2. Reinstall dependencies from `pyproject.toml`
3. Show what changed (commits pulled) and the new version

> **Note**: `falconeye upgrade` requires FalconEYE to be installed from a git clone (`pip install -e .`). If installed from a package archive, reinstall from the repository instead.

### Your First Scan

```bash
# Full scan (index + review in one step)
falconeye scan /path/to/your/project

# Or run the steps separately:
falconeye index /path/to/your/project     # Index codebase (one-time)
falconeye review /path/to/your/project    # Analyze for vulnerabilities

# Scan with MLX backend on Apple Silicon
falconeye scan /path/to/your/project --backend mlx

# Generate an HTML report
falconeye scan /path/to/your/project -o html --output-file report.html

# Skip generated or vendored files in both steps
falconeye scan /path/to/your/project -e "*.stub.ts" -e "*/generated/*"

# Verbose mode -- see LLM streaming and full logs
falconeye scan /path/to/your/project -v
```

> **Run from one directory.** The index, log file and reports live under the directory you run `falconeye` from (`./falconeye_data/`, `./falconeye.log`, `./falconeye_reports/`). Run `index`, `review` and `scan` from the same place, or set absolute paths in `~/.falconeye/config.yaml` -- otherwise `review` finds no index and analyzes without RAG context. See [Data & Log Locations](#data--log-locations).

---

## MLX Backend (Apple Silicon)

FalconEYE supports native Apple Silicon inference via [MLX](https://github.com/ml-explore/mlx), Apple's machine learning framework optimized for the unified memory architecture and Neural Engine of M-series chips.

### Performance Benefits

MLX delivers significant performance improvements over Ollama (which uses llama.cpp internally) on Apple Silicon hardware:

| Metric | Improvement | Details |
|--------|-------------|---------|
| **Token Generation** | **20-40% faster** | MLX runs inference directly on the Apple GPU/Neural Engine. On MoE models like Qwen3-30B-A3B, benchmarks show 17-43% higher tok/s vs llama.cpp. Smaller models see up to 87% gains. |
| **Memory Usage** | **~30% lower RAM** | Zero-copy unified memory eliminates data duplication between CPU and GPU. Lazy evaluation fuses operations and reduces allocation overhead. |
| **First-Token Latency** | **~50% lower** | In-process inference removes the HTTP round-trip overhead of Ollama's REST API on localhost:11434. |
| **Prompt Processing** | **~25% faster prefill** | Native Metal compute path vs llama.cpp's Metal abstraction layer. |
| **Model Availability** | **Broader selection** | Access thousands of quantized models from HuggingFace's `mlx-community`, compared to Ollama's curated library. |

> Figures based on published benchmarks of MLX vs llama.cpp on M-series chips (Barrios et al., arXiv:2601.19139; arXiv:2511.05502; Google Cloud Community Gemma 3 benchmarks). Actual results vary by model size, quantization level, and hardware generation.

**When to use MLX:**
- You have an Apple Silicon Mac (M1/M2/M3/M4)
- You want faster scan times and lower memory consumption
- You want access to the latest quantized models from HuggingFace

**When to stick with Ollama:**
- Cross-platform deployment (Linux, Intel Mac, Windows via WSL)
- You prefer Ollama's curated model library and CLI tooling
- Running on non-Apple hardware

### Hybrid Architecture

The MLX backend uses a hybrid approach because MLX does not natively support embedding models:

```
                 ┌──────────────────────────┐
                 │      FalconEYE CLI       │
                 └────────────┬─────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
    ┌─────────▼──────────┐        ┌──────────▼──────────┐
    │   MLX (Analysis)   │        │  Ollama (Embeddings) │
    │                    │        │                      │
    │  Qwen3-Coder-30B  │        │  embeddinggemma:300m │
    │  4-bit quantized   │        │                      │
    │  Apple Silicon GPU │        │  localhost:11434     │
    └────────────────────┘        └──────────────────────┘
```

- **Inference**: MLX loads a quantized model directly into unified memory and runs on the Apple GPU/Neural Engine. Zero-copy memory access eliminates transfer overhead.
- **Embeddings**: Ollama handles embedding generation during the indexing step. Ollama must be running when you use `falconeye index` or `falconeye scan`.

### MLX Model Configuration

The default model is `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` — a Mixture-of-Experts model (30B total / 3B active parameters, 4-bit quantized). You can change the model in your config:

```yaml
llm:
  provider: mlx
  mlx:
    analysis: mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
```

Any MLX-compatible model from HuggingFace's `mlx-community` namespace can be used. The model is downloaded and cached automatically on first use.

### Checking MLX Availability

```bash
falconeye info
```

The `info` command shows whether your system supports MLX, including Apple Silicon detection and package availability.

---

## LiteLLM Backend (OpenAI-compatible)

`provider: litellm` talks the OpenAI protocol, so FalconEYE can use a LiteLLM proxy, vLLM, SGLang, TGI or any other OpenAI-compatible server. Chat (analysis) and embeddings can live on different hosts -- a common setup is a shared GPU server for analysis and a local Ollama for embeddings:

```yaml
llm:
  provider: litellm
  litellm:
    chat_base_url: https://llm.internal.example:8443/v1
    chat_model: gemma-4-31b-it-heretic
    chat_api_key: ""                         # LiteLLM virtual key; empty for unauthenticated
    embedding_base_url: http://localhost:11434/v1   # Ollama's OpenAI-compatible endpoint
    embedding_model: embeddinggemma:300m
    max_tokens: 32768
    request_timeout: 600.0
    guided_json: true                        # enforce the finding schema (vLLM >= 0.6, SGLang)
  embedding_batch_size: 32                   # max chunks per embedding request (all providers)
```

- `guided_json` sends `response_format.json_schema` so the server constrains output to FalconEYE's finding schema. Set it to `false` for servers that don't support it.
- The `--backend` flag only switches between `ollama` and `mlx`; choose LiteLLM with `llm.provider`.
- With LiteLLM, your code is sent to `chat_base_url` and `embedding_base_url`. Point them at infrastructure you trust.
- Indexing sends each file's chunks in batches of `embedding_batch_size`. If the server rejects a batch, its chunks are retried one at a time and only the rejected ones are skipped; the index summary lists any partially indexed files. (Very large batches made Ollama on Windows refuse connections to its own model runner, reported as HTTP 400.)

---

## Supported Languages

FalconEYE analyzes code in 11 languages with language-specific security knowledge:

| Language | Extensions | Key Vulnerability Categories |
|----------|-----------|------------------------------|
| **Python** | `.py`, `.pyw` | Command/code injection, pickle deserialization, SSRF, Django/Flask/FastAPI issues |
| **JavaScript / TypeScript** | `.js`, `.jsx`, `.ts`, `.tsx`, `.mjs`, `.cjs` | XSS, prototype pollution, eval injection, Node.js/React/Express issues |
| **Go** | `.go` | Command injection, SQL injection, race conditions, goroutine leaks |
| **Rust** | `.rs` | Unsafe blocks, FFI issues, integer overflow, Actix/Axum/Rocket issues |
| **C / C++** | `.c`, `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp`, `.hxx` | Buffer overflow, format strings, use-after-free, memory corruption |
| **Java** | `.java` | SQL injection, deserialization, XXE, Spring/Hibernate issues |
| **C#** | `.cs`, `.csx`, `.cshtml`, `.razor` | SQL injection, BinaryFormatter deserialization, XXE, mass assignment, ASP.NET Core / EF Core issues |
| **PHP** | `.php`, `.phtml`, `.php3`-`.php5`, `.phps` | SQL injection, type juggling, RCE, Laravel/WordPress issues |
| **Ruby** | `.rb`, `.rake`, `.gemspec`, `Gemfile`, `Rakefile` | YAML deserialization, mass assignment, eval/send injection, Rails issues |
| **Dart** | `.dart` | Insecure HTTP, hardcoded secrets, path traversal, Flutter issues |

Each language has a dedicated plugin with tailored security prompts, vulnerability categories, and framework-specific context. New languages can be added by implementing the `LanguagePlugin` interface.

---

## CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `falconeye scan <path>` | Index and review in one step |
| `falconeye index <path>` | Index codebase for analysis |
| `falconeye review <path>` | Analyze code for vulnerabilities |
| `falconeye taint <path>` | Spec-driven source-to-sink taint scan (see [Taint Analysis](#taint-analysis--reachability-confirmation)) |
| `falconeye taint-confirm <path> -f findings.json` | Add call-graph reachability to saved findings |
| `falconeye skill list` | List audit skills (`--show-patterns` to see Step 1 patterns) |
| `falconeye skill lint` | Check skill files for problems without running them |
| `falconeye skill run <name> <path>` | Run one audit skill (see [Audit Skills](#audit-skills)) |
| `falconeye skill run-all <path>` | Run every runnable skill and merge the findings |
| `falconeye feedback <finding-id>` | Record true/false-positive feedback (SAGE memory) |
| `falconeye upgrade` | Pull latest changes and reinstall dependencies |
| `falconeye info` | System and configuration information |
| `falconeye config --init` / `--show` | Create / print the configuration |
| `falconeye projects list` | Show all indexed projects |
| `falconeye projects info <id>` | Display project details |
| `falconeye projects delete <id>` | Delete a project and its data |
| `falconeye projects cleanup <id>` | Remove orphaned project data |

### Common Flags

| Flag | Short | Commands | Description |
|------|-------|----------|-------------|
| `--backend` | `-b` | index, review, scan | LLM backend: `ollama` (default) or `mlx` |
| `--verbose` | `-v` | index, review, scan, taint, taint-confirm, skill run, skill run-all | Logs (and, for review, LLM streaming) on the console; logs always go to the log file |
| `--language` | `-l` | index, review, scan | Programming language (auto-detected if omitted) |
| `--config` | `-c` | all | Path to configuration file |
| `--output` | `-o` | review, scan, taint, skill | Output format: `console`, `json`, `html`, `sarif` (`skill` and `taint-confirm`: `console` or `json`) |
| `--output-file` | | review, scan, taint, skill | Save results to a specific file |
| `--validate` | | review, scan | Enable AI validation pass to reduce false positives |
| `--top-k` | | review | Number of similar code chunks for RAG context (default: 5) |
| `--severity` | | review | Minimum severity to report: `critical`, `high`, `medium`, `low` |
| `--exclude` | `-e` | index, review, scan | Extra exclusion pattern, added to `file_discovery.default_exclusions` (repeatable) |
| `--force-reindex` | | index, scan | Force re-index all files (ignore cache) |
| `--concurrency` | | review, scan | Max files analyzed in parallel (default: `analysis.batch_size`) |
| `--resume` / `--no-resume` | | review, scan | Resume an interrupted scan from its checkpoint, or start fresh |
| `--confirm-taint` | | review, skill run, skill run-all | Add call-graph reachability to each finding |
| `--sage` | | review, scan | Use SAGE persistent memory for historical context |

**Exclusion patterns** strip their wildcards and match the remainder anywhere in the path relative to `<path>`: `*/node_modules/*` skips any `node_modules` directory, `*.stub.ts` skips files containing `.stub.ts`. Either `/` or `\` works. The same list applies to indexing and review.

### Examples

```bash
# Full scan with MLX backend
falconeye scan ./src --backend mlx

# Review with AI validation and HTML output
falconeye review ./src --validate -o html --output-file report.html

# Scan with verbose output (see LLM reasoning in real-time)
falconeye scan ./src -v

# Filter findings by severity
falconeye review ./src --severity high

# SARIF output for CI/CD integration
falconeye review ./src -o sarif --output-file results.sarif

# Force re-index and scan
falconeye scan ./src --force-reindex

# Exclude test and vendored directories (added to the configured defaults)
falconeye scan ./src -e "*/tests/*" -e "*/vendor/*"

# Review and confirm which findings are reachable from HTTP endpoints
falconeye review ./src --confirm-taint
```

### Verbose Mode

**Normal mode** (default): Clean progress bar with file count and percentage. Findings are displayed after LLM enrichment completes with full details. Logs go only to the log file (`logging.file`). After indexing, a summary lists any files that could not be indexed, or were indexed only partially.

**Verbose mode** (`-v`): Full indexing logs, LLM thought process streaming, detailed progress information, and complete error stack traces on the console, in addition to the log file.

---

## Taint Analysis & Reachability Confirmation

Taint support is opt-in per language and currently covers **C#** (ASP.NET Core). Enable it in your config; `csharp` must also be listed in `languages.enabled`:

```yaml
languages:
  per_language:
    csharp:
      taint:
        enabled: true
        max_hops: 5                 # call-graph edges from source to sink (1-20)
```

**Reachability confirmation** is the main use. It builds a call graph of the workspace, detects HTTP entry points (controller actions with `[HttpGet]`, `[HttpPost]`, `[Route]` and similar attributes), and marks each finding with a `taint_confirmation`: `confirmed_reachable` (with the call chain), `not_confirmed`, `no_enclosing_method`, `no_entry_points` or `skipped`. Confirmation only adds signal: findings that aren't confirmed keep their severity and confidence and are never dropped.

```bash
falconeye review ./Portal --confirm-taint                 # review + confirmation in one run
falconeye skill run-all ./Portal --confirm-taint          # same for skill findings
falconeye taint-confirm ./Portal -f findings.json -o json --output-file confirmed.json
```

`taint-confirm` accepts the JSON written by `review -o json` or by `skill run -o json`.

**Spec-driven scan.** `falconeye taint <path>` searches for source-to-sink paths using the shipped spec (`taint_specs/csharp_generic.yaml`: ADO.NET commands, `Process.Start`, file I/O, XML parsers, unsafe deserializers, outbound HTTP, reflection) and has the LLM verify each candidate path (`--no-verify` shows the raw candidates; `--max-hops` overrides the config).

**Limits.**
- Minimal-API endpoints (`app.MapGet(...)`), SignalR hubs and gRPC services are not detected as entry points yet.
- Calls across namespaces are only partly resolved, so reachable code can come back `not_confirmed`.

---

## Audit Skills

Skills are your own audit playbooks in markdown. FalconEYE runs them against a codebase and reports the results as ordinary findings.

**Where skills live.** Skills are read from `skills.directory` (default `~/.falconeye/skills`), as `<dir>/<name>/SKILL.md` or `<dir>/<name>.md`; entries starting with `_` are skipped. Enable them with `skills.enabled: true`, or set `FALCONEYE_SKILLS_DIR` (still requires `skills.enabled`). `--skills-dir` on the command line overrides both and works even when skills are disabled.

**File format.** YAML frontmatter, then the audit procedure that is sent to the LLM. Grep commands in a fenced `bash` block under a `Step 1` heading become discovery patterns:

````markdown
---
name: injection-prevention
description: SQL and command injection audit
function: scan              # scan runs; verify / triage are listed but not run yet
category: web-api-security
---

## Step 1 -- Discover

```bash
# EF Core raw SQL
grep -rn -E "(FromSqlRaw|ExecuteSqlRaw)\(" --include="*.cs" .
```

## Step 2 -- Evaluate
...
````

Patterns use Python regex semantics; `--include` filters by extension and is required.

**How skills run.**
- **With discovery patterns:** the grep hits pick the files, and the LLM evaluates each file once, with its hits listed. Hits are capped at 40 per pattern and `skills.max_discovery_hits` (200) in total, shared fairly across patterns.
- **Without patterns** ("locate and read" skills): the skill runs as a sweep, with one LLM call per source file of the enabled languages, up to `--max-files` (default `skills.max_llm_only_files`, 50). A FalconEYE-specific `targets:` frontmatter list of filename globs replaces the default selection, for example `targets: [package.json, "*.csproj", "requirements*.txt"]` for a dependency audit.
- **`run-all`** runs every scan skill. LLM-only skills are included only with `--include-llm-only`. Findings at the same (file, line, category) are merged by default (`--no-dedupe` to keep them all), and the survivor keeps the `skill:<name>` tags of every skill that flagged it.

```bash
falconeye skill list --show-patterns
falconeye skill lint                                     # find dead patterns and parsing problems
falconeye skill run injection-prevention ./Portal -o json --output-file inj.json
falconeye skill run input-validation ./Portal/src/Api --max-files 20
falconeye skill run-all ./Portal --include-llm-only --max-files 25
```

**`skill lint`** reports what would otherwise fail silently:
- regexes Python can't compile, or reads differently from `grep -E` (BRE-style `\|`, `[[:alpha:]]`, `\<`);
- grep commands the loader ignores (no `-E`, combined `-rnE`, no `--include`);
- broken frontmatter, which would turn a `verify` skill into a runnable `scan` skill;
- duplicate names, and skills that can't run yet.

It exits 1 on errors.

---

## Output Formats

### Console
Color-coded terminal output with Rich formatting:
```
╭─ SQL Injection Vulnerability ────────────────────────────────╮
│ Severity: HIGH | CWE-89                                     │
│ File: app/database.py (lines 42-45)                         │
│                                                             │
│ The function executes raw SQL with user input without       │
│ parameterization, allowing SQL injection attacks.           │
│                                                             │
│ Recommendation:                                             │
│ Replace the string concatenation in `execute_query()` at    │
│ line 43 with parameterized queries using `cursor.execute(   │
│ "SELECT * FROM users WHERE id = ?", (user_id,))`.          │
╰─────────────────────────────────────────────────────────────╯
```

### JSON
Machine-readable format for CI/CD integration and programmatic processing.

### HTML
Interactive reports with executive summaries, severity charts, and detailed finding cards. Auto-generated alongside JSON when no output file is specified.

### SARIF
Industry-standard format for GitHub Security, GitLab, and DevSecOps platforms.

**Default behavior**: When no output file is specified, FalconEYE saves both JSON and HTML reports to `./falconeye_reports/` with timestamps.

---

## Configuration

FalconEYE uses a hierarchical configuration system. Files are loaded in order (later overrides earlier):

1. Default config: `<install-dir>/config.yaml`
2. User config: `~/.falconeye/config.yaml`
3. Project config: `./falconeye.yaml` or `./.falconeye.yaml` (in the directory you run from)
4. `--config <file>` on the command line, then environment overrides such as `FALCONEYE_LLM_BASE_URL` and `FALCONEYE_OUTPUT_COLOR`

Create `~/.falconeye/config.yaml` to customize:

```yaml
llm:
  provider: ollama                  # "ollama", "mlx" or "litellm" (see LiteLLM Backend)
  model:
    analysis: qwen3-coder:30b      # AI model for security analysis
    embedding: embeddinggemma:300m  # Model for code embeddings
  base_url: http://localhost:11434
  timeout: 600                      # Request timeout in seconds
  embedding_batch_size: 32          # Max chunks per embedding request while indexing
  mlx:
    analysis: mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit  # MLX model

analysis:
  top_k_context: 5          # Number of similar code chunks for RAG context
  validate_findings: true    # Enable AI validation pass
  batch_size: 10            # Files per batch

file_discovery:
  default_exclusions:       # used by index AND review; -e adds to this list
    - "*/node_modules/*"
    - "*/dist/*"
    - "*.min.js"

skills:
  enabled: true
  directory: ~/.falconeye/skills
  max_llm_only_files: 50    # file cap for sweeping skills without Step 1 patterns

output:
  default_format: json      # console, json, html, sarif
  color: true               # Color-coded console output
  output_directory: ./falconeye_reports

logging:
  level: INFO               # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: ./falconeye.log     # always written; console output appears with -v
  rotation: daily
  retention_days: 30
```

See the [default config.yaml](config.yaml) for all available options, including taint settings (`languages.per_language`).

### Data & Log Locations

These default paths are **relative to the directory you run `falconeye` from**:

| What | Default path |
|------|--------------|
| Vector index, metadata, index registry | `./falconeye_data/` |
| Log file (JSON lines) | `./falconeye.log` |
| Reports | `./falconeye_reports/` |

Running `index` in one directory and `review` in another means `review` sees no index, so it analyzes without cross-file context. Run the commands from one place, or set absolute `persist_directory`, `logging.file` and `output.output_directory` paths in `~/.falconeye/config.yaml`.

---

## Architecture

FalconEYE follows hexagonal (ports and adapters) architecture for clean separation of concerns:

```
┌──────────────────────────────────────────────────────────────┐
│                      ADAPTERS LAYER                          │
│  CLI (Typer)  |  Console/HTML/JSON/SARIF Formatters          │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                    APPLICATION LAYER                          │
│  IndexCodebaseCommand  |  ReviewFileCommand                  │
│  TaintScan  |  ReviewConfirmation  |  SkillRunner            │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                      DOMAIN LAYER                            │
│  SecurityAnalyzer  |  ContextAssembler  |  Models  |  Ports  │
│  CallGraph  |  EntryPointDetector  |  TaintVerifier          │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                   INFRASTRUCTURE LAYER                        │
│  Ollama Adapter  |  MLX Adapter  |  ChromaDB  |  Plugins     │
│  LiteLLM Adapter  |  C# Taint Adapter  |  Skill Loader       │
│  Config Loader   |  DI Container |  Language Detection       │
└──────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- Domain layer defines abstract ports (`LLMService`, `VectorStore`); infrastructure provides concrete adapters
- DI container wires everything together with a factory pattern (`llm.provider` selects Ollama, MLX or LiteLLM; `--backend` can switch to MLX)
- Language plugins are self-contained: each provides security prompts, vulnerability categories, framework context, and chunking strategies
- Circuit breaker and retry patterns protect against LLM failures; embeddings and chat have separate breakers, so indexing trouble can't block analysis
- Optional stages (reachability confirmation, skills) only add to the findings: a failure in them is reported and never stops the primary output
- Structured JSON logging with correlation IDs

---

## Development

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install with MLX support (Apple Silicon)
pip install -e ".[mlx]"

# Run test suite
pytest

# Run integration tests (requires Ollama)
pytest tests/integration/ -v
```

---

## FAQ

**Q: Does my code get sent to external services?**
A: Not with Ollama or MLX: FalconEYE runs entirely locally and your code never leaves your machine. With `provider: litellm`, code is sent to the `chat_base_url` / `embedding_base_url` you configure, so point them at infrastructure you trust (for example an on-prem vLLM server).

**Q: How accurate is AI-based analysis compared to traditional scanners?**
A: FalconEYE complements traditional tools. It excels at finding context-dependent vulnerabilities and novel patterns that signature-based tools miss. The LLM enrichment pass ensures every finding includes specific details, and the optional AI validation pass reduces false positives.

**Q: Can I use different AI models?**
A: Yes. Configure any Ollama-compatible model in your config file. On Apple Silicon, you can also use any MLX-compatible model from HuggingFace's `mlx-community` namespace by setting `--backend mlx`. Any OpenAI-compatible server (LiteLLM, vLLM, SGLang, TGI) works through `provider: litellm`.

**Q: The index summary says some files "could not be indexed" or were "indexed partially". Does that affect the review?**
A: Those files are still reviewed, because review reads files from disk. They are missing, in whole or in part, from the RAG index, so they won't be offered as related code when other files are analyzed. The summary shows the server's error for each one, and `falconeye.log` has the details. Files that failed completely are retried on the next run; to retry skipped chunks, re-run with `--force-reindex`.

**Q: How do I keep generated or vendored code out of a scan?**
A: Add patterns to `file_discovery.default_exclusions` (for example `"*.stub.ts"`, `"*/generated/*"`), or pass `-e` to `index`, `review` or `scan`. Exclusions apply to indexing and review alike.

**Q: What is the MLX backend and why should I use it?**
A: MLX is Apple's ML framework for Apple Silicon. It provides 20-40% faster inference, ~30% lower memory usage, and ~50% lower first-token latency compared to Ollama by leveraging unified memory and running inference directly on the GPU/Neural Engine without HTTP overhead. Use `--backend mlx` or set `provider: mlx` in your config.

**Q: How does FalconEYE handle incomplete findings from local models?**
A: Local models (especially smaller quantized ones) sometimes produce findings with missing line numbers, vague descriptions, or generic mitigations. FalconEYE detects incomplete findings and automatically sends them back to the LLM with the full source code for enrichment — producing specific descriptions, actionable mitigations referencing actual identifiers, code snippets, and accurate line numbers.

**Q: How do I upgrade FalconEYE to the latest version?**
A: Run `falconeye upgrade` from anywhere. It detects your git clone location, runs `git pull origin main`, and reinstalls any new or updated dependencies automatically. If you're already on the latest version it will tell you so without making any changes.

**Q: How do I integrate this into CI/CD?**
A: Use SARIF output (`-o sarif --output-file results.sarif`), which integrates with GitHub Security, GitLab, and most DevSecOps platforms.

**Q: How long does analysis take?**
A: Initial indexing depends on codebase size. Subsequent scans with smart re-indexing only process changed files. Using the MLX backend on Apple Silicon reduces analysis time by 20-40% compared to Ollama.

---

## Discoveries

CVEs and vulnerabilities discovered by the FalconEYE authors through research and analysis.

---

### CVE-2026-27446 — Apache ActiveMQ Artemis: Authentication Bypass via Core Federation

| Field | Detail |
|-------|--------|
| **CVE ID** | [CVE-2026-27446](https://nvd.nist.gov/vuln/detail/CVE-2026-27446) |
| **Product** | Apache Artemis / Apache ActiveMQ Artemis |
| **CWE** | CWE-306 — Missing Authentication for Critical Function |
| **CVSS 4.0** | **9.3 (Critical)** |
| **CISA KEV** | Yes — added to Known Exploited Vulnerabilities catalog |
| **Affected** | Apache Artemis 2.50.0–2.51.0; Apache ActiveMQ Artemis 2.11.0–2.44.0 |
| **Fixed In** | Apache Artemis 2.52.0 |

**Description:**
An unauthenticated remote attacker can use the Core protocol to force a target broker to establish an outbound Core federation connection to an attacker-controlled rogue broker. This requires no authentication and no user interaction, enabling the attacker to inject messages into any queue and exfiltrate messages from any queue via the rogue broker.

**Mitigation (if patching is not immediately possible):**
- Remove Core protocol support from any acceptor receiving connections from untrusted sources, **or**
- Enforce two-way SSL (certificate-based authentication) on all connections before any message protocol handshake is attempted

---

## License

FalconEYE is licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).

See the [LICENSE](LICENSE) file for the complete license text.

Copyright (c) 2025 hardw00t & h4ckologic

---

## Quick Reference

```bash
# Full scan (index + review)
falconeye scan /path/to/project

# Scan with MLX backend (Apple Silicon)
falconeye scan /path/to/project --backend mlx

# Review only (requires prior indexing)
falconeye review /path/to/project

# Generate HTML report
falconeye review /path/to/project -o html --output-file report.html

# Filter by severity
falconeye review /path/to/project --severity high

# Skip generated files (added to the configured exclusions)
falconeye scan /path/to/project -e "*.stub.ts"

# Confirm which findings are reachable from HTTP entry points (C#)
falconeye review /path/to/project --confirm-taint

# Audit skills
falconeye skill lint
falconeye skill run-all /path/to/project --include-llm-only

# Verbose mode with LLM streaming
falconeye scan /path/to/project -v

# Upgrade to the latest version (git clone installs only)
falconeye upgrade

# System information (shows MLX availability)
falconeye info

# List indexed projects
falconeye projects list
```

### Default Output Locations
Relative to the directory you run `falconeye` from (see [Data & Log Locations](#data--log-locations)):
- **Index**: `./falconeye_data/`
- **Reports**: `./falconeye_reports/`
- **Logs**: `./falconeye.log`

User-level:
- **Config**: `~/.falconeye/config.yaml`
- **Skills**: `~/.falconeye/skills/`

---

**Built for security engineers who demand more than pattern matching.**

Version 2.1.0 | Python 3.12+ | Ollama + MLX + LiteLLM

By [hardw00t](https://github.com/hardw00t) & [h4ckologic](https://github.com/h4ckologic)
