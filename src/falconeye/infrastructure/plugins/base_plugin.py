"""Base plugin interface for language-specific analysis."""

from abc import ABC, abstractmethod
from typing import Dict, List


class LanguagePlugin(ABC):
    """
    Base class for language plugins.

    IMPORTANT: Plugins provide context and prompts for AI analysis,
    NOT pattern-based vulnerability detection rules.

    Each plugin provides:
    - Language-specific system prompts for AI
    - Vulnerability categories for context
    - Validation prompts to reduce false positives
    - Optional chunking strategies
    """

    @property
    @abstractmethod
    def language_name(self) -> str:
        """
        Language name.

        Returns:
            Language name (e.g., "python", "javascript")
        """
        pass

    @property
    @abstractmethod
    def file_extensions(self) -> List[str]:
        """
        File extensions for this language.

        Returns:
            List of file extensions (e.g., [".py", ".pyw"])
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get language-specific system prompt for security analysis.

        This prompt guides the AI to:
        - Understand language semantics
        - Consider common vulnerability patterns (for context, not matching)
        - Perform deep reasoning about code behavior
        - Output findings in structured format

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def get_validation_prompt(self) -> str:
        """
        Get prompt for validating findings to reduce false positives.

        The AI uses this prompt to review findings and determine
        if they are genuine vulnerabilities or false positives.

        Returns:
            Validation prompt string
        """
        pass

    @abstractmethod
    def get_vulnerability_categories(self) -> List[str]:
        """
        Get common vulnerability categories for this language.

        These categories provide context for the AI, NOT matching rules.
        They help the AI understand what types of issues to look for.

        Returns:
            List of vulnerability category names
        """
        pass

    def get_chunking_strategy(self) -> Dict[str, int]:
        """
        Get language-specific chunking parameters.

        Can be overridden for languages that need different chunking.

        Returns:
            Dictionary with 'chunk_size' and 'chunk_overlap' keys
        """
        return {
            "chunk_size": 50,
            "chunk_overlap": 10,
        }

    def get_framework_context(self) -> List[str]:
        """
        Get common frameworks/libraries for this language.

        This provides additional context for the AI about common
        security issues in popular frameworks.

        Returns:
            List of framework names
        """
        return []

    @staticmethod
    def get_severity_guidelines() -> str:
        """
        Get severity reasoning guidelines.

        These guidelines teach the LLM to reason about severity based on
        real-world exploitability and impact, not rigid category mappings.

        Returns:
            Severity reasoning guidance string
        """
        return """

SEVERITY RATING - THINK BEFORE YOU RATE:

Before assigning a severity, reason through these questions for EACH finding:

1. EXPLOITABILITY: How easy is it to exploit in a real attack?
   - Can an unauthenticated remote attacker trigger it directly?
   - Does it require special access, credentials, or local presence?
   - Does it need user interaction (clicking a link, opening a file)?
   - How complex is the attack chain?

2. IMPACT: What is the worst realistic outcome if exploited?
   - Does it give the attacker code execution on the server?
   - Does it expose the entire database or sensitive user data?
   - Does it only degrade security posture without direct compromise?
   - Is the impact theoretical or demonstrated by the code?

3. CONTEXT: What does the surrounding code tell you?
   - Are there mitigating controls elsewhere?
   - Is the vulnerable code in a hot path or a rarely-used feature?
   - Is user input actually reaching the vulnerable sink?

Then assign severity honestly:
- CRITICAL: Attacker can directly achieve RCE, full DB access, or complete auth bypass with minimal effort
- HIGH: Attacker can read/write arbitrary files, access internal services, or escalate privileges
- MEDIUM: Issue weakens security but requires chaining with other flaws or has limited direct impact
- LOW: Minor issue, hard to exploit, minimal real-world consequence
- INFO: Observation or recommendation, not an exploitable vulnerability

Common mistakes to AVOID:
- Using a weak hash (MD5/SHA1) or cipher (DES/RC4) is NOT critical - it weakens crypto but doesn't give direct access
- Weak TLS config is NOT critical - it enables potential MITM but requires network position
- A weak PRNG is NOT critical - it makes values predictable but rarely gives direct system access
- Integer overflow is NOT critical unless you can demonstrate it leads to memory corruption or auth bypass
- Not every finding is critical. A realistic scan should have a MIX of severities."""

    @staticmethod
    def get_reasoning_addendum(allowed_cwe_list: str = "") -> str:
        """
        Additional required fields and self-checks for every finding.

        Appended to every language-specific system prompt. Adds:
        - reasoning_trace: chain-of-thought BEFORE the verdict
        - evidence_lines: specific line numbers proving the vulnerability
        - assumed_context: caller/runtime assumptions being made
        - why_not_false_positive: explicit self-review
        - cwe_id from an enumerated list (banning confabulation)
        - explicit prohibition on inventing CVE numbers

        Design note on structure:
        Empirically, when a language plugin's prompt contains a
        concrete JSON example with only the original fields (which
        every plugin currently does), LLMs anchor on that example and
        ignore later "additional field" instructions -- especially in
        long prompts (>10k chars) where "lost in the middle" effects
        kick in.

        The addendum below therefore does two things:
          1. Explicitly overrides the earlier plugin JSON example
             with a directive ("REPLACES the example shown earlier").
          2. Ends with a NEW concrete JSON example showing ALL fields,
             positioned as the very last thing the model sees before
             generating. This is what LLMs actually copy from.

        Rationale for each requirement:
        - Chain-of-thought before verdict lets the model catch its own
          errors before committing. Empirically ~30% FP reduction.
        - evidence_lines and assumed_context are fields the model
          cannot fake without the fabrication being detectable.
        - why_not_false_positive is prompt-engineered self-review:
          the model must argue against its own finding, which surfaces
          the load-bearing assumption behind false positives.
        - Enumerated CWEs stop the most common confabulation mode
          (inventing plausible-sounding numbers).

        Args:
            allowed_cwe_list: Multi-line string listing valid CWE IDs
                and their names. Injected verbatim into the prompt.
                Sourced from finding_validation.ALLOWED_CWES.
        """
        return f"""

==================================================================
IMPORTANT: THE OUTPUT SCHEMA BELOW REPLACES ANY EARLIER JSON EXAMPLE

The system prompt above may have shown an earlier JSON output
format. IGNORE THAT SCHEMA. Use ONLY the extended schema defined
in this section for every finding.
==================================================================

REQUIRED REASONING PROCESS -- follow this for EVERY finding:

Step 1. Read the code carefully.
Step 2. In your response, before finalizing each finding, write out
        a `reasoning_trace` field: 3-5 sentences describing what the
        code does, what inputs it receives from where, what dangerous
        operation is being performed, and whether user-controlled data
        can actually reach it. This is THINKING OUT LOUD -- if your
        trace concludes the finding is not real, DO NOT emit it.
Step 3. Only after your reasoning_trace confirms the finding is real,
        fill in the other fields.

FIELD DEFINITIONS:

- issue: (string) Brief clear title of the vulnerability.
- reasoning: (string) Detailed 2-3 sentence description of the vuln,
      how it can be exploited, and impact.
- mitigation: (string) Specific actionable remediation referencing
      actual identifiers from the code.
- severity: (string) One of: critical, high, medium, low, info.
- confidence: (float 0.0-1.0) Your confidence in this finding.
- code_snippet: (string) The EXACT vulnerable code lines copied
      verbatim from the source, preserving indentation. The tool
      verifies snippets appear in the file -- paraphrases fail.
- line_start: (integer) First vulnerable line number.
- line_end: (integer) Last vulnerable line number.
- cwe_id: (string, optional) Choose from the enumerated CWE list
      below. Do NOT invent CWE numbers. If no CWE fits well, omit.
      The tool rejects unrecognized CWE IDs.
- reasoning_trace: (string, REQUIRED) The chain-of-thought from
      Step 2 above. 3-5 sentences of analysis before you committed.
- evidence_lines: (array of integers, REQUIRED) The specific line
      numbers whose contents prove the vulnerability. NOT a range --
      the individual lines you would highlight to a reviewer. These
      MUST exist in the file. Fabricated line numbers get the
      finding downgraded or dropped.
- assumed_context: (string, REQUIRED) What you are assuming about
      things you cannot see: caller behavior, runtime configuration,
      external inputs. E.g. "assuming the caller does not validate
      the id parameter before passing it in", or "assuming production
      is not running with debug=True". Load-bearing assumptions are
      often where false positives hide -- make yours explicit.
- why_not_false_positive: (string, REQUIRED) Two sentences explaining
      why this finding is NOT a false positive. Address specifically:
      (a) is the vulnerable path actually reachable with attacker
      input, and (b) do any visible sanitizers, framework protections,
      or checks neutralize the issue before it reaches the sink? If
      you cannot make a credible argument here, DO NOT emit the
      finding.

STRICT PROHIBITIONS:

- Do NOT cite specific CVE numbers (CVE-YYYY-NNNN) unless the code
  pattern exactly matches a known, published CVE for a specific
  library version. Do NOT invent CVE numbers.
- Do NOT emit findings where `code_snippet` is a paraphrase. Quote
  code EXACTLY as it appears, preserving all indentation.
- Do NOT emit findings you cannot ground with specific line numbers.

ALLOWED CWE IDs (choose only from these; leave cwe_id empty if
none fit):

{allowed_cwe_list}

CALIBRATION NOTES:

- Prefer emitting fewer, higher-quality findings over many
  speculative ones. A thoughtful reviewer would rather see 3
  well-grounded findings than 15 that require them to filter.
- If your reasoning_trace mentions "might", "could potentially",
  "if the caller doesn't validate", ask yourself whether you have
  actual evidence of the missing check. If not, the finding is a
  hypothesis, not a finding. Downgrade to INFO or omit.
- The most valuable findings are those where you can point at
  a specific line, name the tainted variable, name the sink, and
  show that no visible sanitizer sits between them. Aim for these.

==================================================================
FINAL OUTPUT SCHEMA -- USE EXACTLY THIS FORMAT

Every finding MUST include every field shown below. Missing REQUIRED
fields cause the finding to be rejected. This example REPLACES any
earlier "Output Format" section in this prompt.

{{
  "reviews": [
    {{
      "issue": "SQL Injection via unparameterized query in GetUserById",
      "reasoning": "The GetUserById method concatenates the id parameter directly into a SQL query string, allowing an attacker to inject arbitrary SQL. This can be exploited to read, modify, or delete arbitrary rows from the Users table.",
      "mitigation": "Replace string concatenation with a SqlParameter. Change the query to 'SELECT * FROM Users WHERE Id = @id' and add cmd.Parameters.AddWithValue(\\"@id\\", id).",
      "severity": "critical",
      "confidence": 0.95,
      "code_snippet": "var sql = \\"SELECT * FROM Users WHERE Id = \\" + userId;",
      "line_start": 42,
      "line_end": 42,
      "cwe_id": "CWE-89",
      "reasoning_trace": "The GetUserById action receives userId from Request.Query on line 41. On line 42 it concatenates that value directly into a raw SQL string. On line 44 the string is passed to SqlCommand and executed. No visible sanitization, validation, or parameterization exists between the input and the sink. User-controlled data reaches the SQL execution path unfiltered.",
      "evidence_lines": [41, 42, 44],
      "assumed_context": "Assuming Request.Query is populated from external HTTP input (standard ASP.NET behavior) and that the connection is not restricted to prepared statements only via database-level configuration.",
      "why_not_false_positive": "The tainted value flows through only two operations (query string retrieval and string concatenation) before hitting SqlCommand.ExecuteReader, with no intervening validation. ASP.NET's default model binding does not perform SQL-safety validation on arbitrary query string parameters, and no custom validation attribute or manual check is present."
    }}
  ]
}}

If no security issues are found, return: {{"reviews": []}}

Emit ONLY this JSON. Do NOT include any text before or after the
JSON block. Do NOT use markdown code fences (no ```json).
==================================================================
"""

    def __repr__(self) -> str:
        """String representation."""
        return f"<{self.__class__.__name__}: {self.language_name}>"