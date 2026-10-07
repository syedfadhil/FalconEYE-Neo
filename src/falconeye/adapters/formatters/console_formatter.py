"""Console formatter with rich, colored output."""

from ...domain.models.security import SecurityReview, SecurityFinding, Severity
from .base_formatter import OutputFormatter


class ConsoleFormatter(OutputFormatter):
    """
    Format security review results for console output.

    Provides human-readable, colored output using ANSI codes.
    """

    def __init__(self, use_color: bool = True, verbose: bool = False):
        """
        Initialize console formatter.

        Args:
            use_color: Enable colored output
            verbose: Enable verbose output
        """
        self.use_color = use_color
        self.verbose = verbose

    def format_review(self, review: SecurityReview) -> str:
        """
        Format complete security review for console.

        Args:
            review: SecurityReview to format

        Returns:
            Formatted console output
        """
        lines = []

        # Header
        lines.append(self._format_header(review))
        lines.append("")

        # Summary
        lines.append(self._format_summary(review))
        lines.append("")

        # Findings
        if review.findings:
            lines.append(self._bold("Security Findings:"))
            lines.append("")
            for i, finding in enumerate(review.findings, 1):
                lines.append(self._bold(f"Finding #{i}"))
                lines.append(self.format_finding(finding))
                if i < len(review.findings):
                    lines.append("")
        else:
            lines.append(self._success("No security issues found"))
            lines.append("")

        # Footer
        lines.append(self._format_footer(review))

        return "\n".join(lines)

    def format_finding(self, finding: SecurityFinding) -> str:
        """
        Format single security finding for console.

        Args:
            finding: SecurityFinding to format

        Returns:
            Formatted finding
        """
        lines = []

        # Severity and issue on one line
        severity_str = self._format_severity(finding.severity)
        lines.append(f"[{severity_str}] {self._bold(finding.issue)}")

        # Divider
        lines.append("-" * 70)

        # Location block
        if finding.file_path:
            lines.append(f"Location: {finding.file_path}")
            if finding.line_start:
                if finding.line_end and finding.line_end != finding.line_start:
                    lines.append(f"Lines: {finding.line_start}-{finding.line_end}")
                else:
                    lines.append(f"Line: {finding.line_start}")

        # Confidence
        confidence_str = self._format_confidence(finding.confidence)
        lines.append(f"Confidence: {confidence_str}")
        lines.append("")

        # Reasoning block
        if finding.reasoning:
            lines.append(self._bold("Description:"))
            lines.append(self._wrap_text(finding.reasoning, prefix="  "))
            lines.append("")

        # Mitigation block
        if finding.mitigation:
            lines.append(self._bold("Recommendation:"))
            lines.append(self._wrap_text(finding.mitigation, prefix="  "))
            lines.append("")

        # Code snippet
        if finding.code_snippet and self.verbose:
            lines.append(self._bold("Code:"))
            for line in finding.code_snippet.split("\n"):
                lines.append(f"  {line}")
            lines.append("")

        return "\n".join(lines)

    def get_file_extension(self) -> str:
        """Get file extension."""
        return ".txt"

    def _format_header(self, review: SecurityReview) -> str:
        """Format header section."""
        lines = []
        lines.append(self._bold("=" * 70))
        lines.append(self._bold("FalconEYE Security Review"))
        lines.append(self._bold("=" * 70))
        lines.append(f"Path: {review.codebase_path}")
        lines.append(f"Language: {review.language}")
        if review.started_at:
            lines.append(f"Started: {review.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)

    def _format_summary(self, review: SecurityReview) -> str:
        """Format summary section."""
        lines = []
        lines.append(self._bold("Summary:"))

        # Count by severity
        critical = review.get_critical_count()
        high = review.get_high_count()
        medium = review.get_medium_count()
        low = review.get_low_count()

        if critical > 0:
            lines.append(f"  {self._colorize('Critical:', 'red')} {critical}")
        if high > 0:
            lines.append(f"  {self._colorize('High:', 'orange')} {high}")
        if medium > 0:
            lines.append(f"  {self._colorize('Medium:', 'yellow')} {medium}")
        if low > 0:
            lines.append(f"  {self._colorize('Low:', 'blue')} {low}")

        # Total findings
        total = len(review.findings)
        lines.append(f"  {self._bold('Total:')} {total} issue{'s' if total != 1 else ''}")

        return "\n".join(lines)

    def _format_footer(self, review: SecurityReview) -> str:
        """Format footer section."""
        lines = []
        lines.append(self._bold("=" * 70))
        if review.completed_at and review.started_at:
            duration = review.completed_at - review.started_at
            lines.append(f"Duration: {duration}")
        lines.append("Analysis powered by AI (ZERO pattern matching)")
        lines.append(self._bold("=" * 70))
        return "\n".join(lines)

    def _format_severity(self, severity: Severity) -> str:
        """Format severity with color and icon."""
        severity_map = {
            Severity.CRITICAL: ("CRITICAL", "red"),
            Severity.HIGH: ("HIGH", "orange"),
            Severity.MEDIUM: ("MEDIUM", "yellow"),
            Severity.LOW: ("LOW", "blue"),
            Severity.INFO: ("INFO", "gray"),
        }

        text, color = severity_map.get(severity, ("UNKNOWN", "white"))
        return self._colorize(text, color)

    def _format_confidence(self, confidence) -> str:
        """Format confidence level."""
        from ...domain.models.security import FindingConfidence

        if confidence == FindingConfidence.HIGH:
            return self._colorize("High", "green")
        elif confidence == FindingConfidence.MEDIUM:
            return self._colorize("Medium", "yellow")
        else:
            return self._colorize("Low", "orange")

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text."""
        if not self.use_color:
            return text

        colors = {
            "red": "\033[91m",
            "orange": "\033[93m",
            "yellow": "\033[33m",
            "green": "\033[92m",
            "blue": "\033[94m",
            "gray": "\033[90m",
            "white": "\033[97m",
            "reset": "\033[0m",
        }

        color_code = colors.get(color, colors["white"])
        reset = colors["reset"]
        return f"{color_code}{text}{reset}"

    def _bold(self, text: str) -> str:
        """Make text bold."""
        if not self.use_color:
            return text
        return f"\033[1m{text}\033[0m"

    def _dim(self, text: str) -> str:
        """Make text dimmed."""
        if not self.use_color:
            return text
        return f"\033[2m{text}\033[0m"

    def _success(self, text: str) -> str:
        """Format success message."""
        return self._colorize(text, "green")

    def _wrap_text(self, text: str, prefix: str = "", width: int = 68) -> str:
        """Wrap text to specified width."""
        import textwrap
        wrapped = textwrap.fill(
            text,
            width=width,
            initial_indent=prefix,
            subsequent_indent=" " * len(prefix)
        )
        return wrapped