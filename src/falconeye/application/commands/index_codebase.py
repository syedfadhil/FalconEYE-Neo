"""Index codebase command and handler."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import time

from ...domain.models.codebase import Codebase, CodeFile
from ...domain.models.code_chunk import CodeChunk, ChunkMetadata
from ...domain.models.document import Document, DocumentChunk, DocumentMetadata
from ...domain.services.llm_service import LLMService
from ...domain.services.language_detector import LanguageDetector
from ...domain.services.project_identifier import ProjectIdentifier
from ...domain.services.checksum_service import ChecksumService
from ...domain.repositories.vector_store_repository import VectorStoreRepository
from ...domain.repositories.metadata_repository import MetadataRepository
from ...domain.repositories.index_registry import IndexRegistryRepository
from ...domain.value_objects.project_metadata import ProjectMetadata, FileMetadata, FileStatus
from ...infrastructure.ast.ast_analyzer import EnhancedASTAnalyzer
from ...infrastructure.logging import FalconEyeLogger
from ...infrastructure.resilience import CircuitBreakerError
from ...infrastructure.text import read_source_file
from ...infrastructure.workspace_files import is_excluded


#: Default number of chunks per embedding request. Small enough for
#: local servers that choke on huge batches (a 578-chunk request made
#: Ollama 0.32 on Windows refuse connections to its own model runner),
#: large enough to keep round trips few. Overridable via
#: config.llm.embedding_batch_size.
DEFAULT_EMBEDDING_BATCH_SIZE = 32

#: Default character cap per code chunk. Sized for embeddinggemma's
#: 2,048-token context: measured on a real TS monorepo, source code
#: runs 2.5-4.2 chars/token (median 3.6) and generated data 2.0-5.0
#: (median 3.1), so 4,500 chars stays within the window for all source
#: samples and nearly all data. Overridable via chunking.max_chunk_chars.
DEFAULT_MAX_CHUNK_CHARS = 4500

#: Consecutive single-chunk failures that mean "the service is down",
#: not "these chunks are bad". The file then fails as a whole and is
#: retried on the next run instead of being stored half-indexed.
_SERVICE_FAILURE_STREAK = 3


class EmbeddingServiceError(RuntimeError):
    """Embedding failed for service reasons; the file should be retried."""


def _char_capped_end(lines: List[str], start: int, end: int, max_chars: int) -> int:
    """Last line index (exclusive) in [start, end) that keeps the chunk
    within ``max_chars``. Always includes at least one line, so a single
    over-long line (minified code) becomes its own chunk rather than
    being split mid-line."""
    total = 0
    for i in range(start, end):
        total += len(lines[i])
        if total > max_chars:
            return max(i, start + 1)
    return end


def _is_input_rejection(error: Exception) -> bool:
    """Whether the endpoint rejected this particular input (HTTP 4xx).

    Backend-agnostic: LiteLLMAdapter raises EmbeddingRequestError (a
    ValueError) for 4xx, and ollama.ResponseError carries status_code.
    A rejected chunk is skipped; anything else counts toward the
    service-failure streak.
    """
    if isinstance(error, ValueError):
        return True
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500


@dataclass
class IndexRunStats:
    """What went wrong during one indexing run, for the CLI summary.

    ``files_failed``: (relative path, error) -- nothing stored; the file
    is not in the registry, so the next run retries it.
    ``files_partial``: (relative path, chunks skipped, chunks total,
    first error) -- the embeddable chunks were stored; the file is in
    the registry, so only --force-reindex (or editing it) retries the
    skipped chunks.
    """
    files_failed: List[Tuple[str, str]] = field(default_factory=list)
    files_partial: List[Tuple[str, int, int, str]] = field(default_factory=list)

    @property
    def chunks_skipped(self) -> int:
        return sum(p[1] for p in self.files_partial)


@dataclass
class IndexCodebaseCommand:
    """
    Command to index a codebase.

    This builds the vector index used for RAG,
    NOT for pattern-based detection.

    New in v2.0: Smart re-indexing with project isolation
    """
    codebase_path: Path
    language: Optional[str] = None
    excluded_patterns: Optional[List[str]] = None
    chunk_size: int = 40  # lines per chunk
    chunk_overlap: int = 15  # lines overlap
    include_documents: bool = True  # Whether to index documentation files
    doc_chunk_size: int = 1000  # characters per document chunk

    # Smart re-indexing fields
    project_id: Optional[str] = None  # Explicit project ID override (for monorepos)
    force_reindex: bool = False  # Force re-index all files regardless of changes

    # Concurrency: how many files to process in parallel. Bounded by a
    # semaphore inside handle() so we don't overwhelm Ollama with parallel
    # embedding requests it can't serve. Falls back to a safe default when
    # not set. This is the same knob that config.analysis.batch_size was
    # supposed to expose but previously didn't wire through.
    concurrency: int = 4

    # Chunks per embedding request. A file's chunks are sent in batches
    # of this size instead of all at once (a 92k-line generated file is
    # ~580 chunks). See DEFAULT_EMBEDDING_BATCH_SIZE.
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE

    # Character cap per code chunk, so each chunk fits the embedding
    # model's context. See DEFAULT_MAX_CHUNK_CHARS.
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS


class IndexCodebaseHandler:
    """
    Handler for index codebase command.

    Orchestrates:
    1. Project identification (git/non-git)
    2. Smart re-indexing (detect changed/new/deleted files)
    3. Language detection
    4. File discovery (code + documents)
    5. AST analysis (code only)
    6. Chunking (code + documents)
    7. Embedding generation (code + documents)
    8. Vector storage (project-scoped collections)
    9. Registry updates (project & file metadata)
    """

    def __init__(
        self,
        vector_store: VectorStoreRepository,
        metadata_repo: MetadataRepository,
        llm_service: LLMService,
        language_detector: LanguageDetector,
        ast_analyzer: EnhancedASTAnalyzer,
        project_identifier: ProjectIdentifier,
        checksum_service: ChecksumService,
        index_registry: IndexRegistryRepository,
    ):
        """
        Initialize handler.

        Args:
            vector_store: Vector storage for embeddings
            metadata_repo: Metadata storage for AST data
            llm_service: LLM for embedding generation
            language_detector: Language detection service
            ast_analyzer: AST analyzer for metadata extraction
            project_identifier: Project identification service
            checksum_service: File change detection service
            index_registry: Registry for project/file metadata
        """
        self.vector_store = vector_store
        self.metadata_repo = metadata_repo
        self.llm_service = llm_service
        self.language_detector = language_detector
        self.ast_analyzer = ast_analyzer
        self.project_identifier = project_identifier
        self.checksum_service = checksum_service
        self.index_registry = index_registry
        self.logger = FalconEyeLogger.get_instance()
        # Reset by every handle() call; read by the CLI afterwards.
        self.last_run_stats = IndexRunStats()

    async def handle(self, command: IndexCodebaseCommand) -> Codebase:
        """
        Execute index codebase command with smart re-indexing.

        Args:
            command: Index command

        Returns:
            Indexed codebase

        Flow:
        1. Identify project (auto-detect or use explicit project_id)
        2. Check registry for previous indexing
        3. Detect changed/new/deleted files
        4. Process only changed/new files (or all if force_reindex=True)
        5. Update registry with new metadata
        """
        start_time = time.time()
        self.last_run_stats = IndexRunStats()

        self.logger.info(
            "Starting codebase indexing",
            extra={
                "codebase_path": str(command.codebase_path),
                "language": command.language,
                "force_reindex": command.force_reindex,
                "include_documents": command.include_documents,
                "chunk_size": command.chunk_size,
            }
        )

        # Step 1: Identify project
        project_id, project_name, project_type, git_remote_url = \
            self.project_identifier.identify_project(
                command.codebase_path,
                explicit_id=command.project_id
            )

        self.logger.info(
            "Project identified",
            extra={
                "project_id": project_id,
                "project_name": project_name,
                "project_type": project_type.value,
                "git_remote_url": git_remote_url,
            }
        )

        # Step 2: Detect language
        if command.language:
            language = command.language
        else:
            language = self.language_detector.detect_language(command.codebase_path)

        self.logger.info(
            "Language detected",
            extra={
                "language": language,
                "auto_detected": command.language is None,
            }
        )

        # Step 3: Check registry for previous indexing
        existing_project = self.index_registry.get_project(project_id)
        is_first_time = existing_project is None

        if is_first_time:
            self.logger.info(
                "First-time indexing for project",
                extra={"project_id": project_id}
            )
        else:
            last_indexed = existing_project.indexed_at if hasattr(existing_project, 'indexed_at') else "unknown"
            self.logger.info(
                "Re-indexing existing project",
                extra={
                    "project_id": project_id,
                    "last_indexed": str(last_indexed),
                }
            )

        # Step 4: Discover current files
        files = self._discover_files(command.codebase_path, language, command.excluded_patterns or [])

        self.logger.info(
            "File discovery completed",
            extra={
                "files_found": len(files),
                "language": language,
            }
        )

        # Step 5: Determine which files to process
        if command.force_reindex or is_first_time:
            files_to_process = files
            skipped_count = 0
            self.logger.info(
                "Processing all files",
                extra={
                    "total_files": len(files),
                    "force_reindex": command.force_reindex,
                    "first_time": is_first_time,
                }
            )
        else:
            # Smart re-indexing: only process changed/new files
            files_to_process, skipped_count = await self._filter_changed_files(
                project_id, files, command.codebase_path
            )
            self.logger.info(
                "Smart re-index analysis completed",
                extra={
                    "files_to_process": len(files_to_process),
                    "files_skipped": skipped_count,
                    "total_files": len(files),
                }
            )

        # Step 6: Handle deleted files
        if not is_first_time and not command.force_reindex:
            await self._handle_deleted_files(project_id, files, command.codebase_path)

        # Step 7: Create codebase entity
        codebase = Codebase.create(
            root_path=command.codebase_path,
            language=language,
            excluded_patterns=command.excluded_patterns or [],
        )

        # Cache the current git commit once per run instead of shelling
        # out to `git` for every file processed.
        cached_git_commit = self._get_current_commit(command.codebase_path)

        # Step 8: Process code files concurrently, bounded by a semaphore.
        # Ordered iteration would have serialized every AST parse, embedding
        # batch, and vector-store write; on a large codebase that dominates
        # wall-clock time even though most of the work is I/O-bound and
        # trivially parallelizable.
        concurrency = max(1, command.concurrency)
        file_sem = asyncio.Semaphore(concurrency)

        async def _process_one(file_path):
            try:
                file_language = self.language_detector.detect_language(file_path)
            except Exception:
                file_language = language
            async with file_sem:
                return await self._process_file(
                    file_path,
                    file_language,
                    command,
                    codebase,
                    project_id,
                    cached_git_commit=cached_git_commit,
                )

        file_results = await asyncio.gather(
            *(_process_one(fp) for fp in files_to_process),
            return_exceptions=False,
        )
        processed_files = [f for f in file_results if f is not None]

        # Step 9: Process documents if enabled (also parallelized).
        doc_count = 0
        if command.include_documents:
            doc_files = self._discover_documents(command.codebase_path, command.excluded_patterns or [])

            self.logger.info(
                "Document discovery completed",
                extra={"documents_found": len(doc_files)}
            )

            doc_sem = asyncio.Semaphore(concurrency)

            async def _process_doc(doc_path):
                async with doc_sem:
                    await self._process_document(doc_path, command)

            await asyncio.gather(*(_process_doc(dp) for dp in doc_files))
            doc_count = len(doc_files)

        # Step 10: Update project metadata in registry
        # Detect all languages for metadata
        try:
            all_languages = self.language_detector.detect_all_languages(command.codebase_path)
        except Exception:
            all_languages = [language]

        project_metadata = ProjectMetadata(
            project_id=project_id,
            project_name=project_name,
            project_root=command.codebase_path,
            project_type=project_type,
            git_remote_url=git_remote_url if project_type.value == "git" else None,
            last_indexed_commit=cached_git_commit if project_type.value == "git" else None,
            total_files=len(files),
            total_chunks=sum(f.chunk_count for f in processed_files),
            languages=all_languages,  # Now stores all detected languages
        )
        self.index_registry.save_project(project_metadata)

        # Calculate duration
        duration = time.time() - start_time

        self.logger.info(
            "Codebase indexing completed",
            extra={
                "project_id": project_id,
                "project_name": project_name,
                "languages_indexed": all_languages,
                "files_processed": len(files_to_process),
                "files_skipped": skipped_count,
                "documents_processed": doc_count,
                "total_chunks": sum(f.chunk_count for f in processed_files),
                "duration_seconds": round(duration, 2),
            }
        )

        return codebase

    async def _process_file(
        self,
        file_path: Path,
        language: str,
        command: IndexCodebaseCommand,
        codebase: Codebase,
        project_id: str,
        cached_git_commit: Optional[str] = None,
    ) -> Optional[FileMetadata]:
        """
        Process a single file and return file metadata.

        Args:
            file_path: Absolute path to file
            language: Programming language
            command: Index command
            codebase: Codebase entity
            project_id: Project identifier

        Returns:
            FileMetadata if successful, None otherwise
        """
        start_time = time.time()
        relative_path = file_path.relative_to(command.codebase_path)

        try:
            self.logger.debug(
                "Starting file processing",
                extra={
                    "file_path": str(relative_path),
                    "language": language,
                    "project_id": project_id,
                }
            )

            # Read file. Tolerates non-UTF-8 encodings (CP1252 etc.).
            content = read_source_file(file_path)

            # Create code file
            code_file = CodeFile.create(
                path=file_path,
                relative_path=str(relative_path),
                content=content,
                language=language,
            )
            codebase.add_file(code_file)

            # Extract AST metadata
            metadata = self.ast_analyzer.analyze_file(
                file_path=str(relative_path),
                content=content,
            )

            # Store metadata
            await self.metadata_repo.store_metadata(metadata)

            # Chunk the file
            chunks = self._chunk_content(
                content=content,
                file_path=str(relative_path),
                language=language,
                chunk_size=command.chunk_size,
                overlap=command.chunk_overlap,
                max_chars=command.max_chunk_chars,
            )

            self.logger.debug(
                "File chunked",
                extra={
                    "file_path": str(relative_path),
                    "chunk_count": len(chunks),
                    "chunk_size": command.chunk_size,
                }
            )

            # Generate embeddings in bounded batches. A chunk the server
            # rejects is skipped on its own instead of failing the file.
            texts = [chunk.content for chunk in chunks]
            vectors, chunk_errors = await self._embed_texts(
                texts, command.embedding_batch_size, str(relative_path),
            )
            chunks_with_embeddings = [
                chunk.with_embedding(vector)
                for chunk, vector in zip(chunks, vectors)
                if vector is not None
            ]
            if chunks and not chunks_with_embeddings:
                raise EmbeddingServiceError(
                    f"none of {len(chunks)} chunks could be embedded: {chunk_errors[0]}"
                )
            if chunk_errors:
                self.last_run_stats.files_partial.append(
                    (str(relative_path), len(chunk_errors), len(chunks), chunk_errors[0])
                )
                self.logger.warning(
                    "Some chunks could not be embedded; stored the rest",
                    extra={
                        "file_path": str(relative_path),
                        "chunks_skipped": len(chunk_errors),
                        "chunks_total": len(chunks),
                        "first_error": chunk_errors[0],
                    },
                )

            # Store chunks (project-scoped collection)
            await self.vector_store.store_chunks(chunks_with_embeddings, collection="code")

            # Get embedding IDs (if vector store supports it)
            embedding_ids = [str(chunk.id) if hasattr(chunk, 'id') else f"chunk_{i}"
                           for i, chunk in enumerate(chunks_with_embeddings)]

            # Create file metadata snapshot. Use the run-level cached
            # commit hash to avoid a `git rev-parse` subprocess per file.
            file_metadata = self.checksum_service.get_file_metadata_snapshot(
                file_path=file_path,
                relative_path=relative_path,
                project_id=project_id,
                language=language,
                git_commit_hash=cached_git_commit,
            )

            # Update with chunk info
            file_metadata = FileMetadata(
                project_id=file_metadata.project_id,
                file_path=file_metadata.file_path,
                relative_path=file_metadata.relative_path,
                language=file_metadata.language,
                file_checksum=file_metadata.file_checksum,
                file_size=file_metadata.file_size,
                file_mtime=file_metadata.file_mtime,
                git_commit_hash=file_metadata.git_commit_hash,
                git_file_hash=file_metadata.git_file_hash,
                indexed_at=file_metadata.indexed_at,
                chunk_count=len(chunks_with_embeddings),
                embedding_ids=embedding_ids,
                status=FileStatus.ACTIVE,
            )

            # Save to registry
            self.index_registry.save_file(file_metadata)

            # Calculate duration
            duration = time.time() - start_time

            self.logger.debug(
                "File processing completed",
                extra={
                    "file_path": str(relative_path),
                    "chunk_count": len(chunks),
                    "file_size": file_metadata.file_size,
                    "duration_seconds": round(duration, 2),
                }
            )

            return file_metadata

        except Exception as e:
            duration = time.time() - start_time
            self.last_run_stats.files_failed.append(
                (str(relative_path), f"{type(e).__name__}: {e}")
            )
            self.logger.error(
                "File processing failed",
                extra={
                    "file_path": str(relative_path),
                    "error": str(e),
                    "duration_seconds": round(duration, 2),
                },
                exc_info=True
            )
            return None

    async def _embed_texts(
        self,
        texts: List[str],
        batch_size: int,
        label: str,
    ) -> Tuple[List[Optional[List[float]]], List[str]]:
        """Embed ``texts`` in requests of at most ``batch_size`` texts.

        Returns one vector per text (``None`` where it couldn't be
        embedded) and an error message per skipped text. A failed batch
        is retried one text at a time, so a single rejected chunk costs
        only itself. Raises :class:`EmbeddingServiceError` instead when
        the failures point at the service rather than the input -- the
        circuit breaker is open, or several single-text retries in a row
        fail with non-rejection errors (timeouts, refused connections,
        5xx). The caller then fails the whole file, which keeps it out
        of the registry so the next run retries it.
        """
        vectors: List[Optional[List[float]]] = [None] * len(texts)
        errors: List[str] = []
        size = max(1, batch_size)
        for start in range(0, len(texts), size):
            batch = texts[start:start + size]
            try:
                out = await self.llm_service.generate_embeddings_batch(batch)
                if len(out) != len(batch):
                    raise ValueError(
                        f"endpoint returned {len(out)} embeddings for {len(batch)} inputs"
                    )
                vectors[start:start + len(batch)] = out
                continue
            except CircuitBreakerError as e:
                raise EmbeddingServiceError(str(e)) from e
            except Exception as e:
                self.logger.warning(
                    "Embedding batch failed; retrying its chunks one at a time",
                    extra={
                        "file_path": label,
                        "batch_start": start,
                        "batch_size": len(batch),
                        "error": str(e),
                    },
                )

            streak = 0
            for offset, text in enumerate(batch):
                try:
                    vectors[start + offset] = (
                        await self.llm_service.generate_embeddings_batch([text])
                    )[0]
                    streak = 0
                except CircuitBreakerError as e:
                    raise EmbeddingServiceError(str(e)) from e
                except Exception as e:
                    errors.append(f"{type(e).__name__}: {e}")
                    if _is_input_rejection(e):
                        continue  # this chunk is bad; keep going
                    streak += 1
                    if streak >= _SERVICE_FAILURE_STREAK:
                        raise EmbeddingServiceError(
                            f"{streak} consecutive embedding failures: {e}"
                        ) from e
        return vectors, errors

    async def _filter_changed_files(
        self,
        project_id: str,
        current_files: List[Path],
        project_root: Path,
    ) -> tuple[List[Path], int]:
        """
        Filter files to only process changed/new ones.

        Args:
            project_id: Project identifier
            current_files: List of current files in project
            project_root: Project root path

        Returns:
            Tuple of (files_to_process, skipped_count)
        """
        # Get cached file metadata from registry
        cached_metadata = self.index_registry.get_files_metadata_dict(project_id)

        # Use checksum service to filter changed files
        changed_files, unchanged_files = self.checksum_service.filter_changed_files_efficient(
            current_files,
            cached_metadata,
            use_checksum=False,  # Use quick check (mtime + size)
        )

        # Identify new files
        new_files = self.checksum_service.identify_new_files(
            set(current_files),
            set(cached_metadata.keys())
        )

        # Combine changed and new files
        files_to_process = list(set(changed_files) | new_files)
        skipped_count = len(unchanged_files)

        return files_to_process, skipped_count

    async def _handle_deleted_files(
        self,
        project_id: str,
        current_files: List[Path],
        project_root: Path,
    ):
        """
        Handle deleted files by marking them in registry.

        Args:
            project_id: Project identifier
            current_files: List of current files in project
            project_root: Project root path
        """
        # Get cached file metadata
        cached_metadata = self.index_registry.get_files_metadata_dict(project_id)

        # Identify deleted files
        deleted_files = self.checksum_service.identify_deleted_files(
            set(current_files),
            set(cached_metadata.keys())
        )

        if deleted_files:
            self.logger.info(
                "Deleted files detected",
                extra={
                    "project_id": project_id,
                    "deleted_count": len(deleted_files),
                }
            )

            for deleted_file in deleted_files:
                relative_path = deleted_file.relative_to(project_root) if deleted_file.is_absolute() else deleted_file

                # Mark as deleted in registry
                self.index_registry.mark_file_deleted(project_id, deleted_file)

                self.logger.info(
                    "File marked as deleted",
                    extra={
                        "file_path": str(relative_path),
                        "project_id": project_id,
                    }
                )

            self.logger.warning(
                "Deleted file embeddings marked for cleanup",
                extra={
                    "project_id": project_id,
                    "deleted_count": len(deleted_files),
                    "cleanup_command": "falconeye projects cleanup",
                }
            )

    def _get_current_commit(self, project_root: Path) -> Optional[str]:
        """
        Get current git commit hash if project is a git repository.

        Args:
            project_root: Project root path

        Returns:
            Commit hash or None
        """
        try:
            return self.project_identifier.get_current_git_commit(project_root)
        except Exception:
            return None

    def _discover_files(
        self,
        root_path: Path,
        language: str,
        excluded_patterns: List[str],
    ) -> List[Path]:
        """
        Discover source files for ALL languages in the codebase.

        This method now detects and indexes files from all supported languages,
        not just the primary language. This enables multi-language codebase support.

        Args:
            root_path: Root directory
            language: Primary language (kept for backward compatibility, but now indexes all)
            excluded_patterns: Patterns to exclude

        Returns:
            List of file paths from all detected languages
        """
        # Detect all languages in the codebase
        try:
            detected_languages = self.language_detector.detect_all_languages(root_path)
            self.logger.info(
                "Multi-language detection completed",
                extra={
                    "languages_detected": detected_languages,
                    "language_count": len(detected_languages),
                }
            )
        except Exception as e:
            # Fallback to single language if detection fails
            self.logger.warning(
                "Multi-language detection failed, using primary language only",
                extra={"error": str(e), "primary_language": language}
            )
            detected_languages = [language]

        # Collect files from all detected languages
        files = []
        for lang in detected_languages:
            extensions = self.language_detector.LANGUAGE_EXTENSIONS.get(lang, [])
            for ext in extensions:
                # Find all files with this extension
                found = list(root_path.rglob(f"*{ext}"))
                files.extend(found)

        # Remove duplicates (in case of overlapping extensions)
        files = list(set(files))

        # Filter excluded patterns (root-relative, separator-agnostic)
        filtered_files = [
            file_path for file_path in files
            if not is_excluded(file_path.relative_to(root_path), excluded_patterns)
        ]

        return filtered_files

    def _chunk_content(
        self,
        content: str,
        file_path: str,
        language: str,
        chunk_size: int,
        overlap: int,
        max_chars: Optional[int] = None,
    ) -> List[CodeChunk]:
        """
        Chunk file content with adaptive sizing for large files.

        Chunks at line boundaries, NOT arbitrary character counts.
        For large files (>1000 lines), uses larger chunks to reduce overhead.

        ``max_chars`` additionally ends a chunk before it would exceed
        that many characters (still at a line boundary; a single longer
        line stays one chunk). It exists because the embedding model
        only sees its context window: 200-line adaptive chunks ran to
        ~9,000 chars, were cut to 8,000 by the adapter and again to
        2,048 tokens by the server, so most of each chunk never reached
        the index. A chunk cut short this way overlaps the next by at
        most 20% of its own lines (the ratio adaptive chunking uses), so
        the walk always advances. Chunks under the cap are exactly as
        before.

        Args:
            content: File content
            file_path: File path
            language: Language
            chunk_size: Lines per chunk
            overlap: Overlap lines
            max_chars: Character cap per chunk (None = lines only)

        Returns:
            List of code chunks
        """
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        # Adaptive chunking for large files.
        # For files above ADAPTIVE_THRESHOLD_LINES, scale chunk size up (capped at
        # MAX_ADAPTIVE_CHUNK_SIZE, which mirrors chunking.max_chunk_size in
        # config.yaml) so the number of chunks — and therefore the number of LLM
        # calls — grows sub-linearly with file size.
        #
        # Target: ~50 chunks for moderately large files (up to ~10k lines). For
        # very large files (>10k lines) the chunk size is capped, so chunk count
        # continues to grow but at a much slower rate than with the default size.
        ADAPTIVE_THRESHOLD_LINES = 1000
        MAX_ADAPTIVE_CHUNK_SIZE = 200
        TARGET_CHUNK_COUNT = 50

        if total_lines > ADAPTIVE_THRESHOLD_LINES:
            target_size = total_lines // TARGET_CHUNK_COUNT
            adaptive_chunk_size = max(chunk_size, min(MAX_ADAPTIVE_CHUNK_SIZE, target_size))

            if adaptive_chunk_size != chunk_size:
                # Scale overlap with chunk size (20%) while honoring the caller's
                # minimum; this preserves cross-chunk context for larger windows.
                adaptive_overlap = max(overlap, adaptive_chunk_size // 5)

                self.logger.info(
                    "Using adaptive chunking for large file",
                    extra={
                        "file_path": file_path,
                        "total_lines": total_lines,
                        "original_chunk_size": chunk_size,
                        "adaptive_chunk_size": adaptive_chunk_size,
                        "original_overlap": overlap,
                        "adaptive_overlap": adaptive_overlap,
                    }
                )
                chunk_size = adaptive_chunk_size
                overlap = adaptive_overlap

        chunks = []
        start = 0
        chunk_index = 0
        char_limited = 0

        while start < len(lines):
            end = min(start + chunk_size, len(lines))
            advance = chunk_size - overlap
            if max_chars:
                capped_end = _char_capped_end(lines, start, end, max_chars)
                if capped_end < end:
                    end = capped_end
                    taken = end - start
                    advance = taken - min(overlap, taken // 5)
                    char_limited += 1

            # Get chunk content
            chunk_lines = lines[start:end]
            chunk_content = "".join(chunk_lines)

            # Create metadata
            metadata = ChunkMetadata(
                file_path=file_path,
                language=language,
                start_line=start + 1,
                end_line=end,
                chunk_index=chunk_index,
                total_chunks=0,  # Will be updated
            )

            # Create chunk
            chunk = CodeChunk.create(
                content=chunk_content,
                metadata=metadata,
                token_count=self.llm_service.count_tokens(chunk_content),
            )

            chunks.append(chunk)
            chunk_index += 1

            # Move to next chunk with overlap
            start += advance

        # Update total chunks
        chunks = [
            CodeChunk.create(
                content=chunk.content,
                metadata=ChunkMetadata(
                    file_path=chunk.metadata.file_path,
                    language=chunk.metadata.language,
                    start_line=chunk.metadata.start_line,
                    end_line=chunk.metadata.end_line,
                    chunk_index=chunk.metadata.chunk_index,
                    total_chunks=len(chunks),
                ),
                token_count=chunk.token_count,
            )
            for chunk in chunks
        ]

        self.logger.info(
            "File chunking completed",
            extra={
                "file_path": file_path,
                "total_lines": total_lines,
                "chunk_count": len(chunks),
                "avg_chunk_size": total_lines // len(chunks) if chunks else 0,
                "char_limited_chunks": char_limited,
            }
        )

        return chunks

    def _discover_documents(
        self,
        root_path: Path,
        excluded_patterns: List[str],
    ) -> List[Path]:
        """
        Discover documentation files.

        Looks for: README, CONTRIBUTING, SECURITY, API docs,
        architecture docs, design docs, etc.

        Args:
            root_path: Root directory
            excluded_patterns: Patterns to exclude

        Returns:
            List of document file paths
        """
        # Document extensions and patterns
        doc_patterns = [
            "*.md",
            "*.markdown",
            "*.txt",
            "*.rst",
            "*.adoc",
            "*.asciidoc",
            "README*",
            "CONTRIBUTING*",
            "SECURITY*",
            "CHANGELOG*",
            "LICENSE*",
            "docs/**/*",
            "documentation/**/*",
        ]

        doc_files = []
        for pattern in doc_patterns:
            found = list(root_path.rglob(pattern))
            doc_files.extend(found)

        # Remove duplicates
        doc_files = list(set(doc_files))

        # Filter excluded patterns and non-files
        return [
            doc_path for doc_path in doc_files
            if doc_path.is_file()
            and not is_excluded(doc_path.relative_to(root_path), excluded_patterns)
        ]

    async def _process_document(
        self,
        doc_path: Path,
        command: IndexCodebaseCommand,
    ):
        """
        Process a documentation file.

        Args:
            doc_path: Path to document
            command: Index command with settings
        """
        start_time = time.time()
        relative_path = str(doc_path.relative_to(command.codebase_path))

        try:
            # Skip binary files based on extension
            binary_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico',
                               '.pdf', '.zip', '.tar', '.gz', '.exe', '.bin',
                               '.woff', '.woff2', '.ttf', '.eot', '.svg'}
            if doc_path.suffix.lower() in binary_extensions:
                self.logger.info(
                    "Skipping binary document",
                    extra={
                        "file_path": relative_path,
                        "extension": doc_path.suffix,
                    }
                )
                return

            # Read document. read_source_file() tolerates non-UTF-8
            # encodings and never raises UnicodeDecodeError, so the
            # previous except-and-skip block is no longer needed.
            content = read_source_file(doc_path)

            self.logger.info(
                "Starting document processing",
                extra={
                    "file_path": relative_path,
                    "doc_size": len(content),
                }
            )

            # Determine document type
            doc_type = self._classify_document(doc_path.name, relative_path)

            # Create document
            document = Document.create(
                path=doc_path,
                relative_path=relative_path,
                content=content,
                document_type=doc_type,
            )

            # Chunk the document
            chunks = self._chunk_document(
                content=content,
                metadata=document.metadata,
                chunk_size=command.doc_chunk_size,
            )

            # Generate embeddings in bounded batches (same isolation as
            # code files: a rejected chunk is skipped on its own).
            texts = [chunk.content for chunk in chunks]
            vectors, chunk_errors = await self._embed_texts(
                texts, command.embedding_batch_size, relative_path,
            )
            chunks_with_embeddings = [
                chunk.with_embedding(vector)
                for chunk, vector in zip(chunks, vectors)
                if vector is not None
            ]
            if chunk_errors:
                self.logger.warning(
                    "Some document chunks could not be embedded; stored the rest",
                    extra={
                        "file_path": relative_path,
                        "chunks_skipped": len(chunk_errors),
                        "chunks_total": len(chunks),
                        "first_error": chunk_errors[0],
                    },
                )

            # Store chunks in separate collection
            await self.vector_store.store_document_chunks(
                chunks_with_embeddings,
                collection="documents"
            )

            # Calculate duration
            duration = time.time() - start_time

            self.logger.info(
                "Document processing completed",
                extra={
                    "file_path": relative_path,
                    "doc_type": doc_type,
                    "chunk_count": len(chunks),
                    "duration_seconds": round(duration, 2),
                }
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                "Document processing failed",
                extra={
                    "file_path": relative_path,
                    "error": str(e),
                    "duration_seconds": round(duration, 2),
                },
                exc_info=True
            )

    def _classify_document(self, filename: str, relative_path: str) -> str:
        """
        Classify document type based on filename and path.

        Args:
            filename: Name of the file
            relative_path: Relative path from root

        Returns:
            Document type classification
        """
        filename_upper = filename.upper()
        path_lower = relative_path.lower()

        # Check filename patterns
        if "README" in filename_upper:
            return "readme"
        elif "CONTRIBUTING" in filename_upper:
            return "contributing"
        elif "SECURITY" in filename_upper:
            return "security_policy"
        elif "CHANGELOG" in filename_upper:
            return "changelog"
        elif "LICENSE" in filename_upper:
            return "license"
        elif "API" in filename_upper or "api" in path_lower:
            return "api_doc"
        elif "ARCHITECTURE" in filename_upper or "architecture" in path_lower:
            return "architecture"
        elif "DESIGN" in filename_upper or "design" in path_lower:
            return "design_doc"
        elif "GUIDE" in filename_upper or "tutorial" in path_lower:
            return "guide"
        else:
            return "documentation"

    def _chunk_document(
        self,
        content: str,
        metadata: DocumentMetadata,
        chunk_size: int,
    ) -> List[DocumentChunk]:
        """
        Chunk document content by character count with overlap.

        Args:
            content: Document content
            metadata: Document metadata
            chunk_size: Characters per chunk

        Returns:
            List of document chunks
        """
        chunks = []
        chunk_index = 0
        overlap = chunk_size // 4  # 25% overlap

        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))

            # Try to break at sentence or paragraph boundary
            if end < len(content):
                # Look for paragraph break
                para_break = content.rfind("\n\n", start, end)
                if para_break > start + chunk_size // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    sent_break = max(
                        content.rfind(". ", start, end),
                        content.rfind(".\n", start, end),
                        content.rfind("! ", start, end),
                        content.rfind("? ", start, end),
                    )
                    if sent_break > start + chunk_size // 2:
                        end = sent_break + 2

            chunk_content = content[start:end].strip()

            if chunk_content:  # Skip empty chunks
                chunk = DocumentChunk.create(
                    content=chunk_content,
                    metadata=metadata,
                    start_char=start,
                    end_char=end,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move to next chunk with overlap
            start = end - overlap if end < len(content) else end

        # Update total chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks