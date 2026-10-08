import hashlib
import math
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# =========================================================
# Paths / Environment
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


RAG_ROOT = PROJECT_ROOT / ".rag"

RAG_UPLOAD_DIR = RAG_ROOT / "uploads"

RAG_DB_PATH = RAG_ROOT / "rag.sqlite"


RAG_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

RAG_UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# RAG Settings
# =========================================================

EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small",
)

EMBEDDING_DIM = int(
    os.getenv(
        "RAG_EMBEDDING_DIM",
        "384",
    )
)


CHUNK_TARGET_CHARS = 1800

CHUNK_OVERLAP_CHARS = 300

MIN_CHUNK_CHARS = 100


EMBED_BATCH_SIZE = 32


DEFAULT_TOP_K = 6

MAX_TOP_K = 12


RRF_K = 60


# =========================================================
# Exceptions
# =========================================================


class RAGError(Exception):
    pass


# =========================================================
# Database
# =========================================================


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(
        RAG_DB_PATH,
        timeout=30,
    )

    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA journal_mode=WAL")

    connection.execute("PRAGMA foreign_keys=ON")

    return connection


def initialize_database():
    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_sha256 TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                extracted_page_count INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                character_count INTEGER NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(thread_id, file_sha256)
            )
            """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                created_at TEXT NOT NULL,

                FOREIGN KEY(document_id)
                    REFERENCES documents(document_id)
                    ON DELETE CASCADE,

                UNIQUE(document_id, chunk_index)
            )
            """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_documents_thread
            ON documents(thread_id)
            """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_chunks_document
            ON chunks(document_id)
            """)


initialize_database()


# =========================================================
# Gemini Embedding Client
# =========================================================


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load the local SentenceTransformer embedding model once.

    First run may download the model from Hugging Face.
    Later runs use the local cache.
    """

    try:
        return SentenceTransformer(
            EMBEDDING_MODEL
        )

    except Exception as exc:
        raise RAGError(
            f"Unable to load local embedding model: {exc}"
        ) from exc


# =========================================================
# General Helpers
# =========================================================


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(
    data: bytes,
) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_whitespace(
    text: str,
) -> str:
    text = text.replace(
        "\x00",
        " ",
    )

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n[ \t]+",
        "\n",
        text,
    )

    text = re.sub(
        r"[ \t]+\n",
        "\n",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def normalize_line_key(
    line: str,
) -> str:
    line = line.lower().strip()

    line = re.sub(
        r"\s+",
        " ",
        line,
    )

    return line


def safe_file_name(
    file_name: str,
) -> str:
    cleaned = Path(file_name).name

    cleaned = re.sub(
        r"[^A-Za-z0-9._()\- ]+",
        "_",
        cleaned,
    )

    cleaned = cleaned.strip()

    if not cleaned:
        cleaned = "document.pdf"

    return cleaned


def tokenize(
    text: str,
) -> list[str]:
    text = text.lower()

    return re.findall(
        r"[a-z0-9_]+|[\u0900-\u097f]+",
        text,
    )


def normalize_vector(
    vector,
) -> np.ndarray:
    array = np.asarray(
        vector,
        dtype=np.float32,
    )

    norm = np.linalg.norm(array)

    if norm == 0:
        return array

    return array / norm


def vector_to_blob(
    vector: np.ndarray,
) -> bytes:
    return vector.astype(np.float32).tobytes()


def blob_to_vector(
    blob: bytes,
    dimension: int,
) -> np.ndarray:
    vector = np.frombuffer(
        blob,
        dtype=np.float32,
        count=dimension,
    )

    return vector.copy()


# =========================================================
# Upload Helpers
# =========================================================


def original_upload_name(
    stored_name: str,
) -> str:
    """
    Convert internal hashed upload names such as
    396a0fadeb4cd40f_rag_survey.pdf
    into the original user-facing filename.
    """

    prefix, separator, original_name = (
        stored_name.partition("_")
    )

    if (
        separator
        and len(prefix) == 16
        and all(
            character in "0123456789abcdefABCDEF"
            for character in prefix
        )
        and original_name
    ):
        return original_name

    return stored_name


def save_uploaded_file(
    file_name: str,
    file_bytes: bytes,
) -> dict:

    if not file_bytes:
        raise RAGError("Uploaded file is empty.")

    cleaned_name = safe_file_name(file_name)

    if not cleaned_name.lower().endswith(".pdf"):
        raise RAGError("Only PDF files are supported right now.")

    file_hash = sha256_bytes(file_bytes)

    stored_name = f"{file_hash[:16]}_" f"{cleaned_name}"

    stored_path = RAG_UPLOAD_DIR / stored_name

    if not stored_path.exists():
        stored_path.write_bytes(file_bytes)

    return {
        "file_name": cleaned_name,
        "file_sha256": file_hash,
        "stored_path": str(stored_path),
    }


def validate_upload_path(
    file_path: str,
) -> Path:

    candidate = Path(file_path).resolve()

    upload_root = RAG_UPLOAD_DIR.resolve()

    try:
        candidate.relative_to(upload_root)

    except ValueError as exc:
        raise RAGError(
            "RAG can only index files stored " "inside the RAG uploads directory."
        ) from exc

    if not candidate.exists():
        raise RAGError("Uploaded PDF does not exist.")

    if not candidate.is_file():
        raise RAGError("Uploaded path is not a file.")

    if candidate.suffix.lower() != ".pdf":
        raise RAGError("Only PDF files are supported.")

    return candidate


# =========================================================
# PDF Extraction
# =========================================================


def extract_page_text(
    page,
) -> str:

    try:
        text = page.extract_text(extraction_mode="layout")

    except Exception:
        text = page.extract_text()

    return normalize_whitespace(text or "")


def is_page_number_line(
    line: str,
) -> bool:
    line = line.strip().lower()

    patterns = [
        r"^\d+$",
        r"^page\s+\d+$",
        r"^page\s+\d+\s+of\s+\d+$",
        r"^\d+\s*/\s*\d+$",
    ]

    return any(
        re.fullmatch(
            pattern,
            line,
        )
        for pattern in patterns
    )


def find_repeated_margin_lines(
    pages: list[str],
) -> set[str]:

    if len(pages) < 4:
        return set()

    candidates = []

    for text in pages:
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        if not lines:
            continue

        margin_lines = lines[:2] + lines[-2:]

        for line in margin_lines:

            if len(line) < 3:
                continue

            if is_page_number_line(line):
                continue

            key = normalize_line_key(line)

            candidates.append(key)

    counts = Counter(candidates)

    minimum_occurrences = max(
        3,
        math.ceil(len(pages) * 0.65),
    )

    return {line for line, count in counts.items() if count >= minimum_occurrences}


def remove_repeated_margins(
    text: str,
    repeated_lines: set[str],
) -> str:

    if not text:
        return ""

    cleaned_lines = []

    for line in text.splitlines():

        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        if is_page_number_line(stripped):
            continue

        key = normalize_line_key(stripped)

        if key in repeated_lines:
            continue

        cleaned_lines.append(stripped)

    return normalize_whitespace("\n".join(cleaned_lines))


def extract_pdf_pages(
    pdf_path: Path,
) -> dict:

    try:
        reader = PdfReader(str(pdf_path))

    except Exception as exc:
        raise RAGError(f"Unable to open PDF: {exc}") from exc

    raw_pages = []

    for page in reader.pages:

        try:
            text = extract_page_text(page)

        except Exception:
            text = ""

        raw_pages.append(text)

    repeated_lines = find_repeated_margin_lines(raw_pages)

    pages = []

    low_text_pages = []

    for index, raw_text in enumerate(
        raw_pages,
        start=1,
    ):

        text = remove_repeated_margins(
            raw_text,
            repeated_lines,
        )

        if len(text) < 40:
            low_text_pages.append(index)

        pages.append(
            {
                "page_number": index,
                "text": text,
            }
        )

    extracted_pages = [page for page in pages if page["text"].strip()]

    if not extracted_pages:
        raise RAGError(
            "No readable text was extracted from this PDF. "
            "The document may be scanned/image-only and "
            "will require OCR support."
        )

    return {
        "pages": pages,
        "page_count": len(raw_pages),
        "extracted_page_count": len(extracted_pages),
        "low_text_pages": (low_text_pages),
    }


# =========================================================
# Smart Chunking
# =========================================================


def split_long_text(
    text: str,
    target_chars: int,
) -> list[str]:

    text = text.strip()

    if len(text) <= target_chars:
        return [text]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    pieces = []

    current = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) <= target_chars:
            current = candidate
            continue

        if current:
            pieces.append(current)

        if len(sentence) <= target_chars:
            current = sentence
            continue

        words = sentence.split()

        word_buffer = ""

        for word in words:

            candidate = f"{word_buffer} {word}".strip()

            if len(candidate) <= target_chars:
                word_buffer = candidate
            else:

                if word_buffer:
                    pieces.append(word_buffer)

                word_buffer = word

        current = word_buffer

    if current:
        pieces.append(current)

    return pieces


def get_overlap_text(
    text: str,
    overlap_chars: int,
) -> str:

    if len(text) <= overlap_chars:
        return text

    overlap = text[-overlap_chars:]

    first_space = overlap.find(" ")

    if first_space >= 0:
        overlap = overlap[first_space + 1 :]

    return overlap.strip()


def chunk_page_text(
    text: str,
) -> list[str]:

    text = normalize_whitespace(text)

    if not text:
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            text,
        )
        if paragraph.strip()
    ]

    if not paragraphs:
        paragraphs = [text]

    units = []

    for paragraph in paragraphs:
        units.extend(
            split_long_text(
                paragraph,
                CHUNK_TARGET_CHARS,
            )
        )

    chunks = []

    current = ""

    for unit in units:

        candidate = f"{current}\n\n{unit}" if current else unit

        if len(candidate) <= CHUNK_TARGET_CHARS:
            current = candidate
            continue

        if len(current) >= MIN_CHUNK_CHARS:
            chunks.append(current.strip())

        overlap = get_overlap_text(
            current,
            CHUNK_OVERLAP_CHARS,
        )

        current = f"{overlap}\n\n{unit}".strip() if overlap else unit

        if len(current) > CHUNK_TARGET_CHARS:
            split_parts = split_long_text(
                current,
                CHUNK_TARGET_CHARS,
            )

            if len(split_parts) > 1:

                chunks.extend(
                    part.strip()
                    for part in split_parts[:-1]
                    if len(part.strip()) >= MIN_CHUNK_CHARS
                )

                current = split_parts[-1]

    if current.strip() and len(current.strip()) >= MIN_CHUNK_CHARS:
        chunks.append(current.strip())

    if not chunks and text:
        chunks.append(text)

    return chunks


def build_chunks(
    pages: list[dict],
) -> list[dict]:

    chunks = []

    chunk_index = 0

    for page in pages:

        page_number = page["page_number"]

        page_text = page["text"]

        page_chunks = chunk_page_text(page_text)

        for chunk_text in page_chunks:

            chunks.append(
                {
                    "chunk_index": (chunk_index),
                    "page_number": (page_number),
                    "text": (chunk_text),
                }
            )

            chunk_index += 1

    return chunks


# =========================================================
# Embeddings
# =========================================================


def prepare_document_embedding_text(
    text: str,
) -> str:
    return f"passage: {text}"


def prepare_query_embedding_text(
    query: str,
) -> str:
    return f"query: {query}"


def embed_text_batch(
    texts: list[str],
    mode: str,
) -> list[np.ndarray]:

    if not texts:
        return []

    if mode not in {
        "document",
        "query",
    }:
        raise RAGError(
            f"Unsupported embedding mode: {mode}"
        )

    if mode == "document":
        prepared = [
            prepare_document_embedding_text(text)
            for text in texts
        ]

    else:
        prepared = [
            prepare_query_embedding_text(text)
            for text in texts
        ]

    model = get_embedding_model()

    try:
        embeddings = model.encode(
            prepared,
            batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    except Exception as exc:
        raise RAGError(
            f"Local embedding failed: {exc}"
        ) from exc

    vectors = []

    for embedding in embeddings:

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if len(vector) != EMBEDDING_DIM:
            raise RAGError(
                "Unexpected embedding dimension. "
                f"Expected {EMBEDDING_DIM}, "
                f"received {len(vector)}."
            )

        vectors.append(
            vector
        )

    return vectors


def embed_documents(
    texts: list[str],
) -> list[np.ndarray]:

    all_vectors = []

    for start in range(
        0,
        len(texts),
        EMBED_BATCH_SIZE,
    ):
        batch = texts[
            start:
            start + EMBED_BATCH_SIZE
        ]

        vectors = embed_text_batch(
            batch,
            mode="document",
        )

        all_vectors.extend(
            vectors
        )

    return all_vectors


def embed_query(
    query: str,
) -> np.ndarray:

    vectors = embed_text_batch(
        [query],
        mode="query",
    )

    return vectors[0]


# =========================================================
# Document Indexing
# =========================================================


def create_document_id(
    thread_id: str,
    file_hash: str,
) -> str:

    value = f"{thread_id}:" f"{file_hash}"

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def index_pdf(
    file_path: str,
    thread_id: str,
) -> dict:

    if not thread_id.strip():
        return {
            "success": False,
            "error": ("thread_id is required"),
        }

    try:
        pdf_path = validate_upload_path(file_path)

        display_file_name = original_upload_name(
            pdf_path.name
        )

        file_bytes = pdf_path.read_bytes()

        file_hash = sha256_bytes(file_bytes)

        document_id = create_document_id(
            thread_id,
            file_hash,
        )

        with get_connection() as connection:

            existing = connection.execute(
                """
                SELECT
                    document_id,
                    file_name,
                    page_count,
                    chunk_count
                FROM documents
                WHERE thread_id = ?
                  AND file_sha256 = ?
                """,
                (
                    thread_id,
                    file_hash,
                ),
            ).fetchone()

        if existing:
            return {
                "success": True,
                "already_indexed": True,
                "document_id": (existing["document_id"]),
                "file_name": (existing["file_name"]),
                "page_count": (existing["page_count"]),
                "chunk_count": (existing["chunk_count"]),
            }

        extraction = extract_pdf_pages(pdf_path)

        chunks = build_chunks(extraction["pages"])

        if not chunks:
            return {
                "success": False,
                "error": ("No usable text chunks " "were created from the PDF."),
            }

        chunk_texts = [chunk["text"] for chunk in chunks]

        embeddings = embed_documents(chunk_texts)

        character_count = sum(len(page["text"]) for page in extraction["pages"])

        created_at = utc_now()

        with get_connection() as connection:

            connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    thread_id,
                    file_name,
                    file_sha256,
                    stored_path,
                    page_count,
                    extracted_page_count,
                    chunk_count,
                    character_count,
                    embedding_model,
                    embedding_dim,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    thread_id,
                    display_file_name,
                    file_hash,
                    str(pdf_path),
                    extraction["page_count"],
                    extraction["extracted_page_count"],
                    len(chunks),
                    character_count,
                    EMBEDDING_MODEL,
                    EMBEDDING_DIM,
                    created_at,
                ),
            )

            for chunk, embedding in zip(
                chunks,
                embeddings,
            ):

                connection.execute(
                    """
                    INSERT INTO chunks (
                        document_id,
                        chunk_index,
                        page_number,
                        text,
                        embedding,
                        embedding_dim,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        chunk["chunk_index"],
                        chunk["page_number"],
                        chunk["text"],
                        vector_to_blob(embedding),
                        EMBEDDING_DIM,
                        created_at,
                    ),
                )

            connection.commit()

        warning = None

        low_text_pages = extraction["low_text_pages"]

        if low_text_pages:

            ratio = len(low_text_pages) / max(
                extraction["page_count"],
                1,
            )

            if ratio >= 0.20:
                warning = (
                    "Some pages contain little or no extractable "
                    "text. The PDF may contain scanned/image pages. "
                    "OCR support will be needed for those pages."
                )

        return {
            "success": True,
            "already_indexed": False,
            "document_id": (document_id),
            "file_name": (display_file_name),
            "page_count": (extraction["page_count"]),
            "extracted_page_count": (extraction["extracted_page_count"]),
            "chunk_count": len(chunks),
            "character_count": (character_count),
            "embedding_model": (EMBEDDING_MODEL),
            "embedding_dim": (EMBEDDING_DIM),
            "low_text_pages": (low_text_pages),
            "warning": warning,
        }

    except RAGError as exc:
        return {
            "success": False,
            "error": str(exc),
        }

    except Exception as exc:
        return {
            "success": False,
            "error": (f"Unable to index PDF: {exc}"),
        }


# =========================================================
# Document Listing
# =========================================================


def list_documents(
    thread_id: str,
) -> dict:

    if not thread_id.strip():
        return {
            "success": False,
            "error": ("thread_id is required"),
        }

    with get_connection() as connection:

        rows = connection.execute(
            """
            SELECT
                document_id,
                file_name,
                page_count,
                extracted_page_count,
                chunk_count,
                character_count,
                embedding_model,
                created_at
            FROM documents
            WHERE thread_id = ?
            ORDER BY created_at DESC
            """,
            (thread_id,),
        ).fetchall()

    documents = [
        {
            "document_id": (row["document_id"]),
            "file_name": (row["file_name"]),
            "page_count": (row["page_count"]),
            "extracted_page_count": (row["extracted_page_count"]),
            "chunk_count": (row["chunk_count"]),
            "character_count": (row["character_count"]),
            "embedding_model": (row["embedding_model"]),
            "created_at": (row["created_at"]),
        }
        for row in rows
    ]

    return {
        "success": True,
        "count": len(documents),
        "documents": (documents),
    }


# =========================================================
# Search Loading
# =========================================================


def load_search_chunks(
    thread_id: str,
    document_ids: list[str] | None,
) -> list[dict]:

    parameters = [thread_id]

    where = "d.thread_id = ?"

    if document_ids:

        placeholders = ",".join("?" for _ in document_ids)

        where += f" AND d.document_id " f"IN ({placeholders})"

        parameters.extend(document_ids)

    query = f"""
        SELECT
            c.id,
            c.document_id,
            c.chunk_index,
            c.page_number,
            c.text,
            c.embedding,
            c.embedding_dim,
            d.file_name
        FROM chunks c
        JOIN documents d
          ON d.document_id = c.document_id
        WHERE {where}
        ORDER BY
            c.document_id,
            c.chunk_index
    """

    with get_connection() as connection:

        rows = connection.execute(
            query,
            parameters,
        ).fetchall()

    return [
        {
            "id": row["id"],
            "document_id": (row["document_id"]),
            "file_name": (row["file_name"]),
            "chunk_index": (row["chunk_index"]),
            "page_number": (row["page_number"]),
            "text": (row["text"]),
            "embedding": (
                blob_to_vector(
                    row["embedding"],
                    row["embedding_dim"],
                )
            ),
        }
        for row in rows
    ]


# =========================================================
# Retrieval Scoring
# =========================================================


def min_max_scores(
    values: np.ndarray,
) -> np.ndarray:

    if len(values) == 0:
        return values

    minimum = float(np.min(values))

    maximum = float(np.max(values))

    if math.isclose(
        minimum,
        maximum,
    ):
        return np.ones_like(
            values,
            dtype=np.float32,
        )

    return (values - minimum) / (maximum - minimum)


def rank_positions(
    scores: np.ndarray,
) -> dict[int, int]:

    order = np.argsort(-scores)

    return {
        int(index): rank
        for rank, index in enumerate(
            order,
            start=1,
        )
    }


def reciprocal_rank_fusion(
    dense_scores: np.ndarray,
    lexical_scores: np.ndarray,
) -> np.ndarray:

    dense_ranks = rank_positions(dense_scores)

    lexical_ranks = rank_positions(lexical_scores)

    result = np.zeros(
        len(dense_scores),
        dtype=np.float32,
    )

    for index in range(len(result)):

        dense_rank = dense_ranks[index]

        lexical_rank = lexical_ranks[index]

        result[index] = 1.0 / (RRF_K + dense_rank) + 1.0 / (RRF_K + lexical_rank)

    return min_max_scores(result)


def keyword_coverage_score(
    query_tokens: list[str],
    text: str,
) -> float:

    unique_query = set(query_tokens)

    if not unique_query:
        return 0.0

    text_tokens = set(tokenize(text))

    matched = unique_query & text_tokens

    return len(matched) / len(unique_query)


def calculate_hybrid_scores(
    query: str,
    chunks: list[dict],
    query_vector: np.ndarray,
) -> np.ndarray:

    matrix = np.stack([chunk["embedding"] for chunk in chunks])

    dense_scores = matrix @ query_vector

    dense_normalized = min_max_scores(dense_scores)

    corpus_tokens = [tokenize(chunk["text"]) for chunk in chunks]

    query_tokens = tokenize(query)

    bm25 = BM25Okapi(corpus_tokens)

    lexical_scores = np.asarray(
        bm25.get_scores(query_tokens),
        dtype=np.float32,
    )

    lexical_normalized = min_max_scores(lexical_scores)

    rrf_scores = reciprocal_rank_fusion(
        dense_scores,
        lexical_scores,
    )

    coverage_scores = np.asarray(
        [
            keyword_coverage_score(
                query_tokens,
                chunk["text"],
            )
            for chunk in chunks
        ],
        dtype=np.float32,
    )

    query_lower = query.strip().lower()

    exact_phrase_scores = np.asarray(
        [
            1.0 if (query_lower and query_lower in chunk["text"].lower()) else 0.0
            for chunk in chunks
        ],
        dtype=np.float32,
    )

    final_scores = (
        0.50 * dense_normalized
        + 0.22 * lexical_normalized
        + 0.18 * rrf_scores
        + 0.08 * coverage_scores
        + 0.02 * exact_phrase_scores
    )

    return final_scores


# =========================================================
# MMR Reranking
# =========================================================


def mmr_rerank(
    chunks: list[dict],
    hybrid_scores: np.ndarray,
    top_k: int,
) -> list[int]:

    candidate_limit = min(
        len(chunks),
        max(
            top_k * 5,
            20,
        ),
    )

    candidate_indices = list(np.argsort(-hybrid_scores)[:candidate_limit])

    selected = []

    lambda_relevance = 0.82

    while candidate_indices and len(selected) < top_k:

        best_index = None

        best_score = -float("inf")

        for index in candidate_indices:

            relevance = float(hybrid_scores[index])

            redundancy = 0.0

            if selected:

                candidate_vector = chunks[index]["embedding"]

                redundancy = max(
                    float(candidate_vector @ chunks[selected_index]["embedding"])
                    for selected_index in selected
                )

            mmr_score = (
                lambda_relevance * relevance - (1.0 - lambda_relevance) * redundancy
            )

            if mmr_score > best_score:
                best_score = mmr_score

                best_index = index

        selected.append(best_index)

        candidate_indices.remove(best_index)

    return selected


# =========================================================
# Context Expansion
# =========================================================


def get_neighbor_context(
    target: dict,
    all_chunks: list[dict],
) -> str:

    neighbors = []

    for chunk in all_chunks:

        if chunk["document_id"] != target["document_id"]:
            continue

        if chunk["page_number"] != target["page_number"]:
            continue

        distance = abs(chunk["chunk_index"] - target["chunk_index"])

        if distance <= 1:
            neighbors.append(chunk)

    neighbors.sort(key=lambda item: (item["chunk_index"]))

    texts = []

    seen = set()

    for chunk in neighbors:

        text = chunk["text"].strip()

        if not text or text in seen:
            continue

        seen.add(text)

        texts.append(text)

    context = "\n\n".join(texts)

    return context[:5000]


# =========================================================
# Hybrid Search
# =========================================================


def search_documents(
    query: str,
    thread_id: str,
    document_ids: list[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict:

    query = query.strip()

    thread_id = thread_id.strip()

    if not query:
        return {
            "success": False,
            "error": ("Search query is required."),
        }

    if not thread_id:
        return {
            "success": False,
            "error": ("thread_id is required."),
        }

    if top_k < 1 or top_k > MAX_TOP_K:
        return {
            "success": False,
            "error": (f"top_k must be between " f"1 and {MAX_TOP_K}."),
        }

    try:
        chunks = load_search_chunks(
            thread_id,
            document_ids,
        )

        if not chunks:
            return {
                "success": True,
                "query": query,
                "count": 0,
                "results": [],
                "message": (
                    "No indexed documents were found " "for this conversation."
                ),
            }

        query_vector = embed_query(query)

        hybrid_scores = calculate_hybrid_scores(
            query,
            chunks,
            query_vector,
        )

        selected_indices = mmr_rerank(
            chunks,
            hybrid_scores,
            top_k,
        )

        results = []

        for rank, index in enumerate(
            selected_indices,
            start=1,
        ):

            chunk = chunks[index]

            page_number = chunk["page_number"]

            file_name = chunk["file_name"]

            context = get_neighbor_context(
                chunk,
                chunks,
            )

            results.append(
                {
                    "rank": rank,
                    "document_id": (chunk["document_id"]),
                    "file_name": (file_name),
                    "page_number": (page_number),
                    "chunk_index": (chunk["chunk_index"]),
                    "score": round(
                        float(hybrid_scores[index]),
                        6,
                    ),
                    "citation": (f"{file_name}, " f"page {page_number}"),
                    "text": (chunk["text"]),
                    "context": (context),
                }
            )

        return {
            "success": True,
            "query": query,
            "count": len(results),
            "retrieval": ("hybrid_dense_bm25_rrf_mmr"),
            "results": results,
        }

    except RAGError as exc:
        return {
            "success": False,
            "error": str(exc),
        }

    except Exception as exc:
        return {
            "success": False,
            "error": (f"Document search failed: {exc}"),
        }


# =========================================================
# Delete Document
# =========================================================


def delete_document(
    document_id: str,
    thread_id: str,
) -> dict:

    document_id = document_id.strip()

    thread_id = thread_id.strip()

    if not document_id:
        return {
            "success": False,
            "error": ("document_id is required."),
        }

    if not thread_id:
        return {
            "success": False,
            "error": ("thread_id is required."),
        }

    with get_connection() as connection:

        row = connection.execute(
            """
            SELECT
                stored_path,
                file_name
            FROM documents
            WHERE document_id = ?
              AND thread_id = ?
            """,
            (
                document_id,
                thread_id,
            ),
        ).fetchone()

        if not row:
            return {
                "success": False,
                "error": ("Document was not found " "in this conversation."),
            }

        connection.execute(
            """
            DELETE FROM documents
            WHERE document_id = ?
              AND thread_id = ?
            """,
            (
                document_id,
                thread_id,
            ),
        )

        connection.commit()

    return {
        "success": True,
        "message": ("Document removed from " "the RAG index."),
        "document_id": (document_id),
        "file_name": (row["file_name"]),
    }


# =========================================================
# RAG Status
# =========================================================


def get_rag_status() -> dict:

    with get_connection() as connection:

        document_count = connection.execute("""
                SELECT COUNT(*)
                FROM documents
                """).fetchone()[0]

        chunk_count = connection.execute("""
                SELECT COUNT(*)
                FROM chunks
                """).fetchone()[0]

    return {
        "success": True,
        "database": str(RAG_DB_PATH),
        "uploads_directory": str(RAG_UPLOAD_DIR),
        "embedding_model": (EMBEDDING_MODEL),
        "embedding_dimension": (EMBEDDING_DIM),
        "documents": (document_count),
        "chunks": (chunk_count),
        "retrieval": ("Dense + BM25 + RRF + MMR"),
    }
