from __future__ import annotations

import os
from typing import Optional
import psycopg2
from psycopg2.extras import execute_values


CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    source_file TEXT NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id           SERIAL PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_file  TEXT NOT NULL,
    page_number  INTEGER NOT NULL,
    chunk_index  INTEGER NOT NULL,
    content_type TEXT NOT NULL CHECK (content_type IN ('text', 'table', 'page_image')),
    content      TEXT NOT NULL,
    image_data   BYTEA,
    embedding    vector(768),
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_VECTOR_INDEX = """
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
ON chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
"""

CREATE_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    id              SERIAL PRIMARY KEY,
    request_id      TEXT NOT NULL,
    endpoint        TEXT NOT NULL,
    rating          SMALLINT CHECK (rating IN (-1, 0, 1)),
    corrected_fields JSONB,
    original_output  JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_REQUEST_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS request_log (
    id                  SERIAL PRIMARY KEY,
    request_id          TEXT NOT NULL UNIQUE,
    endpoint            TEXT NOT NULL,
    timestamp           TIMESTAMPTZ DEFAULT NOW(),
    latency_ms          FLOAT,
    input_size_bytes    INTEGER,
    json_valid          BOOLEAN,
    schema_violations   INTEGER DEFAULT 0,
    retrieval_score     FLOAT,
    null_field_count    INTEGER DEFAULT 0
);
"""


def get_connection(dsn: Optional[str] = None) -> psycopg2.extensions.connection:
    dsn = dsn or os.environ.get("PGVECTOR_DSN", "postgresql://postgres:password@localhost:5432/docint")
    return psycopg2.connect(dsn)


def initialize_schema(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(CREATE_DOCUMENTS_TABLE)
        cur.execute(CREATE_CHUNKS_TABLE)
        cur.execute(CREATE_FEEDBACK_TABLE)
        cur.execute(CREATE_REQUEST_LOG_TABLE)
    conn.commit()


def upsert_document(conn: psycopg2.extensions.connection, doc_id: str, source_file: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (id, source_file) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING;",
            (doc_id, source_file),
        )
    conn.commit()


def insert_chunks(conn: psycopg2.extensions.connection, chunks: list[dict]) -> None:
    rows = [
        (
            chunk["doc_id"],
            chunk["source_file"],
            chunk["page_number"],
            chunk["chunk_index"],
            chunk["content_type"],
            chunk["content"],
            chunk.get("image"),
            chunk.get("embedding"),
        )
        for chunk in chunks
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO chunks
                (doc_id, source_file, page_number, chunk_index, content_type, content, image_data, embedding)
            VALUES %s;
            """,
            rows,
        )
    conn.commit()


def similarity_search(
    conn: psycopg2.extensions.connection,
    query_embedding: list[float],
    top_k: int = 5,
    content_type_filter: Optional[str] = None,
    doc_id_filter: Optional[str] = None,
) -> list[dict]:
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    filters = []
    params = [embedding_str]

    if content_type_filter:
        filters.append(f"content_type = %s")
        params.append(content_type_filter)
    if doc_id_filter:
        filters.append(f"doc_id = %s")
        params.append(doc_id_filter)

    where_clause = "WHERE " + " AND ".join(filters) if filters else ""
    params.append(top_k)

    query = f"""
        SELECT
            id, doc_id, source_file, page_number, chunk_index,
            content_type, content,
            1 - (embedding <=> %s::vector) AS similarity_score
        FROM chunks
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    params.insert(1, embedding_str)

    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "doc_id": r[1],
            "source_file": r[2],
            "page_number": r[3],
            "chunk_index": r[4],
            "content_type": r[5],
            "content": r[6],
            "similarity_score": float(r[7]),
        }
        for r in rows
    ]


def hybrid_search(
    conn: psycopg2.extensions.connection,
    query_embedding: list[float],
    keyword: str,
    top_k: int = 5,
) -> list[dict]:
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    query = """
        SELECT
            id, doc_id, source_file, page_number, chunk_index,
            content_type, content,
            1 - (embedding <=> %s::vector) AS similarity_score,
            ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) AS keyword_score
        FROM chunks
        WHERE
            embedding IS NOT NULL
            AND to_tsvector('english', content) @@ plainto_tsquery('english', %s)
        ORDER BY (0.7 * (1 - (embedding <=> %s::vector)) + 0.3 * ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s))) DESC
        LIMIT %s;
    """
    params = [embedding_str, keyword, keyword, embedding_str, keyword, top_k]

    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "doc_id": r[1],
            "source_file": r[2],
            "page_number": r[3],
            "chunk_index": r[4],
            "content_type": r[5],
            "content": r[6],
            "similarity_score": float(r[7]),
            "keyword_score": float(r[8]),
        }
        for r in rows
    ]


def log_request(conn: psycopg2.extensions.connection, record: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO request_log
                (request_id, endpoint, latency_ms, input_size_bytes,
                 json_valid, schema_violations, retrieval_score, null_field_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (request_id) DO NOTHING;
            """,
            (
                record["request_id"],
                record["endpoint"],
                record.get("latency_ms"),
                record.get("input_size_bytes"),
                record.get("json_valid"),
                record.get("schema_violations", 0),
                record.get("retrieval_score"),
                record.get("null_field_count", 0),
            ),
        )
    conn.commit()


def log_feedback(conn: psycopg2.extensions.connection, record: dict) -> None:
    import json as _json
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO feedback
                (request_id, endpoint, rating, corrected_fields, original_output)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (
                record["request_id"],
                record["endpoint"],
                record.get("rating"),
                _json.dumps(record.get("corrected_fields")) if record.get("corrected_fields") else None,
                _json.dumps(record.get("original_output")) if record.get("original_output") else None,
            ),
        )
    conn.commit()
