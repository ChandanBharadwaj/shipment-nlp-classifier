"""
Database connection helper.

Reads DATABASE_URL from the environment (or .env file) and provides:
  - get_connection()  for one-shot CLI scripts (centroid_builder, fit_calibration,
                      evaluate_threshold, discover_unknowns, init_db)
  - get_pool()        for the FastAPI service (concurrent requests)

DATABASE_URL format:
    postgresql://user:password@host:port/dbname
"""

import os
from contextlib import contextmanager

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg2.pool import ThreadedConnectionPool

load_dotenv()

_pool: ThreadedConnectionPool | None = None


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise EnvironmentError(
            "DATABASE_URL environment variable is not set. "
            "Copy ml-service/.env.example to ml-service/.env and fill in the value."
        )
    return url


def get_connection() -> psycopg2.extensions.connection:
    """Return a standalone psycopg2 connection with pgvector registered.

    Use this in short-lived CLI scripts where a pool is overkill.
    Callers own the lifecycle and must call .close() themselves.
    """
    conn = psycopg2.connect(_db_url())
    register_vector(conn)
    return conn


def get_pool(minconn: int = 2, maxconn: int = 10) -> ThreadedConnectionPool:
    """Return the process-wide ThreadedConnectionPool, creating it lazily.

    Used by the FastAPI service so concurrent requests don't serialize on a
    single connection. Connections returned from this pool have pgvector
    registered on first checkout.
    """
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(minconn, maxconn, _db_url())
    return _pool


@contextmanager
def pooled_connection():
    """Context manager that checks a connection out of the pool and returns it.

        with pooled_connection() as conn:
            with conn.cursor() as cur:
                ...
    """
    pool = get_pool()
    conn = pool.getconn()
    # register_vector is idempotent per connection — safe to re-register.
    register_vector(conn)
    try:
        yield conn
    finally:
        pool.putconn(conn)


def close_pool() -> None:
    """Close all pool connections (call at shutdown)."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
