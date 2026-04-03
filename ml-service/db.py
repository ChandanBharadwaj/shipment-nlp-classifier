"""
Database connection helper.

Reads DATABASE_URL from the environment (or .env file) and returns a
psycopg2 connection with the pgvector type registered.

Usage:
    from db import get_connection
    conn = get_connection()

DATABASE_URL format:
    postgresql://user:password@host:port/dbname

TODO (production): Replace single connection with a ThreadedConnectionPool
     or switch to asyncpg for async FastAPI workers.
"""

import os

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

load_dotenv()


def get_connection() -> psycopg2.extensions.connection:
    """Return a psycopg2 connection with pgvector registered."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise EnvironmentError(
            "DATABASE_URL environment variable is not set. "
            "Copy ml-service/.env.example to ml-service/.env and fill in the value."
        )
    conn = psycopg2.connect(url)
    register_vector(conn)
    return conn
