-- Runs automatically on first container start via /docker-entrypoint-initdb.d/
-- Enables the pgvector extension so vector(384) columns work.
CREATE EXTENSION IF NOT EXISTS vector;
