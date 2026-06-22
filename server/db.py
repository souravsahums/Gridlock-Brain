"""DB helpers for the GridLock Brain API (psycopg 3)."""
import os
import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def get_conn():
    # sslmode=require enforced via the connection string
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)
