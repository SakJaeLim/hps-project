import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from .env
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

print("=" * 60)
print("  Database Connectivity Test: Neo4j & Neon (Postgres)")
print("=" * 60)
print(f"Loading environment from: {env_path}\n")

# -------------------------------------------------------------
# 1. Test Neo4j Connection
# -------------------------------------------------------------
print("[1/2] Testing Neo4j Connection...")
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "changeme")

print(f"  - Target URI: {neo4j_uri}")
print(f"  - User      : {neo4j_user}")

try:
    import neo4j
    driver = neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    driver.verify_connectivity()
    print("  -> SUCCESS: Successfully connected to Neo4j database!")
    driver.close()
except Exception as e:
    print(f"  -> FAILURE: Failed to connect to Neo4j.")
    print(f"     Error details: {e}")
    print("     (Tip: Make sure Neo4j is running locally or check credentials in .env)")

print("-" * 60)

# -------------------------------------------------------------
# 2. Test Neon (PostgreSQL) Connection
# -------------------------------------------------------------
print("[2/2] Testing Neon (Postgres) Connection...")
# Support standard DATABASE_URL or NEON_DATABASE_URL
db_url = os.getenv("DATABASE_URL") or os.getenv("NEON_DATABASE_URL")

if not db_url or db_url.startswith("postgres://user:password") or db_url.strip() == "":
    print("  -> SKIPPED: Connection details not found in .env.")
    print("     Please configure 'DATABASE_URL' in your .env file with your Neon connection string.")
    print("     Example: DATABASE_URL=postgres://user:pass@ep-host.region.aws.neon.tech/dbname?sslmode=require")
else:
    # Print target hostname for safety/verification without exposing password
    from urllib.parse import urlparse
    try:
        parsed = urlparse(db_url)
        print(f"  - Target Host: {parsed.hostname}")
        print(f"  - Database   : {parsed.path.lstrip('/')}")
        print(f"  - User       : {parsed.username}")
    except Exception:
        print("  - Target Host: [Failed to parse URL]")

    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        db_version = cur.fetchone()
        print(f"  -> SUCCESS: Successfully connected to Neon (PostgreSQL)!")
        print(f"     DB Version: {db_version[0]}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"  -> FAILURE: Failed to connect to Neon.")
        print(f"     Error details: {e}")
        print("     (Tip: Verify your DATABASE_URL and check internet connectivity / Neon service status)")

print("=" * 60)
