#!/bin/bash
# entrypoint.sh
# Runs on every container startup.
# Seeds database if empty, then starts FastAPI.

set -e

echo "Starting E-Commerce AI Support System..."

# wait for postgres to be ready
echo "Waiting for PostgreSQL..."
until python3 -c "
from database.connection import test_connection
import sys
sys.exit(0 if test_connection() else 1)
" 2>/dev/null; do
    echo "PostgreSQL not ready — retrying in 2s..."
    sleep 2
done

echo "PostgreSQL is ready"

# create tables and seed if empty
python3 -c "
from database.connection import SessionLocal, create_tables
from database.models import Product

create_tables()

db = SessionLocal()
count = db.query(Product).count()
db.close()

if count == 0:
    print('Database empty — seeding products and policies...')
    import subprocess
    result = subprocess.run(
        ['python3', '-m', 'database.import_kaggle_data'],
        check=True
    )
    print('Seeding complete')
else:
    print(f'Database has {count} products — skipping seed')
"

echo "Starting FastAPI..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
