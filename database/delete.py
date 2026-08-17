import sqlite3

DB_PATH = "wildlife.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get every user-created table
cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    AND name NOT LIKE 'sqlite_%';
""")

tables = [row[0] for row in cursor.fetchall()]

print("Tables found:")
for table in tables:
    print(" -", table)

# Disable foreign-key constraints temporarily
cursor.execute("PRAGMA foreign_keys = OFF;")

for table in tables:
    cursor.execute(f'DELETE FROM "{table}";')

conn.commit()

# Re-enable foreign keys
cursor.execute("PRAGMA foreign_keys = ON;")

# Reclaim unused space
cursor.execute("VACUUM;")

conn.close()

print("\n✅ Database emptied successfully.")
print("Tables/schema were preserved.")